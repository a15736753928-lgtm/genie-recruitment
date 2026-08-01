"""Resume management — HTTP route layer.

Business logic has been extracted to:
- ``resume_parser.py``     — file extraction, LLM parsing, age inference
- ``resume_serializer.py`` — ORM → dict conversion, display helpers
- ``resume_upload.py``     — the full upload pipeline (shared by routes + Agent)
"""

import os
import json
import logging
import asyncio
from typing import Optional, List

from fastapi import APIRouter, Depends, File, Form, UploadFile, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func, desc, asc
from sqlalchemy.orm import selectinload

from app.database import get_db, async_session_factory
from app.core.state_machine import transition, StateError
from app.core.security import CurrentUser, require_permission
from app.models.recruitment import (
    Candidate, Position, CandidateSkill, CandidateEducation,
    CandidateWorkExperience, CandidateProjectExperience,
)
from app.config import get_settings
from app.infrastructure import minio_storage
from app.services.system.system_settings import get_system_setting

# Re-exported from split modules for backward compatibility
from app.api.recruitment.resume_parser import (
    extract_text_from_file,
    parse_resume_with_llm,
    MIME_TYPES,
)
from app.api.recruitment.resume_serializer import (
    UNKNOWN, IN_SCHOOL, DEFAULT_ETHNICITY,
    serialize_candidate, is_candidate_parsed,
    normalize_ethnicity, normalize_experience, display_text,
)
from app.api.recruitment.resume_upload import (
    _upload_one_resume,
    _run_parse as run_resume_parse,
    _load_candidate as load_candidate,
    _extract_and_parse as extract_and_parse_resume,
    fill_candidate_from_parsed,
)
from app.utils.responses import ok, fail, not_found, conflict

router = APIRouter(tags=["简历"])
settings = get_settings()
logger = logging.getLogger(__name__)

os.makedirs(settings.upload_dir, exist_ok=True)

# ── 重新解析异步任务 ──────────────────────────────────────────
# reanalyze 会串行调用多次 LLM（视觉抽取 → LLM 解析 → 8 维评分），单次请求
# 动辄 1~2 分钟，远超前端默认超时（HTTP 层 60s / 前端 120s）。改为后台任务 +
# 轮询状态：POST /resumes/{id}/reanalyze 立即返回任务状态，前端轮询
# /resumes/{id}/reanalyze-status 直到 completed/failed。
#
# 任务状态放在进程内存单例表。这是低频显式操作，单 worker 下可靠；即使进程
# 重启导致任务状态丢失，只要候选人已成功落库解析+评分，前端会兜底用现有数据。
# 不落库避免了为加列而 rebuild 数据库（会清空全家数据）。
_REANALYZE_TASKS: dict[str, dict] = {}
_REANALYZE_TASK_LOCK = asyncio.Lock()


def _reanalyze_task_state(resume_id: str) -> dict | None:
    return _REANALYZE_TASKS.get(str(resume_id))


async def _set_reanalyze_task(resume_id: str, **fields) -> None:
    """更新任务状态并提交到内存表（带锁防止并发写覆盖）。"""
    key = str(resume_id)
    async with _REANALYZE_TASK_LOCK:
        state = _REANALYZE_TASKS.setdefault(
            key, {"status": "pending", "progress": 0, "stage": "排队中", "message": ""}
        )
        state.update(fields)


async def _run_reanalyze_in_background(resume_id: str, position_name: str) -> None:
    """后台执行重新解析 + 8 维评分，用独立异步会话（请求会话已随响应关闭）。

    任意一步失败都落 failed 状态，避免前端一直轮询。
    """
    key = str(resume_id)
    try:
        async with async_session_factory() as task_db:
            await _set_reanalyze_task(resume_id, status="running", progress=10, stage="读取候选人")
            result = await task_db.execute(
                select(Candidate)
                .options(*CANDIDATE_LOAD_OPTIONS)
                .where(Candidate.id == resume_id)
            )
            candidate = result.scalar_one_or_none()
            if not candidate:
                await _set_reanalyze_task(resume_id, status="failed", progress=100, message="候选人不存在")
                return
            if not candidate.resume_file:
                await _set_reanalyze_task(resume_id, status="failed", progress=100, message="该候选人没有上传简历文件")
                return

            await _set_reanalyze_task(resume_id, status="running", progress=30, stage="AI 解析简历中")
            error = await run_resume_parse(
                candidate, position_name or (candidate.position.name if candidate.position else ""), task_db
            )
            if error and not is_candidate_parsed(candidate):
                await _set_reanalyze_task(resume_id, status="failed", progress=100, message=f"解析失败: {error}")
                return
            await task_db.flush()

            # 重新解析后自动触发 8 维权威评分
            await _set_reanalyze_task(resume_id, status="running", progress=75, stage="8 维 AI 评分中")
            try:
                from app.api.recruitment.scoring import auto_score_after_upload
                await auto_score_after_upload(
                    task_db, candidate,
                    position_id=candidate.position_id,
                )
            except Exception as e:
                logger.warning("后台 reanalyze 自动 8 维评分失败: %s", e, exc_info=True)

            await task_db.commit()
            await _set_reanalyze_task(resume_id, status="completed", progress=100, stage="已完成")
            logger.info("后台 reanalyze 完成 candidate=%s", resume_id)

    except Exception as exc:
        logger.error("后台 reanalyze 异常 candidate=%s: %s", resume_id, exc, exc_info=True)
        await _set_reanalyze_task(resume_id, status="failed", progress=100, message=f"后台处理异常: {exc}")


CANDIDATE_LOAD_OPTIONS = (
    selectinload(Candidate.position),
    selectinload(Candidate.skills),
    selectinload(Candidate.educations),
    selectinload(Candidate.work_experiences),
    selectinload(Candidate.project_experiences),
    selectinload(Candidate.ai_analysis),
)

# Lighter load for list queries (avoids 6 eager-loaded relationships)
CANDIDATE_LIST_OPTIONS = (
    selectinload(Candidate.position),
    selectinload(Candidate.skills),
)


async def get_auto_parse_setting(db: AsyncSession) -> bool:
    return bool(await get_system_setting(db, "autoParseResume", True))


# ── Shared query logic (HTTP routes + agent tools) ───────

async def query_candidate_list(
    db: AsyncSession,
    *,
    position_id: str = "all",
    statuses: str = "",
    keyword: str = "",
    gender: str = "",
    education: str = "",
    sort_by: str = "uploadTime",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 10,
    min_score: Optional[int] = None,
) -> dict:
    """List candidates with filters. Use plain Python defaults — safe for direct calls."""
    query = select(Candidate).options(*CANDIDATE_LOAD_OPTIONS)

    if position_id and position_id != "all":
        query = query.where(Candidate.position_id == position_id)

    score_threshold = min_score
    if score_threshold is None:
        score_threshold = int(await get_system_setting(db, "minMatchScore", 70) or 70)
    if min_score is not None:
        query = query.where(Candidate.score >= score_threshold)

    if statuses:
        status_list = [s.strip() for s in statuses.split(",") if s.strip()]
        if status_list:
            query = query.where(Candidate.status.in_(status_list))

    if keyword:
        kw = f"%{keyword}%"
        query = query.outerjoin(Position, Candidate.position_id == Position.id).where(
            or_(
                Candidate.name.ilike(kw),
                Candidate.skills.any(CandidateSkill.skill.ilike(kw)),
                Position.name.ilike(kw),
            )
        )

    if gender:
        if gender == "未知":
            query = query.where(
                or_(
                    Candidate.gender.is_(None),
                    Candidate.gender == "",
                    Candidate.gender == "未知",
                    Candidate.gender.notin_(["男", "女"]),
                )
            )
        else:
            query = query.where(Candidate.gender == gender)

    if education:
        if education == "未知":
            query = query.where(
                or_(
                    Candidate.education.is_(None),
                    Candidate.education == "",
                    Candidate.education == "未知",
                )
            )
        else:
            query = query.where(Candidate.education == education)

    count_query = select(func.count()).select_from(
        query.with_only_columns(Candidate.id).order_by(None).distinct().subquery()
    )
    total = (await db.execute(count_query)).scalar() or 0

    sort_col = Candidate.upload_time if sort_by == "uploadTime" else Candidate.score
    if sort_order == "asc":
        query = query.order_by(asc(sort_col))
    else:
        query = query.order_by(desc(sort_col))

    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    candidates = result.unique().scalars().all()

    return ok({
          "list": [serialize_candidate(c) for c in candidates],
          "total": total,
          "page": page,
          "pageSize": page_size,
      })


# ── Endpoints ───────────────────────────────────────────

@router.get("/resumes")
async def list_resumes(
    positionId: str = Query("all"),
    statuses: str = Query(""),
    keyword: str = Query(""),
    gender: str = Query(""),
    education: str = Query(""),
    sortBy: str = Query("uploadTime"),
    sortOrder: str = Query("desc"),
    page: int = Query(1),
    pageSize: int = Query(10),
    minScore: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    return await query_candidate_list(
        db,
        position_id=positionId,
        statuses=statuses,
        keyword=keyword,
        gender=gender,
        education=education,
        sort_by=sortBy,
        sort_order=sortOrder,
        page=page,
        page_size=pageSize,
        min_score=minScore,
    )


@router.get("/resumes/{resume_id}/file")
async def get_resume_file(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == resume_id))
    candidate = result.scalar_one_or_none()
    if not candidate or not candidate.resume_file:
        return JSONResponse(
            status_code=404,
            content=not_found("简历文件不存在"),
        )

    stored = candidate.resume_file
    filename = os.path.basename(stored) or stored

    from app.services.rag.utils import content_disposition_header
    cd_header = content_disposition_header(filename, "inline")

    # Legacy on-disk path — serve directly
    if os.path.isabs(stored) and os.path.exists(stored):
        ext = os.path.splitext(stored)[1].lower()
        media_type = MIME_TYPES.get(ext, "application/octet-stream")
        return FileResponse(
            stored,
            media_type=media_type,
            # 不传 filename：避免 Starlette 自动设置 Content-Disposition: attachment
            headers={"Content-Disposition": cd_header},
        )

    # MinIO object key — stream from object storage
    try:
        response = await asyncio.to_thread(minio_storage.get_object_stream, stored)
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content=not_found(f"简历文件不存在: {e}"),
        )

    ext = os.path.splitext(filename)[1].lower()
    media_type = MIME_TYPES.get(ext, "application/octet-stream")

    def _iter():
        try:
            for chunk in response.stream(amt=64 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    return StreamingResponse(
        _iter(),
        media_type=media_type,
        headers={"Content-Disposition": cd_header},
    )


@router.get("/resumes/{resume_id}")
async def get_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")
    data = serialize_candidate(candidate)
    # 顺带返回 scoreDetail，避免前端单独请求 score 时遇到401
    try:
        from app.models.phase1 import ResumeScore
        from app.api.recruitment.scoring import serialize_score
        score_row = await db.execute(
            select(ResumeScore).where(ResumeScore.candidate_id == candidate.id)
            .order_by(ResumeScore.created_at.desc()).limit(1)
        )
        score_obj = score_row.scalar_one_or_none()
        if score_obj:
            data["scoreDetail"] = serialize_score(score_obj)
    except Exception:
        pass
    return ok(data)


@router.post("/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    positionId: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    original_name = file.filename or "resume.pdf"
    file_ext = os.path.splitext(original_name)[1].lower() or ".pdf"
    content = await file.read()

    result = await _upload_one_resume(
        db,
        original_name=original_name,
        content=content,
        file_ext=file_ext,
        position_id=positionId if positionId else None,
    )

    if result["status"] == "success":
        response_data = result["data"]
        return ok(response_data, message=result["message"])
    return fail(result["statusCode"], result["message"], result.get("data"))


@router.post("/resumes/batch-upload")
async def batch_upload_resumes(
    files: List[UploadFile] = File(...),
    positionId: Optional[str] = Form(None),
):
    """批量上传简历，并发处理（最多 5 份并发），SSE 流式推送每份文件的处理进度。"""
    position_id = positionId if positionId else None

    # 先读取所有文件内容
    file_payloads: list[dict] = []
    for f in files:
        original_name = f.filename or "resume.pdf"
        file_ext = os.path.splitext(original_name)[1].lower() or ".pdf"
        content = await f.read()
        file_payloads.append({
            "original_name": original_name,
            "content": content,
            "file_ext": file_ext,
        })

    sem = asyncio.Semaphore(5)

    async def _process_one(payload: dict) -> dict:
        async with sem:
            async with async_session_factory() as task_db:
                try:
                    result = await _upload_one_resume(
                        task_db,
                        original_name=payload["original_name"],
                        content=payload["content"],
                        file_ext=payload["file_ext"],
                        position_id=position_id,
                    )
                    await task_db.commit()
                    return {
                        "fileName": payload["original_name"],
                        "status": result["status"],
                        "message": result["message"],
                        "candidateId": result["data"]["id"] if result["data"] and result["status"] == "success" else None,
                        "position": result["data"]["position"] if result["data"] and result["status"] == "success" else None,
                        "score": result["data"]["score"] if result["data"] and result["status"] == "success" else None,
                        "parseStatus": result["data"]["parseStatus"] if result["data"] and result["status"] == "success" else None,
                        "existingCandidateName": result.get("existingCandidateName"),
                    }
                except Exception:
                    await task_db.rollback()
                    raise

    total = len(file_payloads)

    async def _stream():
        items = []
        success = 0
        failed = 0
        skipped = 0
        invalid = 0
        done = 0

        # 用 task→payload 映射 + asyncio.wait(FIRST_COMPLETED)。
        # 不能用 asyncio.as_completed()：它 yield 的是新的包装协程(_wait_for_one)，
        # 不是原始 task，拿去查 pending 字典会 KeyError。
        pending = {
            asyncio.ensure_future(_process_one(p)): p
            for p in file_payloads
        }
        outstanding = set(pending.keys())

        while outstanding:
            finished, outstanding = await asyncio.wait(
                outstanding, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                payload = pending[task]
                done += 1
                try:
                    r = task.result()
                except Exception as e:
                    failed += 1
                    r = {
                        "fileName": payload["original_name"],
                        "status": "failed",
                        "message": str(e),
                    }

                s = r.get("status", "failed")
                if s == "success":
                    success += 1
                elif s == "duplicate":
                    skipped += 1
                elif s == "invalid":
                    invalid += 1
                else:
                    failed += 1

                items.append(r)

                progress_event = {
                    "type": "progress",
                    "fileName": r["fileName"],
                    "status": r["status"],
                    "position": r.get("position"),
                    "score": r.get("score"),
                    "message": r.get("message", ""),
                    "existingCandidateName": r.get("existingCandidateName"),
                    "done": done,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "skipped": skipped,
                    "invalid": invalid,
                }
                yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"

        summary_event = {
            "type": "summary",
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "invalid": invalid,
            "items": items,
        }
        yield f"data: {json.dumps(summary_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/resumes/batch-parse")
async def batch_parse(body: dict, db: AsyncSession = Depends(get_db)):
    ids = body.get("ids", [])
    if not ids:
        return fail(400, "请提供候选人 ID 列表")

    for cid in ids:
        result = await db.execute(
            select(Candidate)
            .options(*CANDIDATE_LOAD_OPTIONS)
            .where(Candidate.id == cid)
        )
        candidate = result.scalar_one_or_none()
        if not candidate or not candidate.resume_file:
            continue

        error = await run_resume_parse(
            candidate, candidate.position.name if candidate.position else "", db
        )
        if error:
            logger.warning("Batch parse error for %s: %s", cid, error)

    return ok()


@router.patch("/resumes/{resume_id}")
async def update_resume(
    resume_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("resume:decide")),
):
    # 注意：AI 工具 handler(app/api/ai/tool_handlers/recruitment.py::_update_resume)
    # 是直接 import 本函数并以 fn(resume_id=..., body=..., db=db) 调用的，
    # 不走 FastAPI 依赖注入，因此 current 必须有默认值(Depends(...) 即默认值)，
    # 否则该调用会 TypeError。current 在本函数体内不使用，仅用于 HTTP 层守卫。
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    # 状态迁移放在所有字段编辑之前：迁移失败要整单拒绝，若放在后面，
    # 前面 setattr 的字段已被 autoflush 刷进事务，而 get_db 只在抛异常时回滚，
    # 正常 return 会把这半截改动静默提交（表现为「报错了但人名已经被改掉」）。
    if "status" in body and body["status"] and body["status"] != candidate.status:
        # hired 只能由「录用审批通过」自动生成（见 offer.py），此通用编辑端点
        # 禁止直接把候选人改成 hired，否则会绕开审批必填字段/审批人记录，
        # 与 offer.py 形成第二条并行入职通道。
        if body["status"] == "hired":
            return fail(400, "无法直接将候选人标记为已录用：请通过「录用审批」流程操作。")
        try:
            await transition(db, "candidate", candidate, body["status"], skip_block_check=True)
        except StateError as e:
            await db.rollback()
            return conflict(e.message)

    # Simple fields（status 不在此列——状态变更必须走 transition() 校验合法迁移，
    # 不能像其他字段一样裸 setattr，见 app/core/state_machine.py）
    for field in ["name", "gender", "age", "education", "experience", "phone", "email", "ethnicity"]:
        if field in body:
            setattr(candidate, field, body[field])

    if "ethnicity" in body:
        candidate.ethnicity = normalize_ethnicity(candidate.ethnicity)

    if "nativePlace" in body:
        candidate.native_place = body["nativePlace"]

    if "positionId" in body and body["positionId"]:
        pos_result = await db.execute(select(Position).where(Position.id == body["positionId"]))
        if pos_result.scalar_one_or_none():
            candidate.position_id = body["positionId"]

    # Skills
    if "skills" in body:
        if candidate.skills:
            for s in list(candidate.skills):
                await db.delete(s)
        for skill_name in (body["skills"] or []):
            candidate.skills.append(CandidateSkill(skill=skill_name))

    # Education history
    if "educationHistory" in body:
        if candidate.educations:
            for e in list(candidate.educations):
                await db.delete(e)
        for edu in (body["educationHistory"] or []):
            candidate.educations.append(CandidateEducation(
                school=edu.get("school"), degree=edu.get("degree"),
                major=edu.get("major"), period=edu.get("period"),
            ))

    # Work history
    if "workHistory" in body:
        if candidate.work_experiences:
            for w in list(candidate.work_experiences):
                await db.delete(w)
        for work in (body["workHistory"] or []):
            candidate.work_experiences.append(CandidateWorkExperience(
                company=work.get("company"), role=work.get("role"),
                period=work.get("period"), description=work.get("description"),
            ))

    # Project history
    if "projectHistory" in body:
        if candidate.project_experiences:
            for p in list(candidate.project_experiences):
                await db.delete(p)
        for proj in (body["projectHistory"] or []):
            candidate.project_experiences.append(CandidateProjectExperience(
                name=proj.get("name"), role=proj.get("role"),
                period=proj.get("period"), description=proj.get("description"),
            ))

    await db.flush()
    await db.refresh(candidate)

    # 注：hired 已在上面被禁止直接设置，故此处不再需要 ensure_employee_for_candidate
    # 兜底调用——员工记录只应由 offer.py 的录用审批流程产生。

    return ok(serialize_candidate(candidate))


@router.post("/resumes/{resume_id}/reanalyze")
async def reanalyze_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    """重新解析简历并重新生成 AI 评分——异步任务。

    立即返回任务状态；前端轮询 ``/resumes/{resume_id}/reanalyze-status``
    直到 ``completed`` 后再刷新候选人数据。避免同步串行多次 LLM 调用
    导致请求超时。
    """
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    if not candidate.resume_file:
        return fail(400, "该候选人没有上传简历文件")

    # 已在后台解析/评分中则不重复触发
    state = _reanalyze_task_state(resume_id)
    if state and state["status"] in ("pending", "running"):
        return ok({**state, "resumeId": str(resume_id)})

    position_name = candidate.position.name if candidate.position else ""
    await _set_reanalyze_task(resume_id, status="pending", progress=0, stage="排队中", message="")
    asyncio.create_task(_run_reanalyze_in_background(resume_id, position_name))

    return ok({
        "resumeId": str(resume_id),
        "status": "pending",
        "progress": 0,
        "stage": "排队中",
        "message": "",
    })


@router.get("/resumes/{resume_id}/reanalyze-status")
async def get_reanalyze_status(resume_id: str, db: AsyncSession = Depends(get_db)):
    """轮询重新解析任务的后台进度。"""
    state = _reanalyze_task_state(resume_id)
    if not state:
        # 任务不存在：可能进程重启丢失。此时候选人已完成解析或从未触发，
        # 语义上按「已完成」处理，前端据此直接刷新现有数据即可。
        return ok({
            "resumeId": str(resume_id),
            "status": "completed",
            "progress": 100,
            "stage": "已完成",
            "message": "",
        })
    return ok({**state, "resumeId": str(resume_id)})


@router.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    db: AsyncSession = Depends(get_db),
):
    """删除简历（候选人）。

    删除该候选人意味着此人不存在了，所有相关数据一并清理。
    与「知识库管理 → 删除简历文档」共用同一套级联删除服务
    （app.services.cascade_delete），保证两边行为一致。
    """
    from app.services.system.cascade_delete import cascade_delete_by_candidate
    logger.info("删除候选人开始 candidate_id=%s", resume_id)
    deleted = await cascade_delete_by_candidate(db, resume_id, delete_resume_file=True)
    if not deleted:
        return not_found("候选人不存在")
    return ok()
