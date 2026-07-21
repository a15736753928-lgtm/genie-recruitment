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

router = APIRouter(tags=["简历"])
settings = get_settings()
logger = logging.getLogger(__name__)

os.makedirs(settings.upload_dir, exist_ok=True)


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


# ── Endpoints ───────────────────────────────────────────

@router.get("/resumes")
async def list_resumes(
    positionId: str = Query("all"),
    statuses: str = Query(""),
    keyword: str = Query(""),
    sortBy: str = Query("uploadTime"),
    sortOrder: str = Query("desc"),
    page: int = Query(1),
    pageSize: int = Query(10),
    minScore: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    query = select(Candidate).options(*CANDIDATE_LOAD_OPTIONS)

    # Filter by position
    if positionId and positionId != "all":
        query = query.where(Candidate.position_id == positionId)

    # 最低匹配分过滤：未传 minScore 时读系统设置
    score_threshold = minScore
    if score_threshold is None:
        score_threshold = int(await get_system_setting(db, "minMatchScore", 70) or 70)
    # 仅当显式传入 minScore 时强制过滤；默认阈值用于「低匹配」标记，列表仍展示全部
    if minScore is not None:
        query = query.where(Candidate.score >= score_threshold)

    # Filter by statuses
    if statuses:
        status_list = [s.strip() for s in statuses.split(",") if s.strip()]
        if status_list:
            expanded: list[str] = []
            for status in status_list:
                expanded.append(status)
                if status == "passed":
                    expanded.extend(["pending_interview"])
            query = query.where(Candidate.status.in_(expanded))

    # Keyword search
    if keyword:
        kw = f"%{keyword}%"
        query = query.outerjoin(Position, Candidate.position_id == Position.id).where(
            or_(
                Candidate.name.ilike(kw),
                Candidate.skills.any(CandidateSkill.skill.ilike(kw)),
                Position.name.ilike(kw),
            )
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Sort
    sort_col = Candidate.upload_time if sortBy == "uploadTime" else Candidate.score
    if sortOrder == "asc":
        query = query.order_by(asc(sort_col))
    else:
        query = query.order_by(desc(sort_col))

    # Paginate
    offset = (page - 1) * pageSize
    query = query.offset(offset).limit(pageSize)
    result = await db.execute(query)
    candidates = result.unique().scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "list": [serialize_candidate(c) for c in candidates],
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.get("/resumes/{resume_id}/file")
async def get_resume_file(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == resume_id))
    candidate = result.scalar_one_or_none()
    if not candidate or not candidate.resume_file:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "简历文件不存在", "data": None},
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
            filename=filename,
            headers={"Content-Disposition": cd_header},
        )

    # MinIO object key — stream from object storage
    try:
        response = await asyncio.to_thread(minio_storage.get_object_stream, stored)
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": f"简历文件不存在: {e}", "data": None},
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
        return {"code": 404, "message": "候选人不存在", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


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
        return {"code": 0, "message": result["message"], "data": response_data}
    else:
        return {
            "code": result["statusCode"],
            "message": result["message"],
            "data": result.get("data"),
        }


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
        return {"code": 400, "message": "请提供候选人 ID 列表", "data": None}

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

    return {"code": 0, "message": "ok", "data": None}


@router.patch("/resumes/{resume_id}")
async def update_resume(
    resume_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    # Simple fields
    for field in ["name", "gender", "age", "education", "experience", "status", "phone", "email", "ethnicity"]:
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
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


@router.post("/resumes/{resume_id}/reanalyze")
async def reanalyze_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    if not candidate.resume_file:
        return {"code": 400, "message": "该候选人没有上传简历文件", "data": None}

    error = await run_resume_parse(
        candidate, candidate.position.name if candidate.position else "", db
    )
    if error and not is_candidate_parsed(candidate):
        return {"code": 500, "message": f"解析失败: {error}", "data": None}

    await db.flush()
    candidate = await load_candidate(db, resume_id)
    if not candidate:
        return {"code": 500, "message": "候选人加载失败", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


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
        return {"code": 404, "message": "候选人不存在", "data": None}
    return {"code": 0, "message": "ok", "data": None}
