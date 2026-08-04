"""Resume upload pipeline — the single shared entry point for resume ingestion.

Called by:
- ``POST /api/resumes/upload`` (single upload)
- ``POST /api/resumes/batch-upload`` (batch SSE streaming)
- ``POST /api/ai-agent/materials`` (Agent chat material upload)

Extracted from ``resumes.py``.  This eliminates the ~100-line duplication
between ``_upload_one_resume`` and ``agent_chat.py:upload_material``.
"""

from __future__ import annotations

import uuid
import hashlib
import logging
import asyncio
from datetime import date
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.infrastructure import minio_storage
from app.models.recruitment import (
    Candidate, Position, CandidateSkill, CandidateEducation,
    CandidateWorkExperience, CandidateProjectExperience, CandidateAIAnalysis,
)
from app.api.recruitment.resume_parser import (
    extract_text_from_file,
    parse_resume_with_llm,
    enrich_parsed_fields,
    augment_gender_from_portrait,
    normalize_text_field,
)
from app.api.recruitment.resume_serializer import (
    UNKNOWN, IN_SCHOOL, DEFAULT_ETHNICITY,
    normalize_ethnicity, normalize_experience, is_candidate_parsed, serialize_candidate,
)
from app.services.recruitment.resume_dedup import find_duplicate_candidate
from app.services.recruitment.resume_validator import validate_is_resume
from app.services.recruitment.education_tier_tag import (
    resolve_school_tier_tag,
    inject_school_tier_into_keywords,
)

settings = get_settings()
logger = logging.getLogger(__name__)


# ── Pipeline ─────────────────────────────────────────────────

async def _upload_one_resume(
    db: AsyncSession,
    *,
    original_name: str,
    content: bytes,
    file_ext: str,
    position_id: Optional[str] = None,
) -> dict:
    """处理单份简历上传的核心逻辑，返回统一的结果 dict。

    供 ``upload_resume``（单份）和 ``batch_upload_resumes``（批量并发）共用。
    MinIO 上传成功后若后续步骤失败，会自动清理已上传的文件（避免孤儿文件）。
    """
    object_key_for_cleanup: Optional[str] = None
    def _ok(data=None, message="ok"):
        return {"status": "success", "statusCode": 200, "message": message, "data": data}

    def _fail(status, code, message, **extra):
        return {"status": status, "statusCode": code, "message": message, "data": None, **extra}

    # 1) 保存文件到 MinIO（先算内容 SHA256，供查重用——只对字节完全相同的
    #    简历判重复，同名不同内容的简历不再误判跳过）
    content_hash = hashlib.sha256(content).hexdigest() if content else ""

    async def _cleanup_minio_file() -> None:
        """上传 MinIO 成功后若后续步骤失败，清理已上传的文件避免孤儿文件。"""
        if object_key_for_cleanup:
            await asyncio.to_thread(minio_storage.delete_object, object_key_for_cleanup)
    object_key = f"resumes/{uuid.uuid4()}{file_ext}"
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return _fail("failed", 500, f"简历存储失败: {e}")

    resume_text, extract_error = await extract_text_from_file(object_key)
    if extract_error or not resume_text.strip():
        await asyncio.to_thread(minio_storage.delete_object, object_key)
        return _fail("invalid", 422, f"无法读取文档内容：{extract_error or '内容为空'}")
    # 从此刻起 MinIO 文件已确认存在：后续失败必须清理
    object_key_for_cleanup = object_key

    # 2) 解析应聘岗位
    resolved_position_id = (position_id or "").strip()
    position_source = "user"
    match_reason = ""
    position: Optional[Position] = None

    if resolved_position_id:
        pos_result = await db.execute(select(Position).where(Position.id == resolved_position_id))
        position = pos_result.scalar_one_or_none()
        if not position:
            await _cleanup_minio_file()
            return _fail("failed", 404, "岗位不存在")
    else:
        from app.services.recruitment.position_matcher import match_position_for_resume
        from app.services.system.system_settings import get_system_setting

        resolved_position_id, position_name, match_reason = await match_position_for_resume(db, resume_text)
        if resolved_position_id:
            position_source = "agent"
            pos_result = await db.execute(select(Position).where(Position.id == resolved_position_id))
            position = pos_result.scalar_one_or_none()
            if not position:
                resolved_position_id = ""
                position = None

        if not position:
            fallback_id = str(await get_system_setting(db, "defaultPositionId", "") or "")
            if fallback_id:
                pos_result = await db.execute(select(Position).where(Position.id == fallback_id))
                position = pos_result.scalar_one_or_none()
                if position:
                    resolved_position_id = fallback_id
                    position_source = "default"
                    match_reason = match_reason or "AI 匹配未命中，已回退到系统默认岗位"

        if not position:
            position_source = "unknown"
            match_reason = match_reason or "AI 未能匹配到在招岗位"

    # 3) 岗位已前置确定（解析依赖岗位 hint）；此后全部只依赖 resume_text / object_key，
    #    一把并行扇出：解析(3) + 性别识别 + 简历校验。
    ai_enabled = await _get_ai_enabled(db)
    auto_parse = await _get_auto_parse(db)
    pre_parsed: Optional[dict] = None
    is_resume_doc = True
    reject_reason = ""
    _document_type = "简历"

    if ai_enabled:
        # 统一走并行解析核心（与 reanalyze/批量/对话材料共用），校验并行
        parsed_task = extract_and_parse_parallel(
            object_key, position.name if position else "", resume_text=resume_text
        )
        valid_task = validate_is_resume(resume_text, parsed=None, use_llm=True)
        (pre_parsed, _parse_err, _text2), validation = await asyncio.gather(
            parsed_task, valid_task
        )
        is_resume_doc, reject_reason, _document_type = validation
    else:
        is_resume_doc, reject_reason, _document_type = await validate_is_resume(
            resume_text, parsed=None, use_llm=False
        )

    if not is_resume_doc:
        await _cleanup_minio_file()
        return _fail("invalid", 422, f"上传的文件不是简历：{reject_reason}")

    # 4) 内容查重（按文件 SHA256，字节完全相同才算重复；不再按姓名）
    if ai_enabled and pre_parsed:
        duplicate_person = await find_duplicate_candidate(db, pre_parsed, content_hash)
        if duplicate_person:
            await _cleanup_minio_file()
            return _fail("duplicate", 409,
                         f"该简历内容与已有记录完全一致（已上传过同一份文件），对应候选人：{duplicate_person.name}",
                         existingCandidateId=str(duplicate_person.id),
                         existingCandidateName=duplicate_person.name)

    # 5) 创建候选人
    candidate = Candidate(
        name=original_name,
        position_id=resolved_position_id,
        status="job_hunting",
        resume_file=object_key,
        resume_file_hash=content_hash,
        upload_time=date.today(),
    )
    db.add(candidate)
    await db.flush()

    # 5.1) 填充候选人（评分走后台，不在此阻塞）
    parse_message = None
    if auto_parse and pre_parsed:
        candidate = await _load_candidate(db, candidate.id)
        if candidate:
            await fill_candidate_from_parsed(candidate, pre_parsed, db)
            # 解析阶段粗估分（保留原 _run_parse 语义：权威 8 维分优先，不覆盖）
            analysis = pre_parsed.get("analysis") or {}
            overall = analysis.get("overallScore")
            if overall is None:
                overall = candidate.score
            if isinstance(overall, (int, float)) and candidate.screening_ai_score is None:
                candidate.screening_ai_score = int(overall)
                candidate.score = int(overall)
            if pre_parsed.get("name") in (None, "", UNKNOWN):
                parse_message = "AI 未能识别姓名，请检查简历格式"
            elif is_candidate_parsed(candidate):
                parse_message = None

    await db.flush()

    # 5.5) 后台评分（4维打分 + 8维 AI 评分 + 性别兜底）。
    #     不在这里 create_task：candidate 此刻尚未 commit，后台任务的独立 DB
    #     session 读不到未提交的候选人，三路会静默提前返回，导致评分全空白
    #     （历史 bug：上传 20 份全部评分空白）。改为把评分参数随返回 dict 带出，
    #     由调用方在 commit 之后调用 launch_resume_scoring() 启动。
    score_task = None
    if ai_enabled and pre_parsed:
        score_task = {
            "candidate_id": str(candidate.id),
            "resume_text": resume_text,
            "parsed": pre_parsed or {},
            "position_name": position.name if position else "",
            "object_key": object_key,
        }

    # 6) RAG 知识库入库已断开（2026-08-01）——知识库/RAG 模块停用，待决定去留。
    #    如需恢复：还原对 ingest_resume_to_kb 的调用即可。

    # 7) 通知 + Webhook
    try:
        from app.services.system.notification import notify_if
        from app.services.system.webhook import dispatch_webhook
        position_label = position.name if position else "未知"
        if position_source == "agent":
            position_label = f"{position.name}（AI 匹配：{match_reason}）"
        elif position_source == "unknown":
            position_label = f"未知（{match_reason}）"
        await notify_if(db, "notifyNewResume", "new_resume",
                        f"新简历入库：{original_name}（岗位：{position_label}）",
                        {"candidateId": str(candidate.id)})
        await dispatch_webhook(db, "candidate.uploaded",
                               {"candidateId": str(candidate.id), "name": original_name,
                                "positionId": resolved_position_id})
    except Exception as e:
        logger.warning("通知/Webhook 钩子失败: %s", e)

    # 8) 返回结果
    candidate = await _load_candidate(db, candidate.id)
    if not candidate:
        return _fail("failed", 500, "候选人加载失败")

    response_data = serialize_candidate(candidate)
    if position_source == "agent":
        response_data["positionMatchReason"] = match_reason
    if parse_message and response_data["parseStatus"] != "parsed":
        result = _ok(response_data, f"简历已上传，但自动解析未完成：{parse_message}")
    else:
        result = _ok(response_data)
    if score_task:
        result["_score_task"] = score_task
    return result


def launch_score_task(score_task: Optional[dict]) -> None:
    """启动单个候选人的后台评分任务（4维打分 + 8维 AI 评分 + 性别兜底）。

    调用方必须在候选人所在事务提交之后调用：后台任务使用独立 DB session，
    读不到未提交的候选人会静默跳过，导致评分空白。score_task 为 None 则跳过。
    """
    if not score_task:
        return
    try:
        asyncio.create_task(_score_candidate_background(**score_task))
    except Exception as e:
        logger.warning("创建后台评分任务失败（不阻断上传）: %s", e)


def launch_resume_scoring(result: dict) -> None:
    """上传结果 dict 若携带评分参数（隐藏键 "_score_task"），在事务提交后启动后台评分。

    必须在 candidate 已 commit 之后调用。result 为 _upload_one_resume 的返回值。
    """
    launch_score_task(result.pop("_score_task", None))


# ── Internal helpers (moved from resumes.py) ──────────────────

async def _get_ai_enabled(db: AsyncSession) -> bool:
    from app.services.system.system_settings import get_system_setting
    return bool(await get_system_setting(db, "aiResumeAnalysis", True))


async def _get_auto_parse(db: AsyncSession) -> bool:
    from app.services.system.system_settings import get_system_setting
    return bool(await get_system_setting(db, "autoParseResume", True))


# 旧的串行预解析 _extract_and_parse 已删除（2026-08-01）——统一走 extract_and_parse_parallel


async def extract_and_parse_parallel(
    resume_file: str,
    position_name: str = "",
    resume_text: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[str], str]:
    """抽取 + 并行解析（profile/extra/analysis 三路）+ enrich。

    上传（_upload_one_resume）与重新解析（resumes.reanalyze）共用的并行解析核心。
    返回 (parsed, error, resume_text)。性别识别不在关键路径（OCR 文本正则已兜底）。
    resume_text 可传入避免重复抽取。
    """
    text = resume_text
    if text is None:
        text, extract_error = await extract_text_from_file(resume_file)
        if extract_error or not text.strip():
            return None, f"无法读取文档内容：{extract_error or '内容为空'}", text or ""

    from app.api.recruitment.resume_parser import (
        _parse_basic, _parse_education, _parse_work, _parse_project,
        _parse_skills, _parse_extra, _parse_score, _parse_insight,
        _safe_parse, enrich_parsed_fields,
    )
    position_hint = f"\n目标应聘岗位：{position_name}" if position_name else ""
    # 8 路聚焦小 JSON 并行：每个输出几百 token（vs 原 profile 4096 / analysis 2048），
    # 总耗时 = 最慢一路，且聚焦任务更稳、不易漏字段。
    basic_t = _safe_parse(lambda: _parse_basic(text, position_hint), "基本信息")
    edu_t = _safe_parse(lambda: _parse_education(text), "教育经历")
    work_t = _safe_parse(lambda: _parse_work(text), "工作经历")
    proj_t = _safe_parse(lambda: _parse_project(text), "项目经历")
    skills_t = _safe_parse(lambda: _parse_skills(text), "技能")
    extra_t = _safe_parse(lambda: _parse_extra(text), "扩展信息")
    score_t = _safe_parse(lambda: _parse_score(text, position_hint), "评分")
    insight_t = _safe_parse(lambda: _parse_insight(text, position_hint), "洞察")

    basic, edu_r, work_r, proj_r, skills_r, extra_r, score_r, insight_r = await asyncio.gather(
        basic_t, edu_t, work_t, proj_t, skills_t, extra_t, score_t, insight_t
    )

    # 空返回检测：哪一路返回空 dict（DeepSeek 空输出/解析失败），在日志里标出来，
    # 便于判断"解析字段缺失"是并发空输出还是提示词问题。
    _empty_parts = [k for k, v in (
        ("基本信息", basic), ("教育经历", edu_r), ("工作经历", work_r),
        ("项目经历", proj_r), ("技能", skills_r), ("扩展信息", extra_r),
        ("评分", score_r), ("洞察", insight_r),
    ) if not v]
    if _empty_parts:
        logger.warning("简历并行解析存在空返回：%s", ",".join(_empty_parts))
    merged = {
        **basic,
        "educationHistory": (edu_r or {}).get("educationHistory", []),
        "workHistory": (work_r or {}).get("workHistory", []),
        "projectHistory": (proj_r or {}).get("projectHistory", []),
        "skills": (skills_r or {}).get("skills", []),
        **(extra_r or {}),
        "analysis": {**(score_r or {}), **(insight_r or {})},
    }
    parsed = enrich_parsed_fields(merged, text)
    return parsed, None, text


async def _load_candidate(db, candidate_id):
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.skills),
            selectinload(Candidate.educations),
            selectinload(Candidate.work_experiences),
            selectinload(Candidate.project_experiences),
            selectinload(Candidate.ai_analysis),
        )
        .where(Candidate.id == candidate_id)
    )
    return result.scalar_one_or_none()


async def _run_parse(
    candidate: Candidate,
    position_name: str = "",
    db: AsyncSession = None,
    pre_parsed: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[dict]]:
    """抽取 + 并行解析 + 填充候选人。返回 (错误消息, 评分参数)。

    评分参数（4维/8维/性别）不在这里启动：后台评分任务使用独立 DB session，
    若在候选人生成事务提交前启动，读不到未提交的候选人会静默跳过（评分空白）。
    调用方须在 commit 之后用 ``launch_score_task(score_task)`` 启动。
    """
    if not candidate.resume_file:
        return "未找到简历文件", None
    if db is None:
        return "内部错误：缺少数据库会话", None

    resume_text = ""
    if pre_parsed is not None:
        parsed = pre_parsed
    else:
        parsed, parse_error, resume_text = await extract_and_parse_parallel(
            candidate.resume_file, position_name
        )
        if parse_error and not parsed:
            return parse_error, None
        if not parsed:
            return "AI 未能解析简历内容", None

    await fill_candidate_from_parsed(candidate, parsed, db)

    # 解析阶段粗估分（权威 8 维分优先，不覆盖）
    analysis = parsed.get("analysis") or {}
    overall = analysis.get("overallScore")
    if overall is None:
        overall = candidate.score
    if isinstance(overall, (int, float)) and candidate.screening_ai_score is None:
        candidate.screening_ai_score = int(overall)
        candidate.score = int(overall)

    score_task = {
        "candidate_id": str(candidate.id),
        "resume_text": resume_text,
        "parsed": parsed,
        "position_name": position_name,
        "object_key": candidate.resume_file,
    }
    return None, score_task


# 后台评分全局并发限制：批量上传会为每份简历创建后台评分任务，若全部并发
# 会再次打爆 DeepSeek。限制同时最多 3 个评分任务。
_SCORE_SEM = asyncio.Semaphore(3)


async def _score_candidate_background(
    candidate_id: str,
    resume_text: str,
    parsed: dict,
    position_name: str = "",
    object_key: str = "",
) -> None:
    """上传后的后台任务：4维打分 + 8维 AI 评分 + 性别兜底，并行执行。

    使用独立 DB 会话（SQLAlchemy async session 不支持并发共享同一 session），
    任一失败只记日志不影响其他。不阻塞上传响应。
    必须在候选人生成的事务提交之后调用，否则独立 session 读不到候选人。
    """
    from app.database import async_session_factory

    async def _dimensions():
        async with async_session_factory() as db:
            cand = await _load_candidate(db, candidate_id)
            if not cand or not parsed:
                return
            from app.services.resume_scoring import score_all
            analysis = parsed.setdefault("analysis", {})
            if not isinstance(analysis, dict):
                return
            for attempt in range(3):
                try:
                    dimensions = await score_all(resume_text, parsed, position_name)
                    if dimensions and cand.ai_analysis:
                        cand.ai_analysis.dimensions = dimensions
                    await db.commit()
                    return
                except Exception as e:
                    if attempt < 2:
                        logger.warning("4维打分第%d次失败（重试）: %s", attempt + 1, e)
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        logger.warning("4维打分重试3次仍失败: %s", e)

    async def _eight_dims():
        async with async_session_factory() as db:
            cand = await _load_candidate(db, candidate_id)
            if not cand:
                return
            from app.api.recruitment.scoring import auto_score_after_upload
            for attempt in range(3):
                try:
                    await auto_score_after_upload(db, cand, position_id=cand.position_id)
                    await db.commit()
                    return
                except Exception as e:
                    if attempt < 2:
                        logger.warning("8维评分第%d次失败（重试）: %s", attempt + 1, e)
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        logger.warning("8维评分重试3次仍失败: %s", e)

    async def _gender_fallback():
        # 文本正则 + 解析都没拿到性别时才走 MIMO 视觉补
        if not object_key:
            return
        async with async_session_factory() as db:
            cand = await _load_candidate(db, candidate_id)
            if not cand or cand.gender in ("男", "女"):
                return
            from app.api.recruitment.resume_parser import augment_gender_from_portrait
            g = await augment_gender_from_portrait({}, object_key)
            if isinstance(g, dict) and g.get("gender") in ("男", "女"):
                cand.gender = g["gender"]
                await db.commit()

    # 全局并发上限：批量上传会为每份简历创建后台评分任务，若全部并发会
    # 打爆 DeepSeek。Semaphore 必须包住 gather 本身（此前只包住了一行 import，
    # 从未生效）。
    async with _SCORE_SEM:
        results = await asyncio.gather(
            _dimensions(), _eight_dims(), _gender_fallback(), return_exceptions=True
        )
    labels = ("4维打分", "8维评分", "性别兜底")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("候选人 %s 后台[%s]失败: %s", candidate_id, labels[i], r)
    logger.info("候选人 %s 后台任务完成", candidate_id)


async def fill_candidate_from_parsed(candidate: Candidate, parsed: dict, db: AsyncSession):
    """Fill candidate fields from LLM parsed result."""
    candidate.name = normalize_text_field(parsed.get("name"))
    candidate.gender = parsed.get("gender") if parsed.get("gender") in ("男", "女", UNKNOWN) else UNKNOWN

    candidate.age = parsed.get("age") if isinstance(parsed.get("age"), int) and parsed.get("age") > 0 else None

    candidate.education = normalize_text_field(parsed.get("education"))
    candidate.phone = normalize_text_field(parsed.get("phone"))
    candidate.email = normalize_text_field(parsed.get("email"))
    candidate.ethnicity = normalize_ethnicity(parsed.get("ethnicity"))
    candidate.native_place = normalize_text_field(parsed.get("nativePlace"))

    for skill in list(candidate.skills):
        await db.delete(skill)
    # 去重后写入：candidate_skills 主键是 (candidate_id, skill)，LLM 解析出的
    # 技能列表偶发含重复项，直接插入会 UniqueViolationError 导致整份上传失败
    # （吴佳熙/闫凯越/张拓 均因此失败）。
    seen_skills = set()
    for skill_name in (parsed.get("skills") or []):
        skill_text = str(skill_name).strip()
        if not skill_text or skill_text in seen_skills:
            continue
        seen_skills.add(skill_text)
        candidate.skills.append(CandidateSkill(skill=skill_text))

    for edu in list(candidate.educations):
        await db.delete(edu)
    for edu in (parsed.get("educationHistory") or []):
        candidate.educations.append(CandidateEducation(
            school=edu.get("school"), degree=edu.get("degree"),
            major=edu.get("major"), period=edu.get("period"),
        ))

    for work in list(candidate.work_experiences):
        await db.delete(work)
    for work in (parsed.get("workHistory") or []):
        candidate.work_experiences.append(CandidateWorkExperience(
            company=work.get("company"), role=work.get("role"),
            period=work.get("period"), description=work.get("description"),
        ))

    candidate.experience = normalize_experience(
        parsed.get("experience"),
        has_work_history=bool(candidate.work_experiences),
    )

    for proj in list(candidate.project_experiences):
        await db.delete(proj)
    for proj in (parsed.get("projectHistory") or []):
        candidate.project_experiences.append(CandidateProjectExperience(
            name=proj.get("name"), role=proj.get("role"),
            period=proj.get("period"), description=proj.get("description"),
        ))

    if candidate.ai_analysis:
        await db.delete(candidate.ai_analysis)

    tier_tag = await resolve_school_tier_tag(parsed)
    if tier_tag:
        inject_school_tier_into_keywords(parsed, tier_tag)

    analysis_data = parsed.get("analysis") or {}
    if analysis_data:
        resume_extra = {}
        for key in ("industryExperience", "managementExperience", "availableDate",
                     "salaryExpectation", "portfolio", "certificates"):
            val = parsed.get(key)
            if val is not None:
                resume_extra[key] = val
        candidate.ai_analysis = CandidateAIAnalysis(
            overall_score=analysis_data.get("overallScore"),
            summary=analysis_data.get("summary"),
            position_match=analysis_data.get("positionMatch"),
            experience_insight=analysis_data.get("experienceInsight"),
            recommendation=analysis_data.get("recommendation"),
            keywords=analysis_data.get("keywords"),
            highlights=analysis_data.get("highlights"),
            risks=analysis_data.get("risks"),
            dimensions=analysis_data.get("dimensions"),
            resume_extra=resume_extra or None,
        )
        candidate.score = analysis_data.get("overallScore") or candidate.score or 0
