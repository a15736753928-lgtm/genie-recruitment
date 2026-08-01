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
    """
    def _ok(data=None, message="ok"):
        return {"status": "success", "statusCode": 200, "message": message, "data": data}

    def _fail(status, code, message, **extra):
        return {"status": status, "statusCode": code, "message": message, "data": None, **extra}

    # 1) 保存文件到 MinIO
    object_key = f"resumes/{uuid.uuid4()}{file_ext}"
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return _fail("failed", 500, f"简历存储失败: {e}")

    resume_text, extract_error = await extract_text_from_file(object_key)
    if extract_error or not resume_text.strip():
        try:
            await asyncio.to_thread(minio_storage.delete_object, object_key)
        except Exception:
            pass
        return _fail("invalid", 422, f"无法读取文档内容：{extract_error or '内容为空'}")

    # 2) 解析应聘岗位
    resolved_position_id = (position_id or "").strip()
    position_source = "user"
    match_reason = ""
    position: Optional[Position] = None

    if resolved_position_id:
        pos_result = await db.execute(select(Position).where(Position.id == resolved_position_id))
        position = pos_result.scalar_one_or_none()
        if not position:
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
        try:
            await asyncio.to_thread(minio_storage.delete_object, object_key)
        except Exception:
            pass
        return _fail("invalid", 422, f"上传的文件不是简历：{reject_reason}")

    # 4) 姓名查重（依赖解析出的姓名，唯一真串行点）
    if ai_enabled and pre_parsed:
        duplicate_person = await find_duplicate_candidate(db, pre_parsed)
        if duplicate_person:
            try:
                await asyncio.to_thread(minio_storage.delete_object, object_key)
            except Exception:
                pass
            return _fail("duplicate", 409,
                         f"该候选人与已有记录为同一人（姓名一致），对应候选人：{duplicate_person.name}",
                         existingCandidateId=str(duplicate_person.id),
                         existingCandidateName=duplicate_person.name)

    # 5) 创建候选人
    candidate = Candidate(
        name=original_name,
        position_id=resolved_position_id,
        status="new",
        resume_file=object_key,
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

    # 5.5) 后台异步评分（4维打分 + 8维 AI 评分并行，不阻塞上传响应）
    try:
        asyncio.create_task(_score_candidate_background(
            candidate_id=str(candidate.id),
            resume_text=resume_text,
            parsed=pre_parsed or {},
            position_name=position.name if position else "",
            object_key=object_key,
        ))
    except Exception as e:
        logger.warning("创建后台评分任务失败（不阻断上传）: %s", e)

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
        return _ok(response_data, f"简历已上传，但自动解析未完成：{parse_message}")
    return _ok(response_data)


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
    basic_t = _safe_parse(_parse_basic(text, position_hint), "基本信息")
    edu_t = _safe_parse(_parse_education(text), "教育经历")
    work_t = _safe_parse(_parse_work(text), "工作经历")
    proj_t = _safe_parse(_parse_project(text), "项目经历")
    skills_t = _safe_parse(_parse_skills(text), "技能")
    extra_t = _safe_parse(_parse_extra(text), "扩展信息")
    score_t = _safe_parse(_parse_score(text, position_hint), "评分")
    insight_t = _safe_parse(_parse_insight(text, position_hint), "洞察")

    basic, edu_r, work_r, proj_r, skills_r, extra_r, score_r, insight_r = await asyncio.gather(
        basic_t, edu_t, work_t, proj_t, skills_t, extra_t, score_t, insight_t
    )
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
) -> Optional[str]:
    """抽取 + 并行解析 + 填充候选人。返回错误消息（None=成功）。

    与上传共用 ``extract_and_parse_parallel`` 并行解析核心。
    评分（4维/8维）不在本函数：调用方负责 ``_score_candidate_background``。
    """
    if not candidate.resume_file:
        return "未找到简历文件"
    if db is None:
        return "内部错误：缺少数据库会话"

    resume_text = ""
    if pre_parsed is not None:
        parsed = pre_parsed
    else:
        parsed, parse_error, resume_text = await extract_and_parse_parallel(
            candidate.resume_file, position_name
        )
        if parse_error and not parsed:
            return parse_error
        if not parsed:
            return "AI 未能解析简历内容"

    await fill_candidate_from_parsed(candidate, parsed, db)

    # 解析阶段粗估分（权威 8 维分优先，不覆盖）
    analysis = parsed.get("analysis") or {}
    overall = analysis.get("overallScore")
    if overall is None:
        overall = candidate.score
    if isinstance(overall, (int, float)) and candidate.screening_ai_score is None:
        candidate.screening_ai_score = int(overall)
        candidate.score = int(overall)

    # 评分异步后台（4维 + 8维 + 性别），不阻塞调用方
    try:
        asyncio.create_task(_score_candidate_background(
            candidate_id=str(candidate.id),
            resume_text=resume_text,
            parsed=parsed,
            position_name=position_name,
            object_key=candidate.resume_file,
        ))
    except Exception as e:
        logger.warning("创建后台评分任务失败（不阻断解析）: %s", e)
    return None


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
    """
    from app.database import async_session_factory

    async def _dimensions():
        async with async_session_factory() as db:
            cand = await _load_candidate(db, candidate_id)
            if not cand or not parsed:
                return
            from app.services.resume_scoring import score_all
            analysis = parsed.setdefault("analysis", {})
            if isinstance(analysis, dict):
                dimensions = await score_all(resume_text, parsed, position_name)
                if dimensions and cand.ai_analysis:
                    cand.ai_analysis.dimensions = dimensions
                await db.commit()

    async def _eight_dims():
        async with async_session_factory() as db:
            cand = await _load_candidate(db, candidate_id)
            if not cand:
                return
            from app.api.recruitment.scoring import auto_score_after_upload
            await auto_score_after_upload(db, cand, position_id=cand.position_id)
            await db.commit()

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
    for skill_name in (parsed.get("skills") or []):
        if skill_name:
            candidate.skills.append(CandidateSkill(skill=str(skill_name)))

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
