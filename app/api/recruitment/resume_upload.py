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

    # 3) 校验是否为个人求职简历
    ai_enabled = await _get_ai_enabled(db)
    pre_parsed: Optional[dict] = None
    if ai_enabled:
        pre_parsed, _ = await _extract_and_parse(
            object_key, position.name if position else "", db=db, resume_text=resume_text
        )

    is_resume_doc, reject_reason, _document_type = await validate_is_resume(
        resume_text, parsed=pre_parsed, use_llm=ai_enabled
    )
    if not is_resume_doc:
        try:
            await asyncio.to_thread(minio_storage.delete_object, object_key)
        except Exception:
            pass
        return _fail("invalid", 422, f"上传的文件不是简历：{reject_reason}")

    # 4) 姓名查重
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

    parse_message = None
    auto_parse = await _get_auto_parse(db)
    if auto_parse:
        candidate = await _load_candidate(db, candidate.id)
        if candidate:
            parse_message = await _run_parse(
                candidate, position.name if position else "", db, pre_parsed=pre_parsed
            )
            if parse_message and is_candidate_parsed(candidate):
                parse_message = None

    # 5.5) 自动 8 维 AI 评分（写入 ResumeScore，供前端 AI 评分标签页展示）
    try:
        from app.api.recruitment.scoring import auto_score_after_upload
        cand_for_scoring = await _load_candidate(db, candidate.id)
        if cand_for_scoring:
            await auto_score_after_upload(
                db, cand_for_scoring,
                position_id=uuid.UUID(resolved_position_id) if resolved_position_id else None,
            )
    except Exception as e:
        logger.warning("自动8维评分失败（不阻断上传）: %s", e)

    await db.flush()

    # 6) RAG 知识库入库
    try:
        from app.services.recruitment.resume_kb import ingest_resume_to_kb
        await ingest_resume_to_kb(
            db, content=content, file_name=original_name,
            source_object_key=object_key, candidate_id=str(candidate.id),
        )
    except Exception as e:
        logger.warning("简历知识库入库钩子失败: %s", e)

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


async def _extract_and_parse(
    object_key: str,
    position_name: str = "",
    *,
    db: AsyncSession,
    resume_text: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """抽取简历文本并用 LLM 解析为结构化数据，供入库前查重或写入候选人。"""
    ai_enabled = await _get_ai_enabled(db)
    if not ai_enabled:
        return None, "AI 简历分析已关闭"

    text = resume_text
    if text is None:
        text, extract_error = await extract_text_from_file(object_key)
        if extract_error:
            return None, extract_error
    if not (text or "").strip():
        return None, "简历文件内容为空"

    parsed, parse_error = await parse_resume_with_llm(text, position_name)
    if parse_error and not parsed:
        return None, parse_error
    if not parsed:
        return None, "AI 未能解析简历内容"

    parsed = enrich_parsed_fields(parsed, text)
    parsed = await augment_gender_from_portrait(parsed, object_key or "")
    return parsed, parse_error


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
    """Extract text, parse with LLM, and fill candidate. Returns error message if any."""
    if not candidate.resume_file:
        return "未找到简历文件"
    if db is None:
        return "内部错误：缺少数据库会话"

    ai_enabled = await _get_ai_enabled(db)
    if not ai_enabled:
        return "AI 简历分析已关闭，仅保存文件"

    text, extract_error = await extract_text_from_file(candidate.resume_file)
    if extract_error:
        return extract_error
    if not text.strip():
        return "简历文件内容为空"

    if pre_parsed is not None:
        parsed = pre_parsed
    else:
        parsed, parse_error = await _extract_and_parse(
            candidate.resume_file, position_name, db=db,
        )
        if parse_error and not parsed:
            return parse_error
        if not parsed:
            return "AI 未能解析简历内容"

    from app.services.resume_scoring import score_all
    analysis = parsed.setdefault("analysis", {})
    if isinstance(analysis, dict):
        analysis["dimensions"] = await score_all(text, parsed, position_name)

    await fill_candidate_from_parsed(candidate, parsed, db)

    from app.services.system.system_settings import get_system_setting
    min_score = int(await get_system_setting(db, "minMatchScore", 70) or 70)
    overall = None
    if isinstance(analysis, dict):
        overall = analysis.get("overallScore")
    if overall is None:
        overall = candidate.score
    # 解析阶段的粗估分。若之后跑过「AI 简历评分」（scoring.py 的 8 维加权），
    # 那边写入的才是权威值，不能被重新解析时的粗估覆盖。
    if isinstance(overall, (int, float)) and candidate.screening_ai_score is None:
        candidate.screening_ai_score = int(overall)
        candidate.score = int(overall)
        _ = min_score

    return parse_error if pre_parsed is None else None


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
