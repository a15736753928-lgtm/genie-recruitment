"""模块三: 面试安排 / 评分 / 媒体 / AI 分析 / 结论

POST /api/interviews               — 安排面试
POST /api/interviews/{id}/scores   — 面试官提交评分
POST /api/interviews/{id}/media    — 上传面试录音/录像
POST /api/interviews/{id}/ai-analysis — AI 分析
POST /api/interviews/{id}/conclusion  — 面试结论
"""
from __future__ import annotations

import json
import os
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Candidate, Position
from app.models.phase1 import Interview, InterviewerScore, InterviewMedia, AIInterviewReport, ResumeScore
from app.models.interview import InterviewTranscript
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.core.exceptions import push_exception, has_blocking_exception
from app.schemas.ai_advice import AIAdvice, parse_ai_advice
from app.utils.responses import ok, fail, not_found
from app.utils.clock import iso_utc
from app.services.ai import llm_chat
from app.services.system.system_settings import get_system_setting
import logging
from app.utils.llm_json import extract_json_object

logger = logging.getLogger("genie.phase1_interview")
router = APIRouter(tags=["面试(Phase1)"])

# ── 维度定义 ─────────────────────────────────────────────────

R1_DIMS = {
    "authenticity": 15,
    "logic_expression": 15,
    "problem_solving": 15,
    "initiative": 10,
    "responsibility": 10,
    "learning": 10,
    "teamwork": 10,
    "position_knowledge": 10,
    "ai_awareness": 5,
}

R2_DIMS = {
    "professional": 20,
    "practical_ops": 25,
    "project_exp": 15,
    "analysis": 15,
    "quality": 10,
    "execution": 5,
    "ai_usage": 5,
    "collaboration": 5,
}

# 维度中文名 —— 前端通过 GET /interviews/dimensions 取，不要再各自硬编码一套 key。
R1_DIM_LABELS = {
    "authenticity": "经历真实性",
    "logic_expression": "逻辑表达",
    "problem_solving": "问题解决",
    "initiative": "主动性",
    "responsibility": "责任心",
    "learning": "学习能力",
    "teamwork": "团队协作",
    "position_knowledge": "岗位认知",
    "ai_awareness": "AI 认知",
}

R2_DIM_LABELS = {
    "professional": "专业能力",
    "practical_ops": "实操能力",
    "project_exp": "项目经验",
    "analysis": "分析能力",
    "quality": "质量意识",
    "execution": "执行力",
    "ai_usage": "AI 应用",
    "collaboration": "协作沟通",
}


def _dims_payload(dims: dict, labels: dict) -> dict:
    return {k: {"key": k, "label": labels.get(k, k), "max": v} for k, v in dims.items()}


# ── Serializers ──────────────────────────────────────────────

def serialize_interview(iv: Interview, viewer=None) -> dict:
    return {
        "id": str(iv.id),
        "candidateId": str(iv.candidate_id),
        "positionId": str(iv.position_id) if iv.position_id else None,
        "round": iv.round,
        "scheduledAt": iso_utc(iv.scheduled_at),
        "interviewerIds": iv.interviewer_ids or [],
        "status": iv.status,
        "practicalScore": iv.practical_score,
        "compositeScore": float(iv.composite_score) if iv.composite_score is not None else None,
        "conclusion": iv.conclusion,
        "createdAt": iso_utc(iv.created_at),
        "updatedAt": iso_utc(iv.updated_at),
    }


def serialize_interviewer_score(s: InterviewerScore) -> dict:
    return {
        "id": str(s.id),
        "interviewId": str(s.interview_id),
        "interviewerId": str(s.interviewer_id),
        "dimensions": s.dimensions,
        "total": s.total,
        "comment": s.comment,
        "isSubmitted": s.is_submitted,
        "createdAt": iso_utc(s.created_at),
    }


# ── Request schemas ───────────────────────────────────────────

class ScheduleInterviewBody(BaseModel):
    model_config = {"populate_by_name": True}
    candidate_id: str = Field(alias="candidateId")
    position_id: str = Field(alias="positionId")
    round: str  # r1 | r2
    scheduled_at: Optional[str] = Field(None, alias="scheduledAt")
    interviewer_ids: List[str] = Field(alias="interviewerIds")


class SubmitScoreBody(BaseModel):
    model_config = {"populate_by_name": True}
    dimensions: dict
    total: int
    comment: Optional[str] = ""


class ConclusionR1Body(BaseModel):
    model_config = {"populate_by_name": True}
    r1_composite: Optional[float] = Field(None, alias="r1Composite")
    advance_to_next_round: bool = Field(alias="advanceToNextRound")
    comment: Optional[str] = ""


class ConclusionR2Body(BaseModel):
    model_config = {"populate_by_name": True}
    r2_composite: Optional[float] = Field(None, alias="r2Composite")
    practical_score: Optional[int] = Field(None, alias="practicalScore")
    decision: str  # recommend | reject
    comment: Optional[str] = ""


# ── Helper ────────────────────────────────────────────────────

async def _load_interview(db: AsyncSession, interview_id: str) -> Optional[Interview]:
    try:
        uid = uuid.UUID(interview_id)
    except (ValueError, TypeError):
        return None
    r = await db.execute(select(Interview).where(Interview.id == uid))
    return r.scalar_one_or_none()


async def _get_threshold(db: AsyncSession, key: str, default: int) -> int:
    val = await get_system_setting(db, key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# 面试的"读"权限：permissions.py 里没有 interview:view 权限点(那是别人的文件，不改)，
# 而 require_permission(*keys) 是"全部满足"语义，单用 interview:manage 会把只有
# interview:score 的 interviewer / manager 挡在列表页外面。故此处做 any-of 校验。
_INTERVIEW_READ_PERMS = ("interview:manage", "interview:score", "resume:view")


async def _require_interview_read(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if any(current.has(p) for p in _INTERVIEW_READ_PERMS):
        return current
    from app.core.security import PermissionError_
    raise PermissionError_(f"无权限: 需要 {' 或 '.join(_INTERVIEW_READ_PERMS)} 之一")


# ── GET /api/interviews/dimensions ────────────────────────────
# 注意：必须声明在任何 /interviews/{...} 动态路径之前，否则会被动态段吞掉。

@router.get("/interviews/dimensions")
async def get_interview_dimensions(current: CurrentUser = Depends(_require_interview_read)):
    """面试评分维度定义(后端真源)。submit_score 严格校验维度键名集合必须与此一致。"""
    return ok({
        "r1": _dims_payload(R1_DIMS, R1_DIM_LABELS),
        "r2": _dims_payload(R2_DIMS, R2_DIM_LABELS),
    })


# ── GET /api/interviews ───────────────────────────────────────

@router.get("/interviews")
async def list_interviews(
    candidate_id: Optional[str] = Query(None, alias="candidateId"),
    position_id: Optional[str] = Query(None, alias="positionId"),
    round: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    """面试列表(分页 + 过滤)，供面试排期/评分页使用。"""
    q = select(Interview)
    if candidate_id:
        try:
            q = q.where(Interview.candidate_id == uuid.UUID(candidate_id))
        except (ValueError, TypeError):
            return ok({"list": [], "total": 0, "page": page, "pageSize": page_size})
    if position_id:
        try:
            q = q.where(Interview.position_id == uuid.UUID(position_id))
        except (ValueError, TypeError):
            return ok({"list": [], "total": 0, "page": page, "pageSize": page_size})
    if round:
        q = q.where(Interview.round == round)
    if status:
        q = q.where(Interview.status == status)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(
        q.order_by(Interview.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()

    # 批量补候选人姓名 / 岗位名(避免 N+1)
    cand_ids = {iv.candidate_id for iv in rows if iv.candidate_id}
    pos_ids = {iv.position_id for iv in rows if iv.position_id}
    cand_map = {}
    if cand_ids:
        cands = (await db.execute(select(Candidate).where(Candidate.id.in_(cand_ids)))).scalars().all()
        cand_map = {c.id: c for c in cands}
    pos_map = {}
    if pos_ids:
        poss = (await db.execute(select(Position).where(Position.id.in_(pos_ids)))).scalars().all()
        pos_map = {p.id: p for p in poss}

    items = []
    for iv in rows:
        item = serialize_interview(iv, current)
        cand = cand_map.get(iv.candidate_id)
        pos = pos_map.get(iv.position_id) if iv.position_id else None
        item["candidateName"] = cand.name if cand else None
        item["candidateStatus"] = cand.status if cand else None
        item["positionName"] = pos.name if pos else None
        items.append(item)

    return ok({"list": items, "total": total, "page": page, "pageSize": page_size})


# ── POST /api/interviews ──────────────────────────────────────

@router.post("/interviews")
async def schedule_interview(
    body: ScheduleInterviewBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("interview:manage")),
):
    # Load candidate
    try:
        cand_uid = uuid.UUID(body.candidate_id)
    except (ValueError, TypeError):
        return not_found("候选人不存在")
    r = await db.execute(select(Candidate).where(Candidate.id == cand_uid))
    candidate = r.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    # Validate candidate status by round
    if body.round == "r1" and candidate.status != "invited":
        return fail(409, "候选人状态不符")
    if body.round == "r2" and candidate.status != "round1":
        return fail(409, "候选人状态不符")

    # Check no existing interview for this candidate+round
    existing_r = await db.execute(
        select(Interview).where(
            and_(Interview.candidate_id == cand_uid, Interview.round == body.round)
        )
    )
    if existing_r.scalar_one_or_none():
        return fail(409, "该候选人本轮面试已安排")

    # Check blocking exception
    block_r = await db.execute(
        select(__import__("app.models.system", fromlist=["ExceptionQueue"]).ExceptionQueue).where(
            __import__("app.models.system", fromlist=["ExceptionQueue"]).ExceptionQueue.entity_type == "candidate",
            __import__("app.models.system", fromlist=["ExceptionQueue"]).ExceptionQueue.entity_id == cand_uid,
            __import__("app.models.system", fromlist=["ExceptionQueue"]).ExceptionQueue.status == "open",
            __import__("app.models.system", fromlist=["ExceptionQueue"]).ExceptionQueue.severity == "block",
        ).limit(1)
    )
    block_exc = block_r.scalar_one_or_none()
    if block_exc:
        return fail(422, f"存在未处理的异常[{block_exc.exception_type}]，请先在异常队列中处理")

    # Parse scheduledAt — strip tzinfo so naive UTC matches the DateTime column
    scheduled_at = None
    if body.scheduled_at:
        try:
            scheduled_at = datetime.fromisoformat(body.scheduled_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            scheduled_at = None

    # Transition candidate status (skip block check since we checked manually)
    try:
        if body.round == "r1":
            await transition(
                db, "candidate", candidate, "round1",
                actor_id=current.id, actor_name=current.username,
                reason="安排一面", skip_block_check=True,
            )
        else:
            await transition(
                db, "candidate", candidate, "round2",
                actor_id=current.id, actor_name=current.username,
                reason="安排二面", skip_block_check=True,
            )
    except StateError as e:
        return fail(409, e.message)

    # Create Interview
    try:
        pos_uid = uuid.UUID(body.position_id)
    except (ValueError, TypeError):
        pos_uid = None

    iv = Interview(
        candidate_id=cand_uid,
        position_id=pos_uid,
        round=body.round,
        scheduled_at=scheduled_at,
        interviewer_ids=body.interviewer_ids,
        status="scheduled",
    )
    db.add(iv)
    await db.flush()
    return ok(serialize_interview(iv))


# ── POST /api/interviews/{id}/scores ─────────────────────────

@router.post("/interviews/{interview_id}/scores")
async def submit_score(
    interview_id: str,
    body: SubmitScoreBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("interview:score")),
):
    iv = await _load_interview(db, interview_id)
    if not iv:
        return not_found("面试记录不存在")

    # Validate dimension keys
    expected_dims = R1_DIMS if iv.round == "r1" else R2_DIMS
    if set(body.dimensions.keys()) != set(expected_dims.keys()):
        return fail(400, "维度键名不匹配")

    # Validate total 0-100
    if not (0 <= body.total <= 100):
        return fail(400, "总分须在0-100之间")

    # Validate sum of dimension values equals total
    dim_sum = sum(int(v) for v in body.dimensions.values())
    if dim_sum != body.total:
        return fail(400, "各维度分之和与总分不符")

    # Upsert InterviewerScore
    existing_r = await db.execute(
        select(InterviewerScore).where(
            and_(
                InterviewerScore.interview_id == iv.id,
                InterviewerScore.interviewer_id == current.id,
            )
        )
    )
    score = existing_r.scalar_one_or_none()
    if score:
        score.dimensions = body.dimensions
        score.total = body.total
        score.comment = body.comment
        score.is_submitted = True
    else:
        score = InterviewerScore(
            interview_id=iv.id,
            interviewer_id=current.id,
            dimensions=body.dimensions,
            total=body.total,
            comment=body.comment,
            is_submitted=True,
        )
        db.add(score)
    await db.flush()

    # Check if ALL interviewers have submitted
    interviewer_ids = iv.interviewer_ids or []
    all_submitted = False
    gap = 0

    if interviewer_ids:
        submitted_r = await db.execute(
            select(InterviewerScore).where(
                and_(
                    InterviewerScore.interview_id == iv.id,
                    InterviewerScore.is_submitted == True,  # noqa: E712
                )
            )
        )
        submitted_scores = submitted_r.scalars().all()
        submitted_ids = {str(s.interviewer_id) for s in submitted_scores}
        all_submitted = all(str(iid) in submitted_ids for iid in interviewer_ids)

        if all_submitted and len(submitted_scores) >= 2:
            totals = [s.total for s in submitted_scores]
            gap = max(totals) - min(totals)
            # interviewer_score_gap 目前不在 system_settings 的 DEFAULT_SETTINGS 里，
            # 除非管理员在系统设置中手动新增该 key，否则这里恒取兜底值 20（分差≥20 视为异常）。
            threshold = await _get_threshold(db, "interviewer_score_gap", 20)
            if gap >= threshold:
                await push_exception(
                    db,
                    entity_type="candidate",
                    entity_id=iv.candidate_id,
                    exception_type="score_gap",
                    detail=f"分差{gap}分≥{threshold}",
                    severity="block",
                )

    return ok({
        "interviewerScore": serialize_interviewer_score(score),
        "allSubmitted": all_submitted,
        "maxGap": gap,
    })


# ── POST /api/interviews/{id}/media ──────────────────────────

@router.post("/interviews/{interview_id}/media")
async def upload_media(
    interview_id: str,
    file: UploadFile = File(...),
    consentObtained: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("media:upload")),
):
    consent = consentObtained.lower() in ("true", "1", "yes")
    if not consent:
        return fail(422, "需获得候选人知情同意方可上传录音")

    iv = await _load_interview(db, interview_id)
    if not iv:
        return not_found("面试记录不存在")

    file_bytes = await file.read()
    filename = file.filename or "media"
    ext = os.path.splitext(filename)[1].lower()
    object_key = f"interviews/{interview_id}/{uuid.uuid4()}{ext}"

    # Detect media type
    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
    media_type = "audio" if ext in audio_exts else "video"

    # Upload to MinIO
    try:
        from app.infrastructure import minio_storage
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, file_bytes, file.content_type or "application/octet-stream"
        )
    except Exception as e:
        logger.error("media upload failed: %s", e)
        return fail(500, f"文件存储失败: {e}")

    m = InterviewMedia(
        interview_id=iv.id,
        file_path=object_key,
        media_type=media_type,
        consent_obtained=True,
    )
    db.add(m)
    await db.flush()

    return ok({
        "mediaId": str(m.id),
        "filePath": m.file_path,
        "mediaType": m.media_type,
    })


# ── POST /api/interviews/{id}/ai-analysis ────────────────────

@router.post("/interviews/{interview_id}/ai-analysis")
async def run_ai_analysis(
    interview_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("interview:manage")),
):
    iv = await _load_interview(db, interview_id)
    if not iv:
        return not_found("面试记录不存在")

    # Load candidate
    cand_r = await db.execute(select(Candidate).where(Candidate.id == iv.candidate_id))
    candidate = cand_r.scalar_one_or_none()

    # Load latest transcript
    transcript_r = await db.execute(
        select(InterviewTranscript).where(
            and_(
                InterviewTranscript.candidate_id == iv.candidate_id,
                InterviewTranscript.round == ("first" if iv.round == "r1" else "second"),
            )
        ).order_by(InterviewTranscript.created_at.desc()).limit(1)
    )
    transcript = transcript_r.scalar_one_or_none()
    transcript_content = transcript.content if transcript else ""

    candidate_info = f"姓名: {candidate.name if candidate else '未知'}"
    if candidate and candidate.education:
        candidate_info += f"\n学历: {candidate.education}"
    if candidate and candidate.experience:
        candidate_info += f"\n工作经验: {candidate.experience}"

    prompt = f"""你是一位资深的面试评估专家。请基于以下面试转写内容对候选人进行综合分析。

候选人信息:
{candidate_info}

面试转写内容:
{transcript_content[:8000] if transcript_content else "（暂无转写内容）"}

请输出以下JSON格式的分析结果（所有字段均为snake_case）:
{{
  "result": "综合评估结论（一句话）",
  "score": 候选人内容质量评分(0-100的整数),
  "authenticity_score": 经历真实性评分(0-100的整数),
  "confidence": 置信度(0.0-1.0的小数),
  "evidence": ["证据1", "证据2", ...],
  "strengths": ["优势1", "优势2"],
  "risks": ["风险1", "风险2"],
  "missing_information": ["缺失信息1"],
  "recommended_action": "建议行动",
  "requires_human_confirmation": true,
  "answered_directly": true或false,
  "role_clear": true或false,
  "concrete_result": true或false,
  "process_described": true或false,
  "contradiction_found": true或false,
  "avoided_key": true或false,
  "logical": true或false
}}

严格返回纯JSON，不要包含markdown代码围栏或解释。"""

    try:
        raw_text = await llm_chat([{"role": "user", "content": prompt}])
        # Strip code fences
        raw_text = raw_text.strip()
        raw_dict = extract_json_object(raw_text)
        if raw_dict is None:
            raise ValueError("模型未返回可解析的 JSON")
    except Exception as e:
        logger.warning("AI analysis LLM call failed: %s", e)
        raw_dict = {
            "result": "AI分析失败，请人工评估",
            "score": 60,
            "authenticity_score": 60,
            "confidence": 0.4,
            "evidence": ["系统异常，无法获取LLM分析"],
            "strengths": [],
            "risks": ["AI分析异常"],
            "missing_information": [],
            "recommended_action": "请HR人工评估",
            "requires_human_confirmation": True,
            "answered_directly": False,
            "role_clear": False,
            "concrete_result": False,
            "process_described": False,
            "contradiction_found": False,
            "avoided_key": False,
            "logical": False,
        }

    # Parse advice
    advice_dict = {
        "result": raw_dict.get("result", ""),
        "score": raw_dict.get("score"),
        "confidence": float(raw_dict.get("confidence", 0.5)),
        "evidence": raw_dict.get("evidence", []),
        "strengths": raw_dict.get("strengths", []),
        "risks": raw_dict.get("risks", []),
        "missing_information": raw_dict.get("missing_information", []),
        "recommended_action": raw_dict.get("recommended_action", ""),
        "requires_human_confirmation": raw_dict.get("requires_human_confirmation", True),
    }

    # Check evidence
    if not advice_dict.get("evidence"):
        await push_exception(
            db,
            entity_type="candidate",
            entity_id=iv.candidate_id,
            exception_type="low_confidence",
            detail="AI 建议缺少证据",
            severity="block",
        )
        return fail(422, "AI 建议缺少证据")

    try:
        advice = parse_ai_advice(advice_dict)
    except Exception as e:
        await push_exception(
            db,
            entity_type="candidate",
            entity_id=iv.candidate_id,
            exception_type="low_confidence",
            detail=f"AI advice 解析失败: {e}",
            severity="block",
        )
        return fail(422, f"AI advice 解析失败: {e}")

    # Check confidence threshold
    conf_threshold = float(await get_system_setting(db, "aiConfidenceThreshold", 0.6))
    if advice.confidence < conf_threshold:
        await push_exception(
            db,
            entity_type="candidate",
            entity_id=iv.candidate_id,
            exception_type="low_confidence",
            detail=f"AI 置信度 {advice.confidence:.2f} < {conf_threshold}",
            severity="block",
        )

    score = int(raw_dict.get("score", 60) or 60)
    authenticity_score = int(raw_dict.get("authenticity_score", 60) or 60)

    # Upsert AIInterviewReport
    rep_r = await db.execute(
        select(AIInterviewReport).where(AIInterviewReport.interview_id == iv.id)
    )
    report = rep_r.scalar_one_or_none()
    if report:
        report.score = score
        report.authenticity_score = authenticity_score
        report.advice = advice.model_dump()
        report.answered_directly = raw_dict.get("answered_directly")
        report.role_clear = raw_dict.get("role_clear")
        report.concrete_result = raw_dict.get("concrete_result")
        report.process_described = raw_dict.get("process_described")
        report.contradiction_found = raw_dict.get("contradiction_found")
        report.avoided_key = raw_dict.get("avoided_key")
        report.logical = raw_dict.get("logical")
    else:
        report = AIInterviewReport(
            candidate_id=iv.candidate_id,
            interview_id=iv.id,
            round=iv.round,
            score=score,
            authenticity_score=authenticity_score,
            advice=advice.model_dump(),
            answered_directly=raw_dict.get("answered_directly"),
            role_clear=raw_dict.get("role_clear"),
            concrete_result=raw_dict.get("concrete_result"),
            process_described=raw_dict.get("process_described"),
            contradiction_found=raw_dict.get("contradiction_found"),
            avoided_key=raw_dict.get("avoided_key"),
            logical=raw_dict.get("logical"),
        )
        db.add(report)
    await db.flush()

    return ok(_serialize_ai_report(report))


def _serialize_ai_report(report: AIInterviewReport) -> dict:
    return {
        "reportId": str(report.id),
        "interviewId": str(report.interview_id),
        "candidateId": str(report.candidate_id),
        "round": report.round,
        "score": report.score,
        "authenticityScore": report.authenticity_score,
        "advice": report.advice,
        "answeredDirectly": report.answered_directly,
        "roleClear": report.role_clear,
        "concreteResult": report.concrete_result,
        "processDescribed": report.process_described,
        "contradictionFound": report.contradiction_found,
        "avoidedKey": report.avoided_key,
        "logical": report.logical,
        "createdAt": iso_utc(report.created_at),
    }


@router.get("/interviews/{interview_id}/ai-analysis")
async def get_ai_analysis(
    interview_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    """读取已生成的 AI 面试分析。

    此前只有 POST（现跑现取），报告虽然落了库却没有读取入口，
    页面刷新后 AI 分数就没了，一面综合分预览也因此算不出来。
    """
    iv = await _load_interview(db, interview_id)
    if not iv:
        return not_found("面试记录不存在")
    report = (await db.execute(
        select(AIInterviewReport).where(AIInterviewReport.interview_id == iv.id)
    )).scalar_one_or_none()
    if not report:
        return not_found("尚未生成 AI 分析")
    return ok(_serialize_ai_report(report))


# ── POST /api/interviews/{id}/conclusion ─────────────────────

@router.post("/interviews/{interview_id}/conclusion")
async def submit_conclusion(
    interview_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("interview:manage")),
):
    iv = await _load_interview(db, interview_id)
    if not iv:
        return not_found("面试记录不存在")

    candidate_id = iv.candidate_id

    # Check blocking exceptions
    if await has_blocking_exception(db, "candidate", candidate_id):
        return fail(422, "存在未处理的 block 级异常，请先处理")

    cand_r = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = cand_r.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    # Load interviewer scores for composite calculation
    scores_r = await db.execute(
        select(InterviewerScore).where(
            and_(InterviewerScore.interview_id == iv.id, InterviewerScore.is_submitted == True)  # noqa: E712
        )
    )
    interviewer_scores = scores_r.scalars().all()
    avg_interviewer = (
        sum(s.total for s in interviewer_scores) / len(interviewer_scores)
        if interviewer_scores else 0.0
    )

    # Load AI report
    rep_r = await db.execute(
        select(AIInterviewReport).where(AIInterviewReport.interview_id == iv.id)
    )
    ai_report = rep_r.scalar_one_or_none()
    ai_content_score = float(ai_report.score or 0) if ai_report else 0.0
    ai_authenticity_score = float(ai_report.authenticity_score or 0) if ai_report else 0.0

    if iv.round == "r1":
        # R1 composite: round(avg_interviewer * 0.60 + ai_content_score * 0.25 + authenticity_score * 0.15)
        composite = round(avg_interviewer * 0.60 + ai_content_score * 0.25 + ai_authenticity_score * 0.15)
        iv.composite_score = composite

        advance = body.get("advanceToNextRound", False)
        comment = body.get("comment", "")
        iv.conclusion = "advance" if advance else "review"

        if advance:
            try:
                await transition(
                    db, "candidate", candidate, "round2",
                    actor_id=current.id, actor_name=current.username,
                    reason=f"一面结论: {comment}", skip_block_check=True,
                )
            except StateError as e:
                # 上面已写入 composite_score/conclusion（autoflush 已刷进事务），
                # get_db 只在抛异常时回滚，正常 return 会把这半截结论静默提交。
                await db.rollback()
                return fail(409, e.message)

    else:  # r2
        practical_score = body.get("practicalScore") or (iv.practical_score or 0)
        if practical_score is None:
            practical_score = 0
        practical_score = float(practical_score)

        # R2 composite: round(practical_score * 0.40 + avg_interviewer * 0.40 + ai_score * 0.20)
        ai_score_r2 = ai_content_score
        composite = round(practical_score * 0.40 + avg_interviewer * 0.40 + ai_score_r2 * 0.20)
        iv.composite_score = composite
        if practical_score:
            iv.practical_score = int(practical_score)

        decision = body.get("decision", "recommend")
        comment = body.get("comment", "")

        if decision == "reject":
            iv.conclusion = "reject"
            try:
                await transition(
                    db, "candidate", candidate, "rejected",
                    actor_id=current.id, actor_name=current.username,
                    reason=f"二面结论-淘汰: {comment}", skip_block_check=True,
                )
            except StateError as e:
                # 上面已写入 composite_score/conclusion（autoflush 已刷进事务），
                # get_db 只在抛异常时回滚，正常 return 会把这半截结论静默提交。
                await db.rollback()
                return fail(409, e.message)
        else:
            iv.conclusion = "recommend"
            try:
                await transition(
                    db, "candidate", candidate, "pending_offer",
                    actor_id=current.id, actor_name=current.username,
                    reason=f"二面结论-推荐录用: {comment}", skip_block_check=True,
                )
            except StateError as e:
                # 上面已写入 composite_score/conclusion（autoflush 已刷进事务），
                # get_db 只在抛异常时回滚，正常 return 会把这半截结论静默提交。
                await db.rollback()
                return fail(409, e.message)

        # AI vs interviewer conflict check
        if ai_report and interviewer_scores:
            ai_s = float(ai_report.score or 0)
            interviewer_avg = avg_interviewer
            diff = abs(ai_s - interviewer_avg)
            if diff >= 20:
                # Opposite direction: AI high + interviewers low, or vice versa
                ai_positive = ai_s >= 60
                human_positive = interviewer_avg >= 60
                if ai_positive != human_positive:
                    await push_exception(
                        db,
                        entity_type="candidate",
                        entity_id=candidate_id,
                        exception_type="ai_vs_human_conflict",
                        detail=f"AI评分{ai_s:.0f} vs 面试官均分{interviewer_avg:.0f}，分差{diff:.0f}且方向相反",
                        severity="block",
                    )

    iv.status = "completed"
    await db.flush()

    return ok({
        "compositeScore": float(iv.composite_score),
        "conclusion": iv.conclusion,
        "candidateStatus": candidate.status,
    })
