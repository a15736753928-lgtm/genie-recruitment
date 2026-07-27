"""模块四: 录用审批

POST /api/offers          — 创建录用审批单(幂等)
PUT  /api/offers/{id}     — 更新审批填写内容
POST /api/offers/{id}/approve — 提交审批决定
GET  /api/offers          — 分页列表
GET  /api/offers/{id}     — 详情
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Candidate, Position
from app.models.phase1 import Interview, OfferApproval, ResumeScore
from app.models.probation import Employee
from app.core.security import get_current_user, require_permission, CurrentUser, PermissionError_
from app.core.state_machine import transition, StateError
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
from app.utils.clock import iso_utc
from app.services.ai import llm_chat
import logging
from app.utils.llm_json import extract_json_object

logger = logging.getLogger("genie.offer")
router = APIRouter(tags=["录用审批"])


# ── Serializer ───────────────────────────────────────────────

_LIST_SEPARATORS = "、\n\r;；,，"


def _as_list(value) -> list[str]:
    """把 Text 字段(顿号/换行/分号/逗号分隔的自由文本)统一成字符串数组。

    strengths / capability_gaps / risks_note 在模型里都是 Column(Text)，
    前端却按数组用(.join()/.map())，直接给字符串会白屏。空值返回 []。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    for sep in _LIST_SEPARATORS[1:]:
        text = text.replace(sep, _LIST_SEPARATORS[0])
    return [seg.strip() for seg in text.split(_LIST_SEPARATORS[0]) if seg.strip()]


def serialize_offer(offer: OfferApproval, viewer=None) -> dict:
    show_salary = viewer is not None and (
        hasattr(viewer, "has") and viewer.has("salary:view")
    )
    strengths = _as_list(offer.strengths)
    gaps = _as_list(offer.capability_gaps)
    risks = _as_list(offer.risks_note)
    return {
        "id": str(offer.id),
        "candidateId": str(offer.candidate_id),
        "positionId": str(offer.position_id) if offer.position_id else None,
        "resumeScore": float(offer.resume_score) if offer.resume_score is not None else None,
        "r1Score": float(offer.r1_score) if offer.r1_score is not None else None,
        "r2Score": float(offer.r2_score) if offer.r2_score is not None else None,
        "practicalScore": float(offer.practical_score) if offer.practical_score is not None else None,
        "teamScore": float(offer.team_score) if offer.team_score is not None else None,
        "finalScore": float(offer.final_score) if offer.final_score is not None else None,
        "aiAdvice": offer.ai_advice,
        # 三个自由文本字段统一以数组输出；同时给出前端在用的别名 gaps / risks。
        "strengths": strengths,
        "capabilityGaps": gaps,
        "gaps": gaps,
        "risksNote": risks,
        "risks": risks,
        # 原始文本(编辑表单回填 textarea 用，PUT /offers/{id} 收的仍是字符串)
        "strengthsText": offer.strengths,
        "capabilityGapsText": offer.capability_gaps,
        "risksNoteText": offer.risks_note,
        "suggestedSalary": offer.suggested_salary if show_salary else None,
        "salaryMasked": not show_salary,
        "probationGoal": offer.probation_goal,
        "trainingPlan": offer.training_plan,
        "mentorId": str(offer.mentor_id) if offer.mentor_id else None,
        "conversionCriteria": offer.conversion_criteria,
        "eliminationCriteria": offer.elimination_criteria,
        "approverId": str(offer.approver_id) if offer.approver_id else None,
        "approvedAt": iso_utc(offer.approved_at),
        "rejectReason": offer.reject_reason,
        "aiResult": offer.ai_result,
        "result": offer.result,
        "status": offer.status,
        "createdAt": iso_utc(offer.created_at),
        "updatedAt": iso_utc(offer.updated_at),
    }


# ── Request schemas ───────────────────────────────────────────

class CreateOfferBody(BaseModel):
    model_config = {"populate_by_name": True}
    candidate_id: str = Field(alias="candidateId")
    position_id: str = Field(alias="positionId")
    team_score: Optional[float] = Field(None, alias="teamScore")


class UpdateOfferBody(BaseModel):
    model_config = {"populate_by_name": True}
    team_score: Optional[float] = Field(None, alias="teamScore")
    strengths: Optional[str] = None
    capability_gaps: Optional[str] = Field(None, alias="capabilityGaps")
    risks_note: Optional[str] = Field(None, alias="risksNote")
    suggested_salary: Optional[str] = Field(None, alias="suggestedSalary")
    probation_goal: Optional[str] = Field(None, alias="probationGoal")
    training_plan: Optional[str] = Field(None, alias="trainingPlan")
    mentor_id: Optional[str] = Field(None, alias="mentorId")
    conversion_criteria: Optional[str] = Field(None, alias="conversionCriteria")
    elimination_criteria: Optional[str] = Field(None, alias="eliminationCriteria")


class ApproveOfferBody(BaseModel):
    model_config = {"populate_by_name": True}
    result: str  # priority | recommend | conditional | reserve | reject
    reason: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────

def _compute_final_score(
    resume: Optional[float],
    r1: Optional[float],
    r2: Optional[float],
    practical: Optional[float],
    team: Optional[float],
) -> float:
    """final = resume*0.15 + r1*0.25 + r2*0.40 + practical*0.10 + team*0.10"""
    return round(
        (resume or 0) * 0.15
        + (r1 or 0) * 0.25
        + (r2 or 0) * 0.40
        + (practical or 0) * 0.10
        + (team or 0) * 0.10,
        2,
    )


async def _load_offer(db: AsyncSession, offer_id: str) -> Optional[OfferApproval]:
    try:
        uid = uuid.UUID(offer_id)
    except (ValueError, TypeError):
        return None
    r = await db.execute(select(OfferApproval).where(OfferApproval.id == uid))
    return r.scalar_one_or_none()


async def _build_ai_advice(candidate: Candidate, offer: OfferApproval) -> Optional[dict]:
    """Call LLM for hire/no-hire recommendation; return None on failure."""
    prompt = f"""你是一位资深HR总监，请基于以下候选人综合评估信息给出录用建议。

候选人: {candidate.name if candidate else '未知'}
简历评分: {float(offer.resume_score) if offer.resume_score else 'N/A'}
一面综合分: {float(offer.r1_score) if offer.r1_score else 'N/A'}
二面综合分: {float(offer.r2_score) if offer.r2_score else 'N/A'}
实操评分: {float(offer.practical_score) if offer.practical_score else 'N/A'}
团队评分: {float(offer.team_score) if offer.team_score else 'N/A'}
综合得分: {float(offer.final_score) if offer.final_score else 'N/A'}

请给出JSON格式的录用建议（snake_case字段）:
{{
  "result": "recommend（建议录用）或 reject（不建议录用）",
  "score": 综合评估分(0-100整数),
  "confidence": 置信度(0.0-1.0),
  "evidence": ["关键依据1", "关键依据2"],
  "strengths": ["优势1"],
  "risks": ["风险1"],
  "missing_information": [],
  "recommended_action": "具体建议",
  "requires_human_confirmation": true
}}

严格返回纯JSON，不含markdown围栏。"""

    try:
        raw_text = await llm_chat([{"role": "user", "content": prompt}])
        raw_text = raw_text.strip()
        raw_dict = extract_json_object(raw_text)
        if raw_dict is None:
            raise ValueError("模型未返回可解析的 JSON")

        evidence = raw_dict.get("evidence", [])
        if not evidence:
            return None  # No evidence — skip storing advice

        from app.schemas.ai_advice import parse_ai_advice
        advice = parse_ai_advice({
            "result": raw_dict.get("result", "recommend"),
            "score": raw_dict.get("score"),
            "confidence": float(raw_dict.get("confidence", 0.7)),
            "evidence": evidence,
            "strengths": raw_dict.get("strengths", []),
            "risks": raw_dict.get("risks", []),
            "missing_information": raw_dict.get("missing_information", []),
            "recommended_action": raw_dict.get("recommended_action", ""),
            "requires_human_confirmation": True,
        })
        return advice.model_dump()
    except Exception as e:
        logger.warning("Offer AI advice generation failed: %s", e)
        return None


# ── POST /api/offers ──────────────────────────────────────────

@router.post("/offers")
async def create_offer(
    body: CreateOfferBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    try:
        cand_uid = uuid.UUID(body.candidate_id)
    except (ValueError, TypeError):
        return not_found("候选人不存在")

    cand_r = await db.execute(select(Candidate).where(Candidate.id == cand_uid))
    candidate = cand_r.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    if candidate.status != "pending_offer":
        return fail(409, "候选人须处于待录用状态")

    # Idempotent: return existing if already created
    existing_r = await db.execute(
        select(OfferApproval).where(OfferApproval.candidate_id == cand_uid)
    )
    existing = existing_r.scalar_one_or_none()
    if existing:
        return ok(serialize_offer(existing, current))

    try:
        pos_uid = uuid.UUID(body.position_id)
    except (ValueError, TypeError):
        pos_uid = None

    # Auto-pull scores
    resume_score_val: Optional[float] = None
    if pos_uid:
        rs_r = await db.execute(
            select(ResumeScore).where(
                and_(
                    ResumeScore.candidate_id == cand_uid,
                    ResumeScore.position_id == pos_uid,
                )
            )
        )
        rs = rs_r.scalar_one_or_none()
        if rs:
            resume_score_val = float(rs.total) if rs.total is not None else None

    r1_score_val: Optional[float] = None
    r2_score_val: Optional[float] = None
    practical_score_val: Optional[float] = None

    ivs_r = await db.execute(
        select(Interview).where(Interview.candidate_id == cand_uid)
    )
    interviews = ivs_r.scalars().all()
    for iv in interviews:
        if iv.round == "r1" and iv.composite_score is not None:
            r1_score_val = float(iv.composite_score)
        if iv.round == "r2":
            if iv.composite_score is not None:
                r2_score_val = float(iv.composite_score)
            if iv.practical_score is not None:
                practical_score_val = float(iv.practical_score)

    team_score_val = body.team_score

    final = _compute_final_score(
        resume_score_val, r1_score_val, r2_score_val, practical_score_val, team_score_val
    )

    offer = OfferApproval(
        candidate_id=cand_uid,
        position_id=pos_uid,
        resume_score=resume_score_val,
        r1_score=r1_score_val,
        r2_score=r2_score_val,
        practical_score=practical_score_val,
        team_score=team_score_val,
        final_score=final,
        status="pending",
    )
    db.add(offer)
    await db.flush()

    # Generate AI advice (non-blocking — skip if evidence empty)
    ai_advice = await _build_ai_advice(candidate, offer)
    if ai_advice:
        offer.ai_advice = ai_advice
        offer.ai_result = ai_advice.get("result")
        await db.flush()

    return ok(serialize_offer(offer, current))


# ── PUT /api/offers/{id} ──────────────────────────────────────

@router.put("/offers/{offer_id}")
async def update_offer(
    offer_id: str,
    body: UpdateOfferBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("录用审批单不存在")

    if offer.status in ("approved", "rejected"):
        return fail(409, "审批已完成，不可修改")

    if body.team_score is not None:
        offer.team_score = body.team_score
    if body.strengths is not None:
        offer.strengths = body.strengths
    if body.capability_gaps is not None:
        offer.capability_gaps = body.capability_gaps
    if body.risks_note is not None:
        offer.risks_note = body.risks_note
    if body.suggested_salary is not None:
        offer.suggested_salary = body.suggested_salary
    if body.probation_goal is not None:
        offer.probation_goal = body.probation_goal
    if body.training_plan is not None:
        offer.training_plan = body.training_plan
    if body.mentor_id is not None:
        try:
            offer.mentor_id = uuid.UUID(body.mentor_id)
        except (ValueError, TypeError):
            offer.mentor_id = None
    if body.conversion_criteria is not None:
        offer.conversion_criteria = body.conversion_criteria
    if body.elimination_criteria is not None:
        offer.elimination_criteria = body.elimination_criteria

    # Recompute final_score if team_score updated
    if body.team_score is not None:
        offer.final_score = _compute_final_score(
            float(offer.resume_score) if offer.resume_score is not None else None,
            float(offer.r1_score) if offer.r1_score is not None else None,
            float(offer.r2_score) if offer.r2_score is not None else None,
            float(offer.practical_score) if offer.practical_score is not None else None,
            float(offer.team_score) if offer.team_score is not None else None,
        )

    await db.flush()
    return ok(serialize_offer(offer, current))


# ── POST /api/offers/{id}/approve ────────────────────────────

@router.post("/offers/{offer_id}/approve")
async def approve_offer(
    offer_id: str,
    body: ApproveOfferBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("录用审批单不存在")

    # Reject requires reason
    if body.result == "reject" and not body.reason:
        return fail(400, "拒绝录用须填写原因")

    # Validate 9 required approval fields
    _REQUIRED_FIELDS = [
        ("strengths", "strengths"),
        ("capability_gaps", "capabilityGaps"),
        ("risks_note", "risksNote"),
        ("suggested_salary", "suggestedSalary"),
        ("probation_goal", "probationGoal"),
        ("training_plan", "trainingPlan"),
        ("mentor_id", "mentorId"),
        ("conversion_criteria", "conversionCriteria"),
        ("elimination_criteria", "eliminationCriteria"),
    ]
    for attr, label in _REQUIRED_FIELDS:
        if not getattr(offer, attr, None):
            return fail(400, f"审批内容不完整: 缺少 {label}")

    # Load candidate
    cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
    candidate = cand_r.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

    # Write approval
    offer.approver_id = current.id
    offer.approved_at = datetime.utcnow()
    offer.result = body.result
    offer.reject_reason = body.reason if body.result == "reject" else None

    hired_employee: Optional[Employee] = None

    # 一个审批动作要同时迁移 offer 和 candidate 两个实体的状态，
    # 而 get_db() 只在路由抛出异常时才 rollback（见 app/database.py:28-37）；
    # 若第一个 transition() 已 flush 成功、第二个才失败，直接 return fail()
    # 会导致前者被静默提交，造成"offer 显示已批准但候选人未同步"的脏数据。
    # 因此第二个 transition() 失败时必须显式 rollback 撤销第一个的 flush。
    if body.result == "reject":
        try:
            await transition(
                db, "offer", offer, "rejected",
                actor_id=current.id, actor_name=current.username,
                reason=f"录用审批-拒绝: {body.reason}", skip_block_check=True,
            )
        except StateError as e:
            return fail(409, e.message)
        try:
            await transition(
                db, "candidate", candidate, "rejected",
                actor_id=current.id, actor_name=current.username,
                reason=f"录用审批-拒绝: {body.reason}", skip_block_check=True,
            )
        except StateError as e:
            await db.rollback()
            return fail(409, e.message)
    else:
        try:
            await transition(
                db, "offer", offer, "approved",
                actor_id=current.id, actor_name=current.username,
                reason=f"录用审批-{body.result}", skip_block_check=True,
            )
        except StateError as e:
            return fail(409, e.message)
        try:
            await transition(
                db, "candidate", candidate, "hired",
                actor_id=current.id, actor_name=current.username,
                reason=f"录用审批-{body.result}", skip_block_check=True,
            )
        except StateError as e:
            await db.rollback()
            return fail(409, e.message)

        # Auto-create Employee
        pos_r = await db.execute(
            select(Position).where(Position.id == offer.position_id)
        ) if offer.position_id else None
        position = pos_r.scalar_one_or_none() if pos_r else None

        hired_employee = Employee(
            candidate_id=offer.candidate_id,
            position_id=offer.position_id,
            name=candidate.name,
            status="pending_onboard",
            department=position.department if position and hasattr(position, "department") else None,
        )
        db.add(hired_employee)

    await db.flush()

    await write_audit(
        db,
        actor=current.username,
        action=f"录用审批-{body.result}",
        section="offer",
    )

    return ok({
        "offerId": str(offer.id),
        "status": offer.status,
        "finalScore": float(offer.final_score or 0),
        "result": offer.result,
        "hiredEmployeeId": str(hired_employee.id) if hired_employee else None,
    })


# ── GET /api/offers ───────────────────────────────────────────

async def _require_offer_read(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """录用审批单的只读权限（any-of）。

    列表原先要求 `offer:approve`（只有 ceo/manager 有），导致 HR 打开招聘看板时
    「待录用审批」恒为 0 —— 403 被前端静默降级成空数据，看不出是权限问题。
    招聘链路上 HR 需要看到审批进度，故放开只读；写操作仍要 offer:approve。
    注意 require_permission(*keys) 是 AND 语义，这里要的是 OR，只能自己写。
    """
    for key in ("offer:approve", "resume:view", "recruitment_request:confirm"):
        if current.has(key):
            return current
    raise PermissionError_("无权限: 查看录用审批")


@router.get("/offers")
async def list_offers(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    offset = (page - 1) * pageSize

    total_r = await db.execute(select(func.count()).select_from(OfferApproval))
    total = total_r.scalar() or 0

    offers_r = await db.execute(
        select(OfferApproval).order_by(OfferApproval.created_at.desc()).offset(offset).limit(pageSize)
    )
    offers = offers_r.scalars().all()

    # Enrich with candidate/position names
    items = []
    for offer in offers:
        item = serialize_offer(offer, current)

        cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
        cand = cand_r.scalar_one_or_none()
        item["candidateName"] = cand.name if cand else None

        if offer.position_id:
            pos_r = await db.execute(select(Position).where(Position.id == offer.position_id))
            pos = pos_r.scalar_one_or_none()
            item["positionName"] = pos.name if pos else None
        else:
            item["positionName"] = None

        items.append(item)

    return ok({
        "list": items,
        "total": total,
        "page": page,
        "pageSize": pageSize,
    })


# ── GET /api/offers/{id} ──────────────────────────────────────

@router.get("/offers/{offer_id}")
async def get_offer(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("录用审批单不存在")
    return ok(serialize_offer(offer, current))
