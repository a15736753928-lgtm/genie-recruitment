"""Offer 管理模块（2026-08-02 重构，8 态状态机）。

状态: draft(待HR发起) → pending_approval(待审批) → approved(审批通过·可发送)
      → sent(已发送) → accepted(已接受) / declined(候选人拒绝)
  approved/sent → expired(超时·视为放弃) / voided(已作废: 审批不通过/HR撤销)

核心动作:
  POST /offers                 — 新建草稿(draft)
  POST /offers/{id}/submit     — 提交审批 → 实例化逐级审批记录
  POST /offers/{id}/approve    — 逐级审批（按当前步骤角色校验）
  POST /offers/{id}/send       — 发送：生成 token + 一键发候选人邮箱
  GET  /public/offer/{token}   — 候选人线上确认页(公开路由, offer_public.py)
  接受/拒绝 → offer_public.py；作废/重新发起 → 本文件

薪资脱敏: compensation/suggested_salary 仅在 salary:view 时下发。
"""
from __future__ import annotations

from app.prompts import render_prompt

import asyncio
import io
import logging
import secrets
import uuid
from datetime import datetime, timedelta, date
from typing import Any, Optional, List

from fastapi import APIRouter, Depends, Query, Request, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Candidate, Position, CandidateAIAnalysis
from app.models.phase1 import Interview, OfferApproval, ResumeScore, RecruitmentRequest
from app.models.probation import Employee
from app.models.settings import SystemSetting
from app.models.offer_module import (
    OfferTemplate, OfferApprovalFlow, OfferApprovalRecord, OfferAttachment,
)
from app.core.security import get_current_user, require_permission, CurrentUser, PermissionError_
from app.core.state_machine import transition, StateError
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
from app.utils.clock import iso_utc
from app.services.ai import llm_chat
from app.utils.llm_json import extract_json_object
from app.services.offer.offer_pdf import build_offer_vars, fill_template_pdf

logger = logging.getLogger("genie.offer")
router = APIRouter(tags=["Offer管理"])

OFFER_STATUS_LABELS: dict[str, str] = {
    "draft": "待HR发起", "pending_approval": "待审批", "approved": "已审批待发送",
    "sent": "已发送", "accepted": "候选人已接受", "declined": "候选人拒绝",
    "expired": "Offer已失效", "voided": "已作废",
}

# 在途状态（同一候选人同一时间最多一条）
_ACTIVE_STATUSES = ["draft", "pending_approval", "approved", "sent"]


# ── Serializer ───────────────────────────────────────────────

_LIST_SEPARATORS = "、\n\r;；,，"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    for sep in _LIST_SEPARATORS[1:]:
        text = text.replace(sep, _LIST_SEPARATORS[0])
    return [seg.strip() for seg in text.split(_LIST_SEPARATORS[0]) if seg.strip()]


def _d(v: Any) -> Optional[str]:
    return v.isoformat() if v is not None else None


def _dt(v: Any) -> Optional[str]:
    return iso_utc(v) if v is not None else None


def serialize_offer(offer: OfferApproval, viewer: Optional[CurrentUser] = None) -> dict:
    show_salary = viewer is not None and viewer.has("salary:view")
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
        "strengths": strengths,
        "capabilityGaps": gaps,
        "gaps": gaps,
        "risksNote": risks,
        "risks": risks,
        "strengthsText": offer.strengths,
        "capabilityGapsText": offer.capability_gaps,
        "risksNoteText": offer.risks_note,
        "suggestedSalary": offer.suggested_salary if show_salary else None,
        "salaryMasked": not show_salary,
        "compensation": offer.compensation if show_salary else None,
        "employmentTerms": offer.employment_terms,
        "otherTerms": offer.other_terms,
        "probationGoal": offer.probation_goal,
        "trainingPlan": offer.training_plan,
        "mentorId": str(offer.mentor_id) if offer.mentor_id else None,
        "conversionCriteria": offer.conversion_criteria,
        "eliminationCriteria": offer.elimination_criteria,
        "approverId": str(offer.approver_id) if offer.approver_id else None,
        "approvedAt": _dt(offer.approved_at),
        "rejectReason": offer.reject_reason,
        "aiResult": offer.ai_result,
        "result": offer.result,
        "status": offer.status,
        "expectedOnboardDate": _d(offer.expected_onboard_date),
        "probationMonths": offer.probation_months,
        "workLocation": offer.work_location,
        "department": offer.department,
        "channel": offer.channel,
        "backgroundCheckRequired": bool(offer.background_check_required),
        "backgroundCheckResult": offer.background_check_result,
        "createdBy": str(offer.created_by) if offer.created_by else None,
        "updatedBy": str(offer.updated_by) if offer.updated_by else None,
        "submittedAt": _dt(offer.submitted_at),
        "submittedBy": str(offer.submitted_by) if offer.submitted_by else None,
        "approvalFlowId": str(offer.approval_flow_id) if offer.approval_flow_id else None,
        "currentApprovalStep": offer.current_approval_step,
        "prefillSource": offer.prefill_source,
        "profileVersion": offer.profile_version,
        "templateId": str(offer.template_id) if offer.template_id else None,
        "sentAt": _dt(offer.sent_at),
        "sentBy": str(offer.sent_by) if offer.sent_by else None,
        "sentChannel": offer.sent_channel,
        "tokenExpiresAt": _dt(offer.token_expires_at),
        "viewedAt": _dt(offer.viewed_at),
        "validityDays": offer.validity_days,
        "expiresAt": _dt(offer.expires_at),
        "voidReason": offer.void_reason,
        "voidedBy": str(offer.voided_by) if offer.voided_by else None,
        "voidedAt": _dt(offer.voided_at),
        "expiredAt": _dt(offer.expired_at),
        "employeeId": str(offer.employee_id) if offer.employee_id else None,
        "declineReason": offer.decline_reason,
        "createdAt": _dt(offer.created_at),
        "updatedAt": _dt(offer.updated_at),
    }


# ── Request schemas ───────────────────────────────────────────

class CreateOfferBody(BaseModel):
    model_config = {"populate_by_name": True}
    candidate_id: str = Field(alias="candidateId")
    position_id: str = Field(alias="positionId")
    team_score: Optional[float] = Field(None, alias="teamScore")


class UpdateOfferBody(BaseModel):
    model_config = {"populate_by_name": True}
    # 旧 9 项审批内容
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
    # 核心聘用信息
    expected_onboard_date: Optional[date] = Field(None, alias="expectedOnboardDate")
    probation_months: Optional[int] = Field(None, alias="probationMonths")
    work_location: Optional[str] = Field(None, alias="workLocation")
    department: Optional[str] = None
    channel: Optional[str] = None
    background_check_required: Optional[bool] = Field(None, alias="backgroundCheckRequired")
    background_check_result: Optional[str] = Field(None, alias="backgroundCheckResult")
    validity_days: Optional[int] = Field(None, alias="validityDays")
    # 结构化 JSON
    compensation: Optional[dict] = None
    employment_terms: Optional[dict] = Field(None, alias="employmentTerms")
    other_terms: Optional[dict] = Field(None, alias="otherTerms")
    template_id: Optional[str] = Field(None, alias="templateId")
    approval_flow_id: Optional[str] = Field(None, alias="approvalFlowId")


class ApproveStepBody(BaseModel):
    model_config = {"populate_by_name": True}
    action: str  # approve | reject
    opinion: Optional[str] = None


class VoidBody(BaseModel):
    model_config = {"populate_by_name": True}
    reason: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────

def _compute_final_score(
    resume: Optional[float], r1: Optional[float], r2: Optional[float],
    practical: Optional[float], team: Optional[float],
) -> float:
    return round(
        (resume or 0) * 0.15 + (r1 or 0) * 0.25 + (r2 or 0) * 0.40
        + (practical or 0) * 0.10 + (team or 0) * 0.10, 2,
    )


async def _load_offer(db: AsyncSession, offer_id: str) -> Optional[OfferApproval]:
    try:
        uid = uuid.UUID(offer_id)
    except (ValueError, TypeError):
        return None
    r = await db.execute(select(OfferApproval).where(OfferApproval.id == uid))
    return r.scalar_one_or_none()


async def _load_candidate(db: AsyncSession, candidate_id) -> Optional[Candidate]:
    r = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    return r.scalar_one_or_none()


async def _load_position(db: AsyncSession, position_id) -> Optional[Position]:
    if not position_id:
        return None
    try:
        uid = uuid.UUID(str(position_id))
    except (ValueError, TypeError):
        return None
    r = await db.execute(select(Position).where(Position.id == uid))
    return r.scalar_one_or_none()


async def _load_company_info(db: AsyncSession) -> dict:
    """读取公司信息（SystemSetting key=company_info，读=登录可见）。"""
    r = await db.execute(select(SystemSetting).where(SystemSetting.key == "company_info"))
    setting = r.scalar_one_or_none()
    return setting.value if setting and setting.value else {}


async def _render_offer_pdf(offer, db: AsyncSession) -> bytes:
    """模板填充生成 Offer 正式 PDF（同步 fill 放线程池，避免阻塞事件循环）。"""
    candidate = await _load_candidate(db, offer.candidate_id)
    position = await _load_position(db, offer.position_id)
    company = await _load_company_info(db)
    data = build_offer_vars(offer, candidate, position, company)
    return await asyncio.to_thread(fill_template_pdf, data)


async def _safe_transition(db, entity_type, entity, to_status, *, actor_id=None, actor_name="系统", reason=None):
    """单实体迁移；失败返回错误串，成功返回 None。"""
    try:
        await transition(db, entity_type, entity, to_status,
                         actor_id=actor_id, actor_name=actor_name, reason=reason,
                         skip_block_check=True)
        return None
    except StateError as e:
        return e.message


async def _build_ai_advice(candidate: Candidate, offer: OfferApproval) -> Optional[dict]:
    """Call LLM for hire/no-hire recommendation; return None on failure."""
    prompt = render_prompt('talent/offer.md', {'candidate_name': candidate.name if candidate else '未知', 'resume_score': float(offer.resume_score) if offer.resume_score else 'N/A', 'r1_score': float(offer.r1_score) if offer.r1_score else 'N/A', 'r2_score': float(offer.r2_score) if offer.r2_score else 'N/A', 'practical_score': float(offer.practical_score) if offer.practical_score else 'N/A', 'team_score': float(offer.team_score) if offer.team_score else 'N/A', 'final_score': float(offer.final_score) if offer.final_score else 'N/A'}, 'Prompt 1')
    try:
        raw_text = await llm_chat([{"role": "user", "content": prompt}], max_tokens=2048)
        raw_dict = extract_json_object(raw_text.strip())
        if raw_dict is None:
            raise ValueError("模型未返回可解析的 JSON")
        evidence = raw_dict.get("evidence", [])
        if not evidence:
            return None
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


def _text_of(value: Any) -> Optional[str]:
    """把字符串/列表/JSON 统一转为单段文本；空值返回 None。"""
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return "；".join(parts) if parts else None
    if isinstance(value, dict):
        # 结构化 JSON（如 position.conversion_criteria）取常见键
        for k in ("summary", "criteria", "description", "content"):
            if value.get(k):
                return _text_of(value[k])
        return None
    s = str(value).strip()
    return s or None


async def _prefill_offer_content(
    db: AsyncSession,
    candidate: Candidate,
    position: Optional[Position],
    offer: OfferApproval,
) -> dict:
    """候选人信息自动流转：Offer 草稿创建时，从既有数据源预填 9 项审批内容。

    数据来源优先级（低 → 高，后写覆盖先写）：
      resume_ai               — 简历 AI 解析（candidate_ai_analyses.highlights/risks、
                                resume_scores.advice）
      recruitment_request     — 招聘需求单（probation_goal / elimination_criteria）
      position                — 岗位配置（conversion_criteria）
      offer_ai                — 本次 Offer 的 LLM 录用建议（ai_advice.strengths/risks）

    返回 prefill_source 标记（字段名 → 来源），供前端展示"引用 vs 补录"。
    """
    source: dict[str, str] = {}
    resume_advice: dict = {}

    # ── 简历 AI 解析（主档）──
    ai_r = await db.execute(
        select(CandidateAIAnalysis).where(CandidateAIAnalysis.candidate_id == candidate.id)
    )
    ai_analysis = ai_r.scalar_one_or_none()
    highlights = _text_of(ai_analysis.highlights if ai_analysis else None)
    risks_ai = _text_of(ai_analysis.risks if ai_analysis else None)

    if position is not None:
        rs_r = await db.execute(
            select(ResumeScore).where(and_(
                ResumeScore.candidate_id == candidate.id,
                ResumeScore.position_id == position.id,
            ))
        )
        rs = rs_r.scalar_one_or_none()
        if rs and rs.advice:
            resume_advice = rs.advice if isinstance(rs.advice, dict) else {}

    strengths_base = _text_of(resume_advice.get("strengths")) or highlights
    gaps_base = _text_of(resume_advice.get("gaps")) or _text_of(resume_advice.get("capabilityGaps"))
    risks_base = _text_of(resume_advice.get("risks")) or risks_ai

    # ── 招聘需求单（最近一条已发布岗位的需求单）──
    req_goal = req_elim = None
    if position is not None:
        req_r = await db.execute(
            select(RecruitmentRequest)
            .where(RecruitmentRequest.position_id == position.id)
            .order_by(RecruitmentRequest.created_at.desc())
            .limit(1)
        )
        req = req_r.scalar_one_or_none()
        if req:
            req_goal = req.probation_goal or None
            req_elim = req.elimination_criteria or None

    # ── 岗位配置 ──
    pos_conv = _text_of(position.conversion_criteria) if position else None

    # ── Offer AI 建议（创建时刚生成，优先级最高）──
    advice = offer.ai_advice if isinstance(offer.ai_advice, dict) else {}
    strengths_ai = _text_of(advice.get("strengths"))
    risks_ai_offer = _text_of(advice.get("risks"))

    def _apply(field: str, value: Optional[str], src: str) -> None:
        if value:
            setattr(offer, field, value)
            source[field] = src

    # 优势：Offer AI → 简历 AI
    if strengths_ai:
        _apply("strengths", strengths_ai, "offer_ai")
    elif strengths_base:
        _apply("strengths", strengths_base, "resume_ai")

    # 风险提示：Offer AI → 简历 AI
    if risks_ai_offer:
        _apply("risks_note", risks_ai_offer, "offer_ai")
    elif risks_base:
        _apply("risks_note", risks_base, "resume_ai")

    # 能力缺口：简历 AI（无 Offer AI 来源）
    if gaps_base:
        _apply("capability_gaps", gaps_base, "resume_ai")

    # 试用期目标 / 淘汰条件：招聘需求单
    if req_goal:
        _apply("probation_goal", req_goal, "recruitment_request")
    if req_elim:
        _apply("elimination_criteria", req_elim, "recruitment_request")

    # 转正条件：岗位配置
    if pos_conv:
        _apply("conversion_criteria", pos_conv, "position")

    # 培训计划 / 建议薪酬 / 带教人：前置阶段无稳定数据源，留给 HR 补录
    return source


async def _require_offer_read(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    for key in ("offer:approve", "resume:view", "recruitment_request:confirm"):
        if current.has(key):
            return current
    raise PermissionError_("无权限: 查看录用审批")


async def _load_records(db: AsyncSession, offer_id: uuid.UUID) -> list[OfferApprovalRecord]:
    r = await db.execute(
        select(OfferApprovalRecord).where(OfferApprovalRecord.offer_id == offer_id)
        .order_by(OfferApprovalRecord.step)
    )
    return list(r.scalars().all())


def _can_approve_step(record: OfferApprovalRecord, current: CurrentUser) -> bool:
    if current.has("system:manage"):
        return True
    if record.approver_id and record.approver_id == current.id:
        return True
    if record.role_code and record.role_code in current.roles:
        return True
    return False


def _serialize_record(r: OfferApprovalRecord) -> dict:
    return {
        "id": str(r.id),
        "step": r.step,
        "roleCode": r.role_code,
        "roleName": r.role_name,
        "approverId": str(r.approver_id) if r.approver_id else None,
        "approverName": r.approver_name,
        "action": r.action,
        "opinion": r.opinion,
        "decidedAt": _dt(r.decided_at),
        "status": r.status,
    }


async def _enrich_offer(item: dict, offer: OfferApproval, db: AsyncSession,
                        current: CurrentUser) -> dict:
    """列表/详情：附上候选人/岗位/操作人名称。"""
    cand = await _load_candidate(db, offer.candidate_id)
    item["candidateName"] = cand.name if cand else None
    item["candidatePhone"] = cand.phone if cand else None
    item["candidateEmail"] = cand.email if cand else None
    item["interviewer"] = cand.interviewer if cand else None
    pos = await _load_position(db, offer.position_id)
    item["positionName"] = pos.name if pos else None
    if not item.get("department"):
        item["department"] = pos.department if pos else None
    if offer.created_by:
        from app.models.auth import User
        ur = await db.execute(select(User).where(User.id == offer.created_by))
        u = ur.scalar_one_or_none()
        item["operatorName"] = u.username if u else None
    else:
        item["operatorName"] = None
    return item


def _confirm_url(request: Request, token: str) -> str:
    base = str(request.base_url)
    return f"{base}public/offer/{token}"


# ═══════════════════════════════════════════════════════════
# 列表 / 可发起候选人
# ═══════════════════════════════════════════════════════════

@router.get("/offers")
async def list_offers(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    keyword: Optional[str] = None,
    positionId: Optional[str] = None,
    status: Optional[str] = None,
    createdFrom: Optional[str] = None,
    createdTo: Optional[str] = None,
    onboardFrom: Optional[str] = None,
    onboardTo: Optional[str] = None,
    ownerId: Optional[str] = None,
    channel: Optional[str] = None,
    backgroundCheckRequired: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    conditions = []
    if keyword and keyword.strip():
        kw = f"%{keyword.strip()}%"
        conditions.append(
            or_(Candidate.name.ilike(kw), Candidate.phone.ilike(kw))
        )
    if positionId:
        try:
            conditions.append(OfferApproval.position_id == uuid.UUID(positionId))
        except (ValueError, TypeError):
            return ok({"list": [], "total": 0, "page": page, "pageSize": pageSize})
    if status:
        conditions.append(OfferApproval.status == status)
    if createdFrom:
        try:
            conditions.append(OfferApproval.created_at >= datetime.fromisoformat(createdFrom))
        except ValueError:
            pass
    if createdTo:
        try:
            conditions.append(OfferApproval.created_at <= datetime.fromisoformat(createdTo))
        except ValueError:
            pass
    if onboardFrom:
        try:
            conditions.append(OfferApproval.expected_onboard_date >= date.fromisoformat(onboardFrom))
        except ValueError:
            pass
    if onboardTo:
        try:
            conditions.append(OfferApproval.expected_onboard_date <= date.fromisoformat(onboardTo))
        except ValueError:
            pass
    if ownerId:
        try:
            conditions.append(OfferApproval.created_by == uuid.UUID(ownerId))
        except (ValueError, TypeError):
            pass
    if channel:
        conditions.append(OfferApproval.channel == channel)
    if backgroundCheckRequired is not None:
        conditions.append(OfferApproval.background_check_required == backgroundCheckRequired)

    base = select(OfferApproval).join(Candidate, OfferApproval.candidate_id == Candidate.id)
    if conditions:
        base = base.where(and_(*conditions))

    total_r = await db.execute(select(func.count()).select_from(base.subquery()))
    total = total_r.scalar() or 0

    offers_r = await db.execute(
        base.order_by(OfferApproval.created_at.desc()).offset((page - 1) * pageSize).limit(pageSize)
    )
    offers = offers_r.scalars().all()

    items = []
    for offer in offers:
        item = serialize_offer(offer, current)
        await _enrich_offer(item, offer, db, current)
        items.append(item)

    return ok({"list": items, "total": total, "page": page, "pageSize": pageSize})


@router.get("/offers/available-candidates")
async def list_available_candidates(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    """可发起 Offer 的候选人：pending_offer 且无在途 Offer。"""
    active_r = await db.execute(
        select(OfferApproval.candidate_id).where(OfferApproval.status.in_(_ACTIVE_STATUSES))
    )
    active_ids = {r[0] for r in active_r.all()}

    cand_r = await db.execute(
        select(Candidate).where(Candidate.status == "pending_offer").order_by(Candidate.score.desc())
    )
    candidates = cand_r.scalars().all()

    items = []
    for c in candidates:
        if c.id in active_ids:
            continue
        pos = await _load_position(db, c.position_id)
        items.append({
            "id": str(c.id),
            "name": c.name,
            "phone": c.phone,
            "email": c.email,
            "positionId": str(c.position_id) if c.position_id else None,
            "position": pos.name if pos else None,
            "department": pos.department if pos else None,
            "score": c.score,
            "interviewer": c.interviewer,
        })
    return ok({"list": items, "total": len(items)})


# ═══════════════════════════════════════════════════════════
# 新建草稿 / 编辑
# ═══════════════════════════════════════════════════════════

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

    candidate = await _load_candidate(db, cand_uid)
    if not candidate:
        return not_found("候选人不存在")
    if candidate.status != "pending_offer":
        return fail(409, "候选人须处于待发Offer状态")

    # 同一候选人禁止重复发起多条有效 Offer
    existing_r = await db.execute(
        select(OfferApproval).where(
            OfferApproval.candidate_id == cand_uid,
            OfferApproval.status.in_(_ACTIVE_STATUSES),
        )
    )
    if existing_r.scalar_one_or_none():
        return fail(409, "该候选人已有在途 Offer，不能重复发起")

    try:
        pos_uid = uuid.UUID(body.position_id)
    except (ValueError, TypeError):
        pos_uid = None

    resume_score_val: Optional[float] = None
    if pos_uid:
        rs_r = await db.execute(
            select(ResumeScore).where(and_(
                ResumeScore.candidate_id == cand_uid, ResumeScore.position_id == pos_uid,
            ))
        )
        rs = rs_r.scalar_one_or_none()
        if rs:
            resume_score_val = float(rs.total) if rs.total is not None else None

    r1 = r2 = practical = None
    ivs_r = await db.execute(select(Interview).where(Interview.candidate_id == cand_uid))
    for iv in ivs_r.scalars().all():
        if iv.round == "r1" and iv.composite_score is not None:
            r1 = float(iv.composite_score)
        if iv.round == "r2":
            if iv.composite_score is not None:
                r2 = float(iv.composite_score)
            if iv.practical_score is not None:
                practical = float(iv.practical_score)

    team = body.team_score
    final = _compute_final_score(resume_score_val, r1, r2, practical, team)

    position = await _load_position(db, pos_uid)
    offer = OfferApproval(
        candidate_id=cand_uid,
        position_id=pos_uid,
        resume_score=resume_score_val,
        r1_score=r1,
        r2_score=r2,
        practical_score=practical,
        team_score=team,
        final_score=final,
        status="draft",
        created_by=current.id,
        department=position.department if position else None,
        validity_days=7,
    )
    db.add(offer)
    await db.flush()

    ai_advice = await _build_ai_advice(candidate, offer)
    if ai_advice:
        offer.ai_advice = ai_advice
        offer.ai_result = ai_advice.get("result")
        await db.flush()

    # 候选人信息自动流转：从简历 AI / 需求单 / 岗位配置预填 9 项审批内容
    offer.prefill_source = await _prefill_offer_content(db, candidate, position, offer)
    offer.profile_version = candidate.profile_version if hasattr(candidate, "profile_version") else None
    await db.flush()

    await write_audit(db, actor=current.username, action=f"发起Offer草稿: {candidate.name}", section="offer")

    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    return ok(item)


@router.put("/offers/{offer_id}")
async def update_offer(
    offer_id: str,
    body: UpdateOfferBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status not in ("draft", "pending_approval"):
        return fail(409, "当前状态不可编辑")

    show_salary = current.has("salary:view")

    # 旧 9 项
    if body.team_score is not None:
        offer.team_score = body.team_score
    for attr in ("strengths", "capability_gaps", "risks_note", "probation_goal",
                 "training_plan", "conversion_criteria", "elimination_criteria"):
        v = getattr(body, attr)
        if v is not None:
            setattr(offer, attr, v)
    if body.mentor_id is not None:
        try:
            offer.mentor_id = uuid.UUID(body.mentor_id)
        except (ValueError, TypeError):
            offer.mentor_id = None
    # 薪资仅 salary:view 可写
    if body.suggested_salary is not None and show_salary:
        offer.suggested_salary = body.suggested_salary
    if body.compensation is not None and show_salary:
        offer.compensation = body.compensation

    # 核心聘用信息
    if body.expected_onboard_date is not None:
        offer.expected_onboard_date = body.expected_onboard_date
    if body.probation_months is not None:
        offer.probation_months = body.probation_months
    if body.work_location is not None:
        offer.work_location = body.work_location
    if body.department is not None:
        offer.department = body.department
    if body.channel is not None:
        offer.channel = body.channel
    if body.background_check_required is not None:
        offer.background_check_required = body.background_check_required
    if body.background_check_result is not None:
        offer.background_check_result = body.background_check_result
    if body.validity_days is not None:
        offer.validity_days = max(1, min(body.validity_days, 30))
    if body.employment_terms is not None:
        offer.employment_terms = body.employment_terms
    if body.other_terms is not None:
        offer.other_terms = body.other_terms
    if body.template_id is not None:
        try:
            offer.template_id = uuid.UUID(body.template_id)
        except (ValueError, TypeError):
            offer.template_id = None
    if body.approval_flow_id is not None:
        try:
            offer.approval_flow_id = uuid.UUID(body.approval_flow_id)
        except (ValueError, TypeError):
            offer.approval_flow_id = None

    if body.team_score is not None:
        offer.final_score = _compute_final_score(
            float(offer.resume_score) if offer.resume_score is not None else None,
            float(offer.r1_score) if offer.r1_score is not None else None,
            float(offer.r2_score) if offer.r2_score is not None else None,
            float(offer.practical_score) if offer.practical_score is not None else None,
            float(offer.team_score) if offer.team_score is not None else None,
        )
    offer.updated_by = current.id
    await db.flush()
    await write_audit(db, actor=current.username, action="编辑Offer表单", section="offer")

    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    return ok(item)


# ═══════════════════════════════════════════════════════════
# 提交审批 / 逐级审批
# ═══════════════════════════════════════════════════════════

_REQUIRED_SUBMIT_FIELDS = [
    ("strengths", "优势"), ("capability_gaps", "能力缺口"), ("risks_note", "风险提示"),
    ("suggested_salary", "建议薪酬"), ("probation_goal", "试用期目标"),
    ("training_plan", "培训计划"), ("conversion_criteria", "转正条件"),
    ("elimination_criteria", "淘汰条件"), ("expected_onboard_date", "预计入职日期"),
    ("probation_months", "试用期时长"),
]


@router.post("/offers/{offer_id}/submit")
async def submit_offer(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status != "draft":
        return fail(409, f"仅草稿可提交审批，当前状态 {OFFER_STATUS_LABELS.get(offer.status, offer.status)}")

    for attr, label in _REQUIRED_SUBMIT_FIELDS:
        if not getattr(offer, attr, None):
            return fail(400, f"审批内容不完整: 缺少 {label}")

    if not offer.approval_flow_id:
        return fail(400, "请选择审批流")

    flow_r = await db.execute(select(OfferApprovalFlow).where(OfferApprovalFlow.id == offer.approval_flow_id))
    flow = flow_r.scalar_one_or_none()
    if not flow or not flow.enabled:
        return fail(400, "审批流不存在或已停用")

    steps = flow.steps or []
    if not steps:
        return fail(400, "审批流未配置审批步骤")

    err = await _safe_transition(db, "offer", offer, "pending_approval",
                                 actor_id=current.id, actor_name=current.username,
                                 reason="提交审批")
    if err:
        return fail(409, err)

    offer.submitted_at = datetime.utcnow()
    offer.submitted_by = current.id
    offer.current_approval_step = 0
    for st in steps:
        rec = OfferApprovalRecord(
            offer_id=offer.id,
            step=int(st.get("step", 0)),
            role_code=st.get("role_code") or "manager",
            role_name=st.get("role_name"),
            approver_id=st.get("approver_id"),
            status="pending",
        )
        db.add(rec)
    await db.flush()

    await write_audit(db, actor=current.username, action="提交Offer审批", section="offer")
    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    return ok(item)


@router.post("/offers/{offer_id}/approve")
async def approve_offer(
    offer_id: str,
    body: ApproveStepBody,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """逐级审批。动作 approve/reject，审批人是当前步骤对应角色的任一用户。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status != "pending_approval":
        return fail(409, f"Offer 不在待审批状态，当前 {OFFER_STATUS_LABELS.get(offer.status, offer.status)}")

    records = await _load_records(db, offer.id)
    idx = offer.current_approval_step
    if idx >= len(records):
        return fail(409, "审批步骤异常")
    record = records[idx]
    if record.status != "pending":
        return fail(409, "当前步骤已处理")
    if not _can_approve_step(record, current):
        raise PermissionError_("无权限: 非当前步骤审批人")

    action = body.action
    if action not in ("approve", "reject"):
        return fail(400, "action 须为 approve 或 reject")
    if action == "reject" and not (body.opinion or "").strip():
        return fail(400, "审批不通过必须填写意见")

    record.approver_id = current.id
    record.approver_name = current.username
    record.action = action
    record.opinion = body.opinion
    record.decided_at = datetime.utcnow()
    record.status = "approved" if action == "approve" else "rejected"

    if action == "reject":
        # 任一环节不通过 → 作废，余下步骤取消；候选人保持 pending_offer 可重新发起
        await transition(db, "offer", offer, "voided",
                         actor_id=current.id, actor_name=current.username,
                         reason=f"审批不通过: {body.opinion}", skip_block_check=True)
        offer.void_reason = body.opinion
        offer.voided_by = current.id
        offer.voided_at = datetime.utcnow()
        for r in records[idx + 1:]:
            r.status = "cancelled"
        await db.flush()
        await write_audit(db, actor=current.username, action=f"审批不通过: {body.opinion}", section="offer")
        return ok({"offerId": str(offer.id), "status": offer.status, "result": "reject"})

    # approve
    if idx == len(records) - 1:
        # 末级通过 → approved（可发送）
        await transition(db, "offer", offer, "approved",
                         actor_id=current.id, actor_name=current.username,
                         reason="逐级审批全部通过", skip_block_check=True)
        offer.approver_id = current.id
        offer.approved_at = datetime.utcnow()
        offer.result = "recommend"
        await db.flush()
        await write_audit(db, actor=current.username, action="Offer审批通过", section="offer")
        return ok({"offerId": str(offer.id), "status": offer.status, "result": "approved"})
    else:
        offer.current_approval_step = idx + 1
        await db.flush()
        await write_audit(db, actor=current.username,
                          action=f"审批通过第{record.step}级({record.role_name or record.role_code})", section="offer")
        return ok({"offerId": str(offer.id), "status": offer.status, "currentApprovalStep": idx + 1})


# ═══════════════════════════════════════════════════════════
# 发送 / 补发 / 作废 / 重新发起
# ═══════════════════════════════════════════════════════════

async def _do_send(db: AsyncSession, offer: OfferApproval, current: CurrentUser, request: Request,
                   first_send: bool = True):
    """发送/补发 Offer；生成 token + 发邮件。

    first_send=True: approved→sent + candidate→offer_sent（新发）。
    first_send=False: offer 已 sent，仅重新生成 token + 延长有效期 + 重发（不迁移）。
    返回 (confirm_url, email_sent, message)。
    """
    from app.services.offer.emailer import send_offer_email

    candidate = await _load_candidate(db, offer.candidate_id)
    if not candidate:
        return None, False, "候选人不存在"

    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    validity = offer.validity_days or 7
    expires = now + timedelta(days=validity)

    if first_send:
        err = await _safe_transition(db, "offer", offer, "sent",
                                     actor_id=current.id, actor_name=current.username,
                                     reason="HR 发送Offer")
        if err:
            return None, False, err
        err2 = await _safe_transition(db, "candidate", candidate, "offer_sent",
                                      actor_id=current.id, actor_name=current.username,
                                      reason="Offer 已发送，等待候选人确认")
        if err2:
            await db.rollback()
            return None, False, err2

    offer.confirm_token = token
    offer.token_expires_at = expires
    offer.expires_at = expires
    offer.token_used_at = None  # 补发时清除旧 token 使用标记，允许再次确认
    offer.sent_at = now
    offer.sent_by = current.id
    offer.sent_channel = "email"
    await db.flush()

    url = _confirm_url(request, token)
    email_sent, message = await send_offer_email(db, offer, candidate, url)
    await write_audit(db, actor=current.username,
                      action=f"发送Offer({message})" if not email_sent else "发送Offer", section="offer")
    return url, email_sent, message


@router.post("/offers/{offer_id}/send")
async def send_offer(
    offer_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status != "approved":
        return fail(409, "审批通过后才能发送 Offer，当前状态 " + OFFER_STATUS_LABELS.get(offer.status, offer.status))

    url, email_sent, message = await _do_send(db, offer, current, request)
    if url is None:
        return fail(409, message or "发送失败")

    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    item["confirmUrl"] = url
    item["emailSent"] = email_sent
    item["emailMessage"] = message
    return ok(item)


@router.post("/offers/{offer_id}/resend")
async def resend_offer(
    offer_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    """补发：重新生成 token + 延长有效期 + 重发邮件（仅已发送状态）。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status != "sent":
        return fail(409, "仅「已发送」的 Offer 可补发")

    url, email_sent, message = await _do_send(db, offer, current, request, first_send=False)
    if url is None:
        return fail(409, message or "补发失败")

    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    item["confirmUrl"] = url
    item["emailSent"] = email_sent
    item["emailMessage"] = message
    return ok(item)


@router.post("/offers/{offer_id}/void")
async def void_offer(
    offer_id: str,
    body: VoidBody = None,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    """HR 手动作废。已接受/已拒绝/已失效为终态，不可作废。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status in ("accepted", "declined", "expired"):
        return fail(409, "已终态的 Offer 不可作废")
    if offer.status == "voided":
        return fail(409, "Offer 已作废")

    reason = body.reason if body else None
    was_sent = offer.status == "sent"
    err = await _safe_transition(db, "offer", offer, "voided",
                                 actor_id=current.id, actor_name=current.username,
                                 reason=reason or "HR 手动作废")
    if err:
        return fail(409, err)
    offer.void_reason = reason
    offer.voided_by = current.id
    offer.voided_at = datetime.utcnow()

    if was_sent:
        cand = await _load_candidate(db, offer.candidate_id)
        if cand:
            err2 = await _safe_transition(db, "candidate", cand, "pending_offer",
                                          actor_id=current.id, actor_name=current.username,
                                          reason="Offer 作废，候选人回流待发")
            if err2:
                await db.rollback()
                return fail(409, err2)
    await db.flush()
    await write_audit(db, actor=current.username, action=f"作废Offer: {reason or ''}", section="offer")
    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)
    return ok(item)


@router.post("/offers/{offer_id}/reinitiate")
async def reinitiate_offer(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    """重新发起：仅已作废/已拒绝的 Offer 可克隆新草稿，候选人回流 pending_offer。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status not in ("voided", "declined"):
        return fail(409, "仅已作废/已拒绝的 Offer 可重新发起")

    candidate = await _load_candidate(db, offer.candidate_id)
    if candidate and candidate.status != "pending_offer":
        err = await _safe_transition(db, "candidate", candidate, "pending_offer",
                                     actor_id=current.id, actor_name=current.username,
                                     reason="重新发起Offer，候选人回流待发")
        if err:
            return fail(409, err)

    new_offer = OfferApproval(
        candidate_id=offer.candidate_id,
        position_id=offer.position_id,
        resume_score=offer.resume_score,
        r1_score=offer.r1_score,
        r2_score=offer.r2_score,
        practical_score=offer.practical_score,
        team_score=offer.team_score,
        final_score=offer.final_score,
        ai_advice=offer.ai_advice,
        ai_result=offer.ai_result,
        strengths=offer.strengths,
        capability_gaps=offer.capability_gaps,
        risks_note=offer.risks_note,
        suggested_salary=offer.suggested_salary,
        probation_goal=offer.probation_goal,
        training_plan=offer.training_plan,
        mentor_id=offer.mentor_id,
        conversion_criteria=offer.conversion_criteria,
        elimination_criteria=offer.elimination_criteria,
        expected_onboard_date=offer.expected_onboard_date,
        probation_months=offer.probation_months,
        work_location=offer.work_location,
        department=offer.department,
        channel=offer.channel,
        background_check_required=offer.background_check_required,
        background_check_result=offer.background_check_result,
        validity_days=offer.validity_days,
        template_id=offer.template_id,
        approval_flow_id=offer.approval_flow_id,
        compensation=offer.compensation,
        employment_terms=offer.employment_terms,
        other_terms=offer.other_terms,
        status="draft",
        created_by=current.id,
    )
    db.add(new_offer)
    await db.flush()
    await write_audit(db, actor=current.username, action=f"重新发起Offer: 从 {offer.status}", section="offer")

    item = serialize_offer(new_offer, current)
    await _enrich_offer(item, new_offer, db, current)
    item["previousOfferId"] = str(offer.id)
    return ok(item)


# ═══════════════════════════════════════════════════════════
# 统计 / 我的待办 / 过期扫描（静态路径必须定义在 /offers/{offer_id} 之前）
# ═══════════════════════════════════════════════════════════

@router.get("/offers/stats")
async def offer_stats(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    counts: dict[str, int] = {}
    for st in OFFER_STATUS_LABELS:
        r = await db.execute(
            select(func.count()).select_from(OfferApproval).where(OfferApproval.status == st)
        )
        counts[st] = r.scalar() or 0

    # 拒绝原因统计（TOP，按文本归类）
    decl_r = await db.execute(
        select(OfferApproval.decline_reason, func.count())
        .where(OfferApproval.status == "declined", OfferApproval.decline_reason.isnot(None))
        .group_by(OfferApproval.decline_reason).order_by(func.count().desc()).limit(10)
    )
    declineReasons = [{"reason": reason or "未说明", "count": c} for reason, c in decl_r.all()]

    return ok({
        "statusCounts": counts,
        "total": sum(counts.values()),
        "declineReasons": declineReasons,
    })


@router.get("/offers/my-todos")
async def my_offer_todos(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """我的待审批 Offer：当前步骤角色命中我的角色。"""
    result = await db.execute(
        select(OfferApproval).where(OfferApproval.status == "pending_approval")
        .order_by(OfferApproval.submitted_at.desc())
    )
    offers = result.scalars().all()
    items = []
    for offer in offers:
        records = await _load_records(db, offer.id)
        idx = offer.current_approval_step
        if idx >= len(records):
            continue
        rec = records[idx]
        if rec.status != "pending" or not _can_approve_step(rec, current):
            continue
        cand = await _load_candidate(db, offer.candidate_id)
        pos = await _load_position(db, offer.position_id)
        items.append({
            "offerId": str(offer.id),
            "candidateName": cand.name if cand else None,
            "positionName": pos.name if pos else None,
            "department": offer.department or (pos.department if pos else None),
            "step": rec.step,
            "stepRoleName": rec.role_name,
            "submittedAt": _dt(offer.submitted_at),
        })
    return ok({"list": items, "total": len(items)})


@router.post("/offers/scan-expired")
async def scan_expired(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("system:manage")),
):
    from app.services.offer.scheduler import offer_expiry_scan
    count = await offer_expiry_scan(db)
    await db.commit()
    return ok({"expired": count})


# ═══════════════════════════════════════════════════════════
# 详情 / 日志 / PDF
# ═══════════════════════════════════════════════════════════

@router.get("/offers/{offer_id}")
async def get_offer(
    offer_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")

    item = serialize_offer(offer, current)
    await _enrich_offer(item, offer, db, current)

    records = await _load_records(db, offer.id)
    item["approvalRecords"] = [_serialize_record(r) for r in records]

    att_r = await db.execute(
        select(OfferAttachment).where(OfferAttachment.offer_id == offer.id).order_by(OfferAttachment.created_at)
    )
    item["attachments"] = [{
        "id": str(a.id), "category": a.category, "filename": a.filename,
        "contentType": a.content_type, "size": a.size,
        "createdAt": _dt(a.created_at),
    } for a in att_r.scalars().all()]

    if current.has("offer:approve") and offer.confirm_token and offer.status in ("sent", "approved"):
        item["confirmUrl"] = _confirm_url(request, offer.confirm_token)

    # 当前用户能否批当前步骤
    item["canApproveCurrentStep"] = False
    if offer.status == "pending_approval":
        if offer.current_approval_step < len(records):
            rec = records[offer.current_approval_step]
            if rec.status == "pending":
                item["canApproveCurrentStep"] = _can_approve_step(rec, current)

    item["logs"] = await _collect_logs(db, offer)
    return ok(item)


async def _collect_logs(db: AsyncSession, offer: OfferApproval) -> list[dict]:
    """操作日志：state_transitions + audit_logs + 关键时间戳合并，按时间倒序。"""
    from app.models.system import StateTransition

    logs: list[dict] = []
    st_r = await db.execute(
        select(StateTransition).where(
            StateTransition.entity_type == "offer", StateTransition.entity_id == offer.id
        ).order_by(StateTransition.created_at)
    )
    for st in st_r.scalars().all():
        logs.append({
            "time": _dt(st.created_at),
            "actor": st.actor_name,
            "content": f"状态变更: {st.from_status or '新建'} → {st.to_status}"
                       + (f"（{st.reason}）" if st.reason else ""),
        })

    # 关键时间戳
    ts = [
        ("提交审批", offer.submitted_at, offer.submitted_by),
        ("审批通过", offer.approved_at, offer.approver_id),
        ("发送Offer", offer.sent_at, offer.sent_by),
        ("候选人查看", offer.viewed_at, None),
        ("作废", offer.voided_at, offer.voided_by),
        ("自动失效", offer.expired_at, None),
    ]
    for label, t, actor_id in ts:
        if t:
            logs.append({"time": _dt(t), "actor": "系统" if actor_id is None else "相关人", "content": label})

    logs.sort(key=lambda x: x["time"] or "", reverse=True)
    return logs


@router.get("/offers/{offer_id}/export-pdf")
async def export_offer_pdf(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    """Offer 正式 PDF 导出：模板填充（PyMuPDF 在 Offer 通用模板上写入字段值）。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    try:
        pdf = await _render_offer_pdf(offer, db)
    except Exception:
        logger.exception("Offer PDF 生成失败 offer=%s", offer_id)
        return fail(500, "Offer PDF 生成失败，请检查模板文件与中文字体配置")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="offer_{offer.id}.pdf"'},
    )


@router.get("/offers/{offer_id}/preview")
async def preview_offer_pdf(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    """Offer PDF 预览（inline，供前端内嵌展示）。"""
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    try:
        pdf = await _render_offer_pdf(offer, db)
    except Exception:
        logger.exception("Offer PDF 预览生成失败 offer=%s", offer_id)
        return fail(500, "Offer PDF 生成失败，请检查模板文件与中文字体配置")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="offer_{offer.id}.pdf"'},
    )


# ═══════════════════════════════════════════════════════════
# 附件
# ═══════════════════════════════════════════════════════════

_ALLOWED_CATEGORIES = {"salary_confirmation", "supplemental_agreement", "other"}


@router.get("/offers/{offer_id}/attachments")
async def list_attachments(
    offer_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    att_r = await db.execute(
        select(OfferAttachment).where(OfferAttachment.offer_id == offer.id).order_by(OfferAttachment.created_at)
    )
    items = [{
        "id": str(a.id), "category": a.category, "filename": a.filename,
        "contentType": a.content_type, "size": a.size, "createdAt": _dt(a.created_at),
    } for a in att_r.scalars().all()]
    return ok({"list": items, "total": len(items)})


@router.post("/offers/{offer_id}/attachments")
async def upload_attachment(
    offer_id: str,
    category: str = Form("other"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    from app.infrastructure import minio_storage
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status not in ("draft", "pending_approval"):
        return fail(409, "仅草稿/待审批状态的 Offer 可上传附件")
    if category not in _ALLOWED_CATEGORIES:
        return fail(400, "附件类型不合法")

    content = await file.read()
    if not content:
        return fail(400, "文件为空")
    ext = ""
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    object_key = f"offers/{uuid.uuid4()}{ext}"
    try:
        await __import__("asyncio").to_thread(
            minio_storage.upload_bytes, object_key, content,
            file.content_type or "application/octet-stream",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Offer 附件上传失败: %s", e)
        return fail(500, "文件上传失败")

    att = OfferAttachment(
        offer_id=offer.id, category=category,
        filename=file.filename or "attachment",
        object_key=object_key, content_type=file.content_type,
        size=len(content), uploaded_by=current.id,
    )
    db.add(att)
    await db.flush()
    return ok({"id": str(att.id), "filename": att.filename, "category": att.category})


@router.get("/offers/{offer_id}/attachments/{att_id}")
async def download_attachment(
    offer_id: str,
    att_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    from app.infrastructure import minio_storage
    try:
        offer_uid = uuid.UUID(offer_id)
        att_uid = uuid.UUID(att_id)
    except (ValueError, TypeError):
        return not_found("附件不存在")
    att_r = await db.execute(select(OfferAttachment).where(
        OfferAttachment.id == att_uid, OfferAttachment.offer_id == offer_uid))
    att = att_r.scalar_one_or_none()
    if not att:
        return not_found("附件不存在")

    def _iter():
        resp = minio_storage.get_object_stream(att.object_key)
        try:
            for chunk in resp.stream(amt=64 * 1024):
                yield chunk
        finally:
            resp.close()
            resp.release_conn()

    return StreamingResponse(
        _iter(),
        media_type=att.content_type or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{att.filename}"'},
    )


@router.delete("/offers/{offer_id}/attachments/{att_id}")
async def delete_attachment(
    offer_id: str,
    att_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    from app.infrastructure import minio_storage
    offer = await _load_offer(db, offer_id)
    if not offer:
        return not_found("Offer 不存在")
    if offer.status not in ("draft", "pending_approval"):
        return fail(409, "仅草稿/待审批状态的 Offer 可删除附件")
    try:
        att_uid = uuid.UUID(att_id)
    except (ValueError, TypeError):
        return not_found("附件不存在")
    att_r = await db.execute(select(OfferAttachment).where(
        OfferAttachment.id == att_uid, OfferAttachment.offer_id == offer.id))
    att = att_r.scalar_one_or_none()
    if not att:
        return not_found("附件不存在")
    minio_storage.delete_object(att.object_key)
    await db.delete(att)
    await db.flush()
    return ok({"deleted": True})


# ═══════════════════════════════════════════════════════════
# Offer 模板
# ═══════════════════════════════════════════════════════════

def _serialize_template(t: OfferTemplate) -> dict:
    return {
        "id": str(t.id), "name": t.name, "code": t.code, "category": t.category,
        "content": t.content, "variables": t.variables or [], "isDefault": t.is_default,
        "enabled": t.enabled, "createdAt": _dt(t.created_at), "updatedAt": _dt(t.updated_at),
    }


class TemplateBody(BaseModel):
    model_config = {"populate_by_name": True}
    name: str
    code: Optional[str] = None
    category: Optional[str] = "offer_letter"
    content: Optional[str] = ""
    variables: Optional[list[str]] = None
    is_default: Optional[bool] = False
    enabled: Optional[bool] = True


@router.get("/offer-templates")
async def list_templates(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    r = await db.execute(select(OfferTemplate).order_by(OfferTemplate.created_at.desc()))
    items = [_serialize_template(t) for t in r.scalars().all()]
    return ok({"list": items, "total": len(items)})


@router.post("/offer-templates")
async def create_template(
    body: TemplateBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    code = body.code or body.name
    t = OfferTemplate(
        name=body.name, code=code, category=body.category,
        content=body.content or "", variables=body.variables or [],
        is_default=bool(body.is_default), enabled=bool(body.enabled),
        created_by=current.id,
    )
    db.add(t)
    await db.flush()
    return ok(_serialize_template(t))


@router.put("/offer-templates/{tpl_id}")
async def update_template(
    tpl_id: str,
    body: TemplateBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    try:
        uid = uuid.UUID(tpl_id)
    except (ValueError, TypeError):
        return not_found("模板不存在")
    r = await db.execute(select(OfferTemplate).where(OfferTemplate.id == uid))
    t = r.scalar_one_or_none()
    if not t:
        return not_found("模板不存在")
    t.name = body.name
    if body.code:
        t.code = body.code
    t.category = body.category
    t.content = body.content or ""
    if body.variables is not None:
        t.variables = body.variables
    t.is_default = bool(body.is_default)
    t.enabled = bool(body.enabled)
    await db.flush()
    return ok(_serialize_template(t))


@router.delete("/offer-templates/{tpl_id}")
async def delete_template(
    tpl_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("offer:approve")),
):
    try:
        uid = uuid.UUID(tpl_id)
    except (ValueError, TypeError):
        return not_found("模板不存在")
    r = await db.execute(select(OfferTemplate).where(OfferTemplate.id == uid))
    t = r.scalar_one_or_none()
    if not t:
        return not_found("模板不存在")
    ref = await db.execute(select(OfferApproval).where(OfferApproval.template_id == t.id).limit(1))
    if ref.scalar_one_or_none():
        return fail(409, "该模板已被 Offer 引用，不可删除")
    await db.delete(t)
    await db.flush()
    return ok({"deleted": True})


@router.get("/offer-templates/{tpl_id}/preview")
async def preview_template(
    tpl_id: str,
    offerId: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    try:
        uid = uuid.UUID(tpl_id)
    except (ValueError, TypeError):
        return not_found("模板不存在")
    r = await db.execute(select(OfferTemplate).where(OfferTemplate.id == uid))
    t = r.scalar_one_or_none()
    if not t:
        return not_found("模板不存在")
    content = t.content or ""
    # 用指定 Offer 或示例数据渲染占位符
    if offerId:
        offer = await _load_offer(db, offerId)
        if offer:
            cand = await _load_candidate(db, offer.candidate_id)
            pos = await _load_position(db, offer.position_id)
            content = _render_template(content, offer, cand, pos)
    return ok({"content": content})


def _render_template(content: str, offer: Optional[OfferApproval] = None,
                     candidate: Optional[Candidate] = None, position: Optional[Position] = None) -> str:
    comp = (offer.compensation if offer else {}) or {}
    values = {
        "name": candidate.name if candidate else "",
        "position": position.name if position else "",
        "department": (offer.department if offer else None) or (position.department if position else ""),
        "baseSalary": comp.get("base_salary", ""),
        "performanceSalary": comp.get("performance_salary", ""),
        "allowance": comp.get("allowance", ""),
        "onboardDate": offer.expected_onboard_date.isoformat() if offer and offer.expected_onboard_date else "",
        "probationMonths": str(offer.probation_months) if offer and offer.probation_months else "",
        "workLocation": offer.work_location if offer else "",
        "validityDays": str(offer.validity_days) if offer and offer.validity_days else "",
        "contractType": "固定期限" if (offer and offer.employment_terms and offer.employment_terms.get("contract_type") == "fixed") else "无固定期限",
        "workMode": "全职" if (offer and offer.employment_terms and offer.employment_terms.get("work_mode") == "fulltime") else "外包",
    }
    for k, v in values.items():
        content = content.replace("{{" + k + "}}", str(v))
    return content


# ═══════════════════════════════════════════════════════════
# 审批流配置
# ═══════════════════════════════════════════════════════════

def _serialize_flow(f: OfferApprovalFlow) -> dict:
    return {
        "id": str(f.id), "name": f.name, "code": f.code, "isDefault": f.is_default,
        "enabled": f.enabled, "description": f.description, "steps": f.steps or [],
        "createdAt": _dt(f.created_at), "updatedAt": _dt(f.updated_at),
    }


class FlowBody(BaseModel):
    model_config = {"populate_by_name": True}
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    is_default: Optional[bool] = False
    enabled: Optional[bool] = True
    steps: list[dict]


@router.get("/offer-flows")
async def list_flows(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_offer_read),
):
    r = await db.execute(select(OfferApprovalFlow).order_by(OfferApprovalFlow.created_at.desc()))
    items = [_serialize_flow(f) for f in r.scalars().all()]
    return ok({"list": items, "total": len(items)})


@router.post("/offer-flows")
async def create_flow(
    body: FlowBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("system:manage")),
):
    f = OfferApprovalFlow(
        name=body.name, code=body.code or body.name,
        description=body.description, is_default=bool(body.is_default),
        enabled=bool(body.enabled), steps=body.steps or [],
    )
    db.add(f)
    await db.flush()
    return ok(_serialize_flow(f))


@router.put("/offer-flows/{flow_id}")
async def update_flow(
    flow_id: str,
    body: FlowBody,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("system:manage")),
):
    try:
        uid = uuid.UUID(flow_id)
    except (ValueError, TypeError):
        return not_found("审批流不存在")
    r = await db.execute(select(OfferApprovalFlow).where(OfferApprovalFlow.id == uid))
    f = r.scalar_one_or_none()
    if not f:
        return not_found("审批流不存在")
    f.name = body.name
    if body.code:
        f.code = body.code
    f.description = body.description
    f.is_default = bool(body.is_default)
    f.enabled = bool(body.enabled)
    f.steps = body.steps or []
    await db.flush()
    return ok(_serialize_flow(f))


@router.delete("/offer-flows/{flow_id}")
async def delete_flow(
    flow_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(require_permission("system:manage")),
):
    try:
        uid = uuid.UUID(flow_id)
    except (ValueError, TypeError):
        return not_found("审批流不存在")
    r = await db.execute(select(OfferApprovalFlow).where(OfferApprovalFlow.id == uid))
    f = r.scalar_one_or_none()
    if not f:
        return not_found("审批流不存在")
    if f.code == "standard":
        return fail(409, "标准审批流不可删除")
    ref = await db.execute(select(OfferApproval).where(OfferApproval.approval_flow_id == f.id).limit(1))
    if ref.scalar_one_or_none():
        return fail(409, "该审批流已被 Offer 引用，不可删除")
    await db.delete(f)
    await db.flush()
    return ok({"deleted": True})
