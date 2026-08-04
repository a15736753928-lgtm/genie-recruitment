"""候选人线上确认页 —— 公开路由（无鉴权，凭 confirm_token）。

GET  /public/offer/{token}            — 查看 Offer + 接受/拒绝表单（置 viewed_at）
POST /public/offer/{token}/accept     — 接受：offer→accepted, candidate→hired, 生成 Employee(pending_onboard)
POST /public/offer/{token}/decline    — 拒绝：offer→declined, candidate→talent_pool, 记录原因

安全：token 为 secrets.token_urlsafe(32)；POST 内 SELECT FOR UPDATE 锁行防并发双击双写；
已用（token_used_at）或已终态即渲染"已处理"页。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.phase1 import OfferApproval
from app.models.recruitment import Candidate, Position
from app.models.probation import Employee
from app.core.state_machine import transition
from app.services.offer import public_page
from app.services.system.system_settings import get_system_setting
from app.services.talent.employee_builder import build_employee_from_candidate

logger = logging.getLogger("genie.offer.public")
router = APIRouter(tags=["Offer公开确认"])


def _fmt_dt(v) -> str:
    if not v:
        return "—"
    return v.strftime("%Y-%m-%d %H:%M")


async def _load_by_token(db: AsyncSession, token: str, lock: bool = False):
    q = select(OfferApproval).where(OfferApproval.confirm_token == token)
    if lock:
        q = q.with_for_update()
    r = await db.execute(q)
    return r.scalar_one_or_none()


async def _build_ctx(db: AsyncSession, offer: OfferApproval) -> dict:
    cand = None
    cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
    cand = cand_r.scalar_one_or_none()
    pos = None
    if offer.position_id:
        pos_r = await db.execute(select(Position).where(Position.id == offer.position_id))
        pos = pos_r.scalar_one_or_none()
    company = str(await get_system_setting(db, "companyName", "公司") or "")
    return {
        "company": company,
        "candidate_name": cand.name if cand else "",
        "candidate_phone": cand.phone if cand else "",
        "candidate_email": cand.email if cand else "",
        "position_name": pos.name if pos else "",
        "department": offer.department or (pos.department if pos else ""),
        "work_location": offer.work_location or "",
        "expected_onboard_date": offer.expected_onboard_date,
        "probation_months": offer.probation_months,
        "token_expires_at": _fmt_dt(offer.token_expires_at),
        "compensation": offer.compensation or {},
        "employment_terms": offer.employment_terms or {},
        "other_terms": offer.other_terms or {},
        "accept_url": f"/public/offer/{offer.confirm_token}/accept",
        "decline_url": f"/public/offer/{offer.confirm_token}/decline",
        "confirm_url": f"/public/offer/{offer.confirm_token}",
    }


async def _result_html(db, offer, state: str, message: str = "") -> HTMLResponse:
    ctx = await _build_ctx(db, offer)
    return HTMLResponse(public_page.render_result_page(state, ctx, message))


@router.get("/public/offer/{token}", response_class=HTMLResponse)
async def public_view_offer(token: str, db: AsyncSession = Depends(get_db)):
    offer = await _load_by_token(db, token)
    if not offer:
        return HTMLResponse(public_page.render_result_page("not_found", {}))

    # 已用/终态 → 结果页
    if offer.token_used_at or offer.status in ("accepted", "declined", "expired", "voided"):
        state = "already_processed"
        if offer.status == "accepted":
            state = "accepted"
        elif offer.status == "declined":
            state = "declined"
        elif offer.status == "expired":
            state = "expired"
        elif offer.status == "voided":
            state = "voided"
        return await _result_html(db, offer, state)

    # 超期（offer 尚未被扫描到 expired 时按 token 有效期判定）
    if offer.token_expires_at and offer.token_expires_at < datetime.utcnow():
        return await _result_html(db, offer, "expired")

    # 置查看时间（首访）
    if offer.status == "sent" and not offer.viewed_at:
        offer.viewed_at = datetime.utcnow()
        from app.utils.audit import write_audit
        await write_audit(db, actor="候选人", action="查看Offer链接", section="offer")
        await db.commit()

    ctx = await _build_ctx(db, offer)
    return HTMLResponse(public_page.render_confirm_page(ctx))


@router.post("/public/offer/{token}/accept", response_class=HTMLResponse)
async def public_accept_offer(token: str, db: AsyncSession = Depends(get_db)):
    offer = await _load_by_token(db, token, lock=True)
    if not offer:
        return HTMLResponse(public_page.render_result_page("not_found", {}))
    if offer.token_used_at or offer.status != "sent":
        return await _result_html(db, offer, "already_processed")
    if offer.token_expires_at and offer.token_expires_at < datetime.utcnow():
        return await _result_html(db, offer, "expired")

    cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
    candidate = cand_r.scalar_one_or_none()
    if not candidate:
        return await _result_html(db, offer, "not_found", "候选人不存在")

    try:
        await transition(db, "offer", offer, "accepted",
                         actor_id=None, actor_name="候选人", reason="候选人在线接受Offer")
        await transition(db, "candidate", candidate, "hired",
                         actor_id=None, actor_name="候选人", reason="候选人接受Offer，入职")
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        logger.warning("Offer 在线接受失败 %s: %s", token, e)
        return await _result_html(db, offer, "already_processed", "处理失败，请联系 HR")

    # 生成待入职员工档案（统一构建器：身份信息从候选人主档拷贝，department 取 Offer 快照）
    onboard = offer.expected_onboard_date or datetime.utcnow().date()
    employee = build_employee_from_candidate(
        candidate,
        position_id=offer.position_id,
        department=offer.department,
        onboard_date=onboard,
        probation_months=offer.probation_months,
    )
    db.add(employee)
    await db.flush()
    offer.employee_id = employee.id
    offer.token_used_at = datetime.utcnow()
    await db.flush()
    await db.commit()
    return await _result_html(db, offer, "accepted")


@router.post("/public/offer/{token}/decline", response_class=HTMLResponse)
async def public_decline_offer(
    token: str,
    reason: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    offer = await _load_by_token(db, token, lock=True)
    if not offer:
        return HTMLResponse(public_page.render_result_page("not_found", {}))
    if offer.token_used_at or offer.status != "sent":
        return await _result_html(db, offer, "already_processed")
    if offer.token_expires_at and offer.token_expires_at < datetime.utcnow():
        return await _result_html(db, offer, "expired")

    cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
    candidate = cand_r.scalar_one_or_none()
    if not candidate:
        return await _result_html(db, offer, "not_found", "候选人不存在")

    try:
        await transition(db, "offer", offer, "declined",
                         actor_id=None, actor_name="候选人", reason=f"候选人在线拒绝Offer: {reason or ''}")
        await transition(db, "candidate", candidate, "talent_pool",
                         actor_id=None, actor_name="候选人", reason=f"候选人拒绝Offer: {reason or ''}")
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        logger.warning("Offer 在线拒绝失败 %s: %s", token, e)
        return await _result_html(db, offer, "already_processed", "处理失败，请联系 HR")

    offer.decline_reason = reason
    offer.token_used_at = datetime.utcnow()
    await db.flush()
    await db.commit()
    return await _result_html(db, offer, "declined")
