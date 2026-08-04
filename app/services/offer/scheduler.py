"""Offer 定时任务 —— 过期扫描 / 待审批提醒 / 快到期提醒。

均走 notification.notify_if log_only 策略；过期扫描是唯一真正改状态的。
在 app/app.py 以 APScheduler job 注册，也可经 POST /offers/scan-expired 手动触发。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.phase1 import OfferApproval
from app.models.recruitment import Candidate
from app.core.state_machine import transition, StateError
from app.services.system.notification import notify_if

logger = logging.getLogger("genie.offer.scheduler")


async def offer_expiry_scan(db: AsyncSession) -> int:
    """扫描超期未处理的 Offer → expired；候选人视为放弃(talent_pool)。

    只扫 status in (approved, sent) 且 expires_at < now —— 天然幂等，
    不会与并发 accept/void 竞争把已决策单再迁失效。
    返回处理条数。
    """
    now = datetime.utcnow()
    result = await db.execute(
        select(OfferApproval).where(
            OfferApproval.status.in_(["approved", "sent"]),
            OfferApproval.expires_at.isnot(None),
            OfferApproval.expires_at < now,
        )
    )
    offers = list(result.scalars().all())
    if not offers:
        return 0

    count = 0
    for offer in offers:
        # 加载候选人
        cand = None
        cand_r = await db.execute(select(Candidate).where(Candidate.id == offer.candidate_id))
        cand = cand_r.scalar_one_or_none()

        try:
            was_sent = offer.status == "sent"
            await transition(db, "offer", offer, "expired",
                             actor_id=None, actor_name="系统",
                             reason="Offer 超过有效期，自动失效（视为放弃）", skip_block_check=True)
            offer.expired_at = now
            if was_sent and cand is not None:
                await transition(db, "candidate", cand, "talent_pool",
                                 actor_id=None, actor_name="系统",
                                 reason="Offer 超过有效期未接受，视为放弃，归入人才池", skip_block_check=True)
            await db.flush()
            count += 1
            await notify_if(db, "notifyOfferPending", "offer_expired",
                            f"Offer 已超期自动失效：{cand.name if cand else '候选人'} → 已视为放弃，可重新发起",
                            {"offerId": str(offer.id)})
        except StateError as e:
            logger.warning("Offer 过期扫描跳过 %s: %s", offer.id, e.message)
            await db.rollback()
            continue
    return count


async def check_offer_pending_approval_reminders(db: AsyncSession) -> int:
    """待审批 Offer 提醒（log_only）：提醒各级待办审批人。"""
    result = await db.execute(
        select(OfferApproval).where(OfferApproval.status == "pending_approval")
    )
    offers = list(result.scalars().all())
    count = 0
    for offer in offers:
        ok = await notify_if(db, "notifyOfferPending", "offer_pending_approval",
                             f"待审批 Offer：{offer.department or ''} 候选人 Offer 等待审批",
                             {"offerId": str(offer.id)})
        if ok:
            count += 1
    return count


async def check_offer_expiring_reminders(db: AsyncSession) -> int:
    """已发送 Offer 临近到期提醒 HR（log_only），阈值 24 小时。"""
    soon = datetime.utcnow() + timedelta(hours=24)
    result = await db.execute(
        select(OfferApproval).where(
            OfferApproval.status == "sent",
            OfferApproval.expires_at.isnot(None),
            OfferApproval.expires_at <= soon,
        )
    )
    offers = list(result.scalars().all())
    count = 0
    for offer in offers:
        ok = await notify_if(db, "notifyOfferPending", "offer_expiring",
                             f"Offer 即将到期：候选人需在 {offer.expires_at} 前确认，否则视为放弃",
                             {"offerId": str(offer.id)})
        if ok:
            count += 1
    return count
