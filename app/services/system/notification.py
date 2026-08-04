"""通知服务 — log_only 策略：仅写日志 + 审计，不真正发送邮件/短信。"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.settings import AuditLog
from app.services.system.system_settings import get_system_setting

logger = logging.getLogger("genie.notification")


async def notify(
    db: AsyncSession,
    event: str,
    payload: dict[str, Any],
    setting_key: str,
) -> bool:
    """
    按设置开关决定是否记录通知。
    关闭则直接 return False；开启则 logger.info + 写入 AuditLog。
    """
    enabled = await get_system_setting(db, setting_key, True)
    if not enabled:
        logger.debug("通知已关闭 [%s] event=%s", setting_key, event)
        return False

    summary = payload.get("summary") or str(payload)[:200]
    logger.info("[通知] event=%s key=%s %s", event, setting_key, summary)

    log = AuditLog(
        id=f"log-{uuid.uuid4().hex[:12]}",
        time=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        actor="系统通知",
        action=f"[{event}] {summary}"[:255],
        section=None,
    )
    db.add(log)
    await db.flush()
    return True


async def notify_if(
    db: AsyncSession,
    setting_key: str,
    event: str,
    summary: str,
    extra: Optional[dict] = None,
) -> bool:
    payload = {"summary": summary, **(extra or {})}
    return await notify(db, event, payload, setting_key)


async def check_interview_reminders(db: AsyncSession) -> int:
    """对处于待面试状态的候选人发出提醒（log_only）。返回提醒条数。"""
    from sqlalchemy import select
    from app.models.recruitment import Candidate

    enabled = await get_system_setting(db, "notifyInterviewReminder", True)
    if not enabled:
        return 0

    result = await db.execute(
        select(Candidate).where(
            Candidate.status.in_(["round1", "round2"])
        )
    )
    candidates = list(result.scalars().all())
    count = 0
    for c in candidates:
        ok = await notify_if(
            db,
            "notifyInterviewReminder",
            "interview_reminder",
            f"面试前提醒：{c.name}（状态：{c.status}）",
            {"candidateId": str(c.id)},
        )
        if ok:
            count += 1
    return count
