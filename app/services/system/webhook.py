"""Webhook 推送 — 候选人状态变更等事件推送到外部 URL。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.system.system_settings import get_system_settings

logger = logging.getLogger("genie.webhook")


async def dispatch_webhook(
    db: AsyncSession,
    event: str,
    payload: dict[str, Any],
) -> bool:
    """
    读 webhookEnabled / webhookUrl。
    关闭或无 URL 则跳过；开启则 POST JSON，失败只记 warning 不抛。
    """
    settings = await get_system_settings(db)
    if not settings.get("webhookEnabled"):
        return False

    url = (settings.get("webhookUrl") or "").strip()
    if not url:
        logger.warning("Webhook 已启用但 webhookUrl 为空，跳过 event=%s", event)
        return False

    body = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code >= 400:
                logger.warning(
                    "Webhook 推送失败 event=%s status=%s body=%s",
                    event,
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        logger.info("Webhook 已推送 event=%s url=%s", event, url)
        return True
    except Exception as e:
        logger.warning("Webhook 推送异常 event=%s: %s", event, e)
        return False
