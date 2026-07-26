"""统一操作审计写入 —— 携带真实操作人(满足人工确认合规要求 §21.6)。

用法:
    from app.utils.audit import write_audit
    await write_audit(db, actor="hr01", action="录用审批-批准", section="offer")
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.settings import AuditLog
from app.utils.clock import iso_utc


async def write_audit(
    db: AsyncSession,
    *,
    actor: str = "系统",
    action: str,
    section: str | None = None,
    flush: bool = True,
) -> None:
    # time 历史上是 "%Y-%m-%d %H:%M" 自定义格式，不是标准 ISO，
    # 前端 new Date() 解析在部分浏览器（如 Safari）下会得到 Invalid Date。
    # 改用 iso_utc() 统一为带 Z 的 ISO 字符串。
    db.add(AuditLog(
        id=uuid.uuid4().hex,
        time=iso_utc(datetime.utcnow()),
        actor=actor or "系统",
        action=action,
        section=section,
    ))
    if flush:
        await db.flush()
