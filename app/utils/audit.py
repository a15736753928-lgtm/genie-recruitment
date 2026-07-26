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


async def write_audit(
    db: AsyncSession,
    *,
    actor: str = "系统",
    action: str,
    section: str | None = None,
    flush: bool = True,
) -> None:
    db.add(AuditLog(
        id=uuid.uuid4().hex,
        time=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        actor=actor or "系统",
        action=action,
        section=section,
    ))
    if flush:
        await db.flush()
