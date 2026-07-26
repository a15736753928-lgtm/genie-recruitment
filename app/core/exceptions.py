"""异常暂停队列辅助函数 (§2.4)。

用法:
    from app.core.exceptions import push_exception, mark_exception_handled
    await push_exception(db, entity_type="candidate", entity_id=cid,
                         exception_type="low_confidence",
                         detail="AI 置信度 0.45 < 0.6",
                         severity="block")
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update

from app.utils.clock import iso_utc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system import ExceptionQueue


async def push_exception(
    db: AsyncSession,
    *,
    entity_type: str,
    entity_id: uuid.UUID,
    exception_type: str,
    detail: str | None = None,
    severity: str = "warn",          # warn / block
    flush: bool = True,
) -> ExceptionQueue:
    """向异常队列写入一条新记录。重复写入不去重(记录每次触发)。"""
    exc = ExceptionQueue(
        entity_type=entity_type,
        entity_id=entity_id,
        exception_type=exception_type,
        detail=detail,
        severity=severity,
    )
    db.add(exc)
    if flush:
        await db.flush()
    return exc


async def has_blocking_exception(
    db: AsyncSession,
    entity_type: str,
    entity_id: uuid.UUID,
) -> bool:
    """当前实体是否有 open+block 级异常。"""
    row = await db.execute(
        select(ExceptionQueue).where(
            ExceptionQueue.entity_type == entity_type,
            ExceptionQueue.entity_id == entity_id,
            ExceptionQueue.status == "open",
            ExceptionQueue.severity == "block",
        ).limit(1)
    )
    return row.scalar_one_or_none() is not None


def serialize_exception(exc: ExceptionQueue) -> dict:
    return {
        "id": str(exc.id),
        "entityType": exc.entity_type,
        "entityId": str(exc.entity_id),
        "exceptionType": exc.exception_type,
        "detail": exc.detail,
        "severity": exc.severity,
        "status": exc.status,
        "handlerId": str(exc.handler_id) if exc.handler_id else None,
        "handlerName": exc.handler_name,
        "resolution": exc.resolution,
        "handledAt": iso_utc(exc.handled_at),
        "createdAt": iso_utc(exc.created_at),
    }
