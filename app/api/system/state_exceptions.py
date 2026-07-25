"""异常队列 + 状态流转日志查询接口 (§2.3.5 / §2.4.3)。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.system import ExceptionQueue, StateTransition
from app.core.security import get_current_user, CurrentUser
from app.core.exceptions import serialize_exception
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
import uuid as _uuid
from datetime import datetime

router = APIRouter(tags=["异常队列&状态流转"])


# ── 异常队列 ─────────────────────────────────────────────────

@router.get("/exceptions")
async def list_exceptions(
    status: str | None = Query(None),
    entity_type: str | None = Query(None, alias="entityType"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(ExceptionQueue)
    if status:
        q = q.where(ExceptionQueue.status == status)
    if entity_type:
        q = q.where(ExceptionQueue.entity_type == entity_type)
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar() or 0
    rows = (await db.execute(
        q.order_by(ExceptionQueue.created_at.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()
    return ok({
        "list": [serialize_exception(e) for e in rows],
        "total": total, "page": page, "pageSize": page_size,
    })


class HandleExceptionRequest(BaseModel):
    resolution: str


@router.post("/exceptions/{exc_id}/handle")
async def handle_exception(
    exc_id: str,
    body: HandleExceptionRequest,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.execute(select(ExceptionQueue).where(
        ExceptionQueue.id == _uuid.UUID(exc_id)
    ))
    exc = row.scalar_one_or_none()
    if exc is None:
        return not_found("异常记录不存在")
    if exc.status == "handled":
        return fail(409, "该异常已处理")
    exc.status = "handled"
    exc.handler_id = current.id
    exc.handler_name = current.display_name
    exc.resolution = body.resolution
    exc.handled_at = datetime.utcnow()
    await write_audit(
        db, actor=current.username,
        action=f"处理异常 [{exc.exception_type}]",
        section="exceptions",
    )
    return ok(serialize_exception(exc))


# ── 状态流转日志 ─────────────────────────────────────────────

def _serialize_transition(t: StateTransition) -> dict:
    return {
        "id": str(t.id),
        "entityType": t.entity_type,
        "entityId": str(t.entity_id),
        "fromStatus": t.from_status,
        "toStatus": t.to_status,
        "reason": t.reason,
        "evidence": t.evidence,
        "actorId": str(t.actor_id) if t.actor_id else None,
        "actorName": t.actor_name,
        "createdAt": t.created_at.isoformat() if t.created_at else None,
    }


@router.get("/{entity_type}/{entity_id}/transitions")
async def list_transitions(
    entity_type: str,
    entity_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = _uuid.UUID(entity_id)
    except ValueError:
        return not_found("ID 格式无效")
    rows = (await db.execute(
        select(StateTransition).where(
            StateTransition.entity_type == entity_type,
            StateTransition.entity_id == eid,
        ).order_by(StateTransition.created_at.desc())
    )).scalars().all()
    return ok([_serialize_transition(t) for t in rows])
