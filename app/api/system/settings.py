from datetime import datetime
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.settings import SystemSetting, AuditLog
from app.services.system.system_settings import (
    DEFAULT_SETTINGS,
    get_system_settings,
    invalidate_cache,
)

router = APIRouter(tags=["系统设置"])


@router.get("/settings")
async def get_settings(db: AsyncSession = Depends(get_db)):
    data = await get_system_settings(db)
    return {"code": 0, "message": "ok", "data": data}


@router.put("/settings")
async def update_settings(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()

    if setting:
        merged = {**(setting.value or {}), **body}
        setting.value = merged
    else:
        merged = {**DEFAULT_SETTINGS, **body}
        setting = SystemSetting(
            key="global",
            value=merged,
        )
        db.add(setting)

    await db.flush()
    invalidate_cache()
    return {"code": 0, "message": "ok", "data": merged}


@router.post("/settings/cleanup")
async def trigger_cleanup(db: AsyncSession = Depends(get_db)):
    """手动触发过期数据清理。"""
    from app.services.system.data_retention import cleanup_expired

    result = await cleanup_expired(db)
    return {"code": 0, "message": "ok", "data": result}


# ── Audit log ──────────────────────────────────────────

@router.get("/settings/audit-log")
async def list_audit_log(db: AsyncSession = Depends(get_db)):
    """获取系统设置操作审计日志（最近 50 条，按时间倒序）。"""
    result = await db.execute(
        select(AuditLog).order_by(desc(AuditLog.created_at)).limit(50)
    )
    logs = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": log.id,
                "time": log.time,
                "actor": log.actor,
                "action": log.action,
                "section": log.section or "",
            }
            for log in logs
        ],
    }


@router.post("/settings/audit-log")
async def append_audit_log(body: dict, db: AsyncSession = Depends(get_db)):
    """追加一条审计日志。body: { actor, action, section }"""
    log = AuditLog(
        id=f"log-{uuid.uuid4().hex[:12]}",
        time=body.get("time") or datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        actor=body.get("actor", "系统"),
        action=body.get("action", ""),
        section=body.get("section"),
    )
    db.add(log)
    await db.flush()
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": log.id,
            "time": log.time,
            "actor": log.actor,
            "action": log.action,
            "section": log.section or "",
        },
    }
