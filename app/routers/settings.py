from datetime import datetime
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.settings import SystemSetting, AuditLog

router = APIRouter(tags=["系统设置"])

DEFAULT_SETTINGS = {
    "systemName": "Genie 智能招聘系统",
    "companyName": "Genie Tech",
    "contactEmail": "hr@genietech.com",
    "defaultQuarter": "2026-Q3",
    "autoParseResume": True,
    "minMatchScore": 70,
    "defaultPositionId": "",
    "offerApprovalRequired": True,
    "defaultQuestionCount": 8,
    "defaultScoringMode": "ai",
    "passScoreThreshold": 75,
    "allowAudioUpload": True,
    "probationDays": 90,
    "defaultProbationTasks": 5,
    "aiResumeAnalysis": True,
    "aiQuestionGeneration": True,
    "aiInterviewScoring": True,
    "recallThreshold": 0.75,
    "notifyNewResume": True,
    "notifyInterviewReminder": True,
    "notifyOfferPending": True,
    "notifyProbationRisk": True,
    "notifyPerformanceDue": True,
    "dataRetentionDays": 365,
    "exportFormat": "xlsx",
    "webhookEnabled": False,
}


@router.get("/settings")
async def get_settings(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()
    data = setting.value if setting and setting.value else DEFAULT_SETTINGS
    return {"code": 0, "message": "ok", "data": data}


@router.put("/settings")
async def update_settings(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()

    if setting:
        # Merge with existing
        merged = {**setting.value, **body}
        setting.value = merged
    else:
        merged = {**DEFAULT_SETTINGS, **body}
        setting = SystemSetting(
            key="global",
            value=merged,
        )
        db.add(setting)

    await db.flush()
    return {"code": 0, "message": "ok", "data": merged}


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
        time=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
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
