from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.settings import SystemSetting
from app.models.user import User, AuditLog
from app.routers.auth import get_current_user

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
    "autoTrainKnowledge": False,
    "autoTrainSchedule": "0 2 * * 0",
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
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()

    if setting:
        # Merge with existing
        merged = {**setting.value, **body}
        setting.value = merged
        setting.updated_by = current_user.id
    else:
        merged = {**DEFAULT_SETTINGS, **body}
        setting = SystemSetting(
            key="global",
            value=merged,
            updated_by=current_user.id,
        )
        db.add(setting)

    # Audit log
    log = AuditLog(
        actor_id=current_user.id,
        actor_name=current_user.display_name,
        action="更新系统设置",
        section="basic",
    )
    db.add(log)

    await db.flush()
    return {"code": 0, "message": "ok", "data": merged}


@router.get("/settings/audit-logs")
async def get_audit_logs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(100)
    )
    logs = result.scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(log.id),
                "time": log.created_at.strftime("%Y-%m-%d %H:%M") if log.created_at else "",
                "actor": log.actor_name or "系统",
                "action": log.action,
                "section": log.section,
            }
            for log in logs
        ],
    }
