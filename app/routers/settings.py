from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.settings import SystemSetting

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
