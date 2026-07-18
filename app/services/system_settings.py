"""统一系统设置服务 — 带 60s 内存缓存，供各业务模块读取。"""

from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.settings import SystemSetting

# 与 routers/settings.py 保持一致；此处为兜底默认值，避免循环导入
DEFAULT_SETTINGS: dict[str, Any] = {
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
    "webhookUrl": "",
}

_CACHE: dict[str, Any] = {"data": None, "expires_at": 0.0}
_CACHE_TTL = 60.0


def invalidate_cache() -> None:
    """PUT 设置后调用，立即失效缓存。"""
    _CACHE["data"] = None
    _CACHE["expires_at"] = 0.0


async def get_system_settings(db: AsyncSession) -> dict[str, Any]:
    """读取全局设置并与默认值合并，结果缓存 60 秒。"""
    now = time.monotonic()
    if _CACHE["data"] is not None and now < _CACHE["expires_at"]:
        return dict(_CACHE["data"])

    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()
    stored = setting.value if setting and isinstance(setting.value, dict) else {}
    merged = {**DEFAULT_SETTINGS, **stored}

    _CACHE["data"] = merged
    _CACHE["expires_at"] = now + _CACHE_TTL
    return dict(merged)


async def get_system_setting(db: AsyncSession, key: str, default: Any = None) -> Any:
    """读取单项设置。"""
    settings = await get_system_settings(db)
    if key in settings:
        return settings[key]
    return default
