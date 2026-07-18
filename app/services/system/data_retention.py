"""数据保留清理 — 按 dataRetentionDays 删除过期候选人/面试/审计日志。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure import minio_storage
from app.models.recruitment import Candidate
from app.models.interview import InterviewEvaluation, InterviewQuestion, InterviewTranscript
from app.models.settings import AuditLog
from app.services.system.system_settings import get_system_setting

logger = logging.getLogger("genie.data_retention")


async def cleanup_expired(db: AsyncSession) -> dict[str, Any]:
    """清理超过保留期的候选人及相关面试数据、过期审计日志。"""
    days = int(await get_system_setting(db, "dataRetentionDays", 365) or 365)
    if days < 1:
        days = 365

    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_date = cutoff.date()

    deleted_candidates = 0
    deleted_files = 0
    deleted_audit = 0

    # 过期候选人（按 upload_time）
    result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.ai_analysis))
        .where(Candidate.upload_time.is_not(None))
        .where(Candidate.upload_time < cutoff_date)
    )
    candidates = list(result.scalars().all())

    for cand in candidates:
        # 清理面试相关
        for model in (InterviewEvaluation, InterviewQuestion, InterviewTranscript):
            await db.execute(delete(model).where(model.candidate_id == cand.id))

        if cand.resume_file:
            try:
                await __import__("asyncio").to_thread(
                    minio_storage.delete_object, cand.resume_file
                )
                deleted_files += 1
            except Exception as e:
                logger.warning("删除简历文件失败 %s: %s", cand.resume_file, e)

        await db.delete(cand)
        deleted_candidates += 1

    # 过期审计日志
    audit_result = await db.execute(
        select(AuditLog).where(AuditLog.created_at < cutoff)
    )
    for log in audit_result.scalars().all():
        await db.delete(log)
        deleted_audit += 1

    await db.flush()

    summary = {
        "retentionDays": days,
        "cutoff": cutoff.isoformat(),
        "deletedCandidates": deleted_candidates,
        "deletedFiles": deleted_files,
        "deletedAuditLogs": deleted_audit,
    }
    logger.info("数据清理完成: %s", summary)
    return summary
