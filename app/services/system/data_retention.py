"""数据保留清理 — 按 dataRetentionDays 删除过期候选人/面试/审计日志。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recruitment import Candidate
from app.models.settings import AuditLog
from app.services.system.cascade_delete import cascade_delete_by_candidate
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
        select(Candidate.id, Candidate.resume_file)
        .where(Candidate.upload_time.is_not(None))
        .where(Candidate.upload_time < cutoff_date)
    )
    rows = result.all()

    for cid, resume_file in rows:
        # 走统一级联删除：面试题/评分/转写(FK cascade)、试用期/绩效、人才库、
        # 关联的「简历」知识库文档（含分片 + Milvus 向量 + 共享的 MinIO 原件）。
        # 简历原件与 KB 文档 object_key 是同一对象，cascade 内部统一清理，避免悬空引用。
        deleted = await cascade_delete_by_candidate(db, str(cid), delete_resume_file=True)
        if deleted:
            deleted_candidates += 1
            if resume_file:
                deleted_files += 1

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
