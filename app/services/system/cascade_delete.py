"""统一级联删除服务。

无论是从「知识库管理 → 文档」删除一份简历文档，还是从「简历管理」删除一个候选人，
都意味着这个人不存在了。两边调用同一个函数，确保所有相关数据被一致清理：

  - 简历原件（MinIO / 本地）
  - 面试题 / 评分 / 录音转写（FK ON DELETE CASCADE 自动删除）
  - 试用期员工记录 + 试用期任务 / 周评估 / 转正评估
  - 绩效记录
  - 人才库记录
  - RAG「简历」知识库中对应的文档（含分片 + Milvus 向量 + MinIO 文件）
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from sqlalchemy import select, delete as _delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure import minio_storage
from app.models.recruitment import Candidate, TalentPool
from app.models.probation import Employee
from app.models.performance import PerformanceRecord
from app.models.knowledge import KnowledgeDocument, KnowledgeChunk

logger = logging.getLogger("genie.cascade_delete")

RESUME_KB_NAME = "简历"


async def cascade_delete_by_candidate(
    db: AsyncSession,
    candidate_id: str,
    *,
    delete_resume_file: bool = True,
) -> int:
    """按候选人 ID 级联删除其所有相关数据。

    Args:
        candidate_id: 候选人 UUID（字符串）
        delete_resume_file: 是否删除 MinIO 上的简历原件

    Returns:
        实际删除的候选人数量（0 表示候选人不存在）
    """
    cid = str(candidate_id)
    logger.info("级联删除候选人 candidate_id=%s", cid)

    cand_result = await db.execute(select(Candidate).where(Candidate.id == cid))
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        logger.warning("级联删除：候选人不存在 candidate_id=%s", cid)
        return 0

    # 1) 试用期员工 + 绩效记录
    await _delete_employee_and_performance(db, cid)

    # 2) 人才库
    await db.execute(_delete(TalentPool).where(TalentPool.candidate_id == cid))

    # 3) RAG「简历」知识库中对应的文档
    await _delete_linked_kb_documents(db, cid)

    # 4) 简历原件
    if delete_resume_file and candidate.resume_file:
        await _delete_resume_file(candidate.resume_file)

    # 5) 候选人本身（面试题/评分/转写通过 FK CASCADE 自动删除）
    await db.delete(candidate)
    await db.flush()
    logger.info("候选人已删除 candidate_id=%s name=%s", cid, candidate.name)
    return 1


async def cascade_delete_by_kb_document(
    db: AsyncSession,
    doc: KnowledgeDocument,
) -> None:
    """按知识库文档级联删除。

    若该文档属于「简历」知识库：
      - 优先用 source_id 找到候选人并级联删除（新数据）
      - source_id 缺失时按文件名回退匹配候选人（老数据）
    然后再删除文档本身（分片 + Milvus 向量 + MinIO 文件）。
    """
    candidate_id = await _resolve_candidate_id_from_doc(db, doc)

    if candidate_id:
        # 级联删除候选人（内部会删除其关联的 KB 文档，可能包括本 doc）
        await cascade_delete_by_candidate(db, candidate_id, delete_resume_file=True)

    # 若候选人不存在或未匹配到，仍要确保当前文档被删除
    # （cascade_delete_by_candidate 已删 source_id 关联的文档；这里兜底）
    still_exists = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.id == doc.id)
    )
    if still_exists.scalar_one_or_none() is None:
        return  # 已被候选人级联删除

    await _delete_single_kb_document(db, doc)


# ── 内部实现 ──────────────────────────────────────────────

async def _resolve_candidate_id_from_doc(
    db: AsyncSession,
    doc: KnowledgeDocument,
) -> Optional[str]:
    """从知识库文档反查候选人 ID。

    1) 优先用 source_type='resume' + source_id（新数据）
    2) 回退：文档属于「简历」KB 时，按文件名匹配 candidate.name
    """
    source_type = (getattr(doc, "source_type", "") or "").strip()
    source_id = (getattr(doc, "source_id", "") or "").strip()
    if source_type == "resume" and source_id:
        # 校验候选人确实存在
        exists = await db.execute(
            select(Candidate.id).where(Candidate.id == source_id)
        )
        if exists.scalar_one_or_none():
            return source_id
        logger.warning("文档 source_id 指向的候选人不存在 doc=%s source_id=%s", doc.id, source_id)

    # 回退：检查是否属于「简历」知识库
    from app.models.knowledge import KnowledgeBase
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id))
    kb = kb_result.scalar_one_or_none()
    if not kb or kb.name != RESUME_KB_NAME:
        return None

    # 按文件名匹配候选人（candidate.name 在上传时被设为原始文件名）
    cand_result = await db.execute(
        select(Candidate).where(Candidate.name == doc.file_name).limit(1)
    )
    candidate = cand_result.scalar_one_or_none()
    if candidate:
        logger.info(
            "按文件名回退匹配到候选人 doc=%s file=%s candidate_id=%s",
            doc.id, doc.file_name, candidate.id,
        )
        return str(candidate.id)

    logger.info("文档属于简历知识库但未匹配到候选人 doc=%s file=%s", doc.id, doc.file_name)
    return None


async def _delete_employee_and_performance(db: AsyncSession, candidate_id: str) -> None:
    emp_result = await db.execute(select(Employee).where(Employee.candidate_id == candidate_id))
    employees = emp_result.scalars().all()
    if not employees:
        return
    emp_ids = [str(e.id) for e in employees]
    await db.execute(
        _delete(PerformanceRecord).where(PerformanceRecord.employee_id.in_(emp_ids))
    )
    logger.info("已删除绩效记录 count=%d candidate=%s", len(emp_ids), candidate_id)
    for e in employees:
        await db.delete(e)
    logger.info("已删除试用期员工 count=%d candidate=%s", len(employees), candidate_id)


async def _delete_linked_kb_documents(db: AsyncSession, candidate_id: str) -> None:
    """删除该候选人关联的所有 RAG 文档（按 source_id 匹配）。"""
    doc_result = await db.execute(
        select(KnowledgeDocument).where(
            KnowledgeDocument.source_type == "resume",
            KnowledgeDocument.source_id == candidate_id,
        )
    )
    docs = doc_result.scalars().all()
    for doc in docs:
        await _delete_single_kb_document(db, doc)
        logger.info("已删除关联知识库文档 doc_id=%s candidate=%s", doc.id, candidate_id)


async def _delete_single_kb_document(db: AsyncSession, doc: KnowledgeDocument) -> None:
    """删除单个知识库文档：MinIO 文件 + PG 分片 + 文档 + Milvus 向量(后台)。"""
    # 收集 Milvus PK
    chunk_result = await db.execute(
        select(KnowledgeChunk.milvus_pk).where(KnowledgeChunk.doc_id == doc.id)
    )
    milvus_pks = [r[0] for r in chunk_result.fetchall() if r[0] > 0]

    # 更新父 KB 计数
    from app.models.knowledge import KnowledgeBase, _now_ms
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id))
    kb = kb_result.scalar_one_or_none()
    if kb:
        kb.doc_count = max(0, (kb.doc_count or 0) - 1)
        kb.chunk_count = max(0, (kb.chunk_count or 0) - (doc.chunk_count or 0))
        kb.updated_at = _now_ms()

    # 删 MinIO 文件
    if doc.object_key:
        try:
            await asyncio.to_thread(minio_storage.delete_object, doc.object_key)
        except Exception as e:
            logger.warning("删除 MinIO 文件失败（忽略）: %s key=%s", e, doc.object_key)

    # 删文档（PG cascade 删分片）
    await db.delete(doc)
    await db.flush()

    # Milvus 向量后台清理
    if milvus_pks:
        async def _cleanup(pks=milvus_pks):
            try:
                from app.infrastructure.milvus_manager import delete_by_ids
                await asyncio.to_thread(delete_by_ids, pks)
                logger.info("Milvus 向量已清理 count=%d", len(pks))
            except Exception as ex:
                logger.warning("Milvus 向量清理失败（可忽略）: %s", ex)
        asyncio.create_task(_cleanup())


async def _delete_resume_file(stored_path: str) -> None:
    if not stored_path:
        return
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        try:
            os.remove(stored_path)
        except Exception as e:
            logger.warning("删除本地简历文件失败（忽略）: %s path=%s", e, stored_path)
        return
    try:
        await asyncio.to_thread(minio_storage.delete_object, stored_path)
    except Exception as e:
        logger.warning("删除 MinIO 简历文件失败（忽略）: %s key=%s", e, stored_path)
