"""简历 → RAG「简历」知识库自动入库。

上传简历时：若尚无名为「简历」的知识库则创建，再复用简历原件的 MinIO 对象触发
分片 / 向量化 / Milvus 入库（ingest_file_async 后台线程）。

注意：本文档的 object_key 与 Candidate.resume_file 指向同一个 MinIO 对象，
不再单独复制一份到 rag/ 前缀，避免两份文件各自失踪、不同步。
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeBase, _now_ms, _short_uuid
from app.services.rag.ingest_service import ingest_file_async
from app.services.rag.utils import SUPPORTED_EXTENSIONS, check_extension

logger = logging.getLogger("genie.resume_kb")

RESUME_KB_NAME = "简历"
RESUME_KB_DESCRIPTION = "候选人简历自动入库知识库（由简历上传自动维护）"


async def ensure_resume_knowledge_base(db: AsyncSession) -> KnowledgeBase:
    """查找或创建名为「简历」的知识库。"""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.name == RESUME_KB_NAME)
    )
    kb = result.scalar_one_or_none()
    if kb:
        return kb

    now = _now_ms()
    kb = KnowledgeBase(
        id=_short_uuid("kb"),
        name=RESUME_KB_NAME,
        description=RESUME_KB_DESCRIPTION,
        owner_id=None,
        doc_count=0,
        chunk_count=0,
        created_at=now,
        updated_at=now,
    )
    db.add(kb)
    await db.flush()
    logger.info("已自动创建知识库「%s」id=%s", RESUME_KB_NAME, kb.id)
    return kb


async def ingest_resume_to_kb(
    db: AsyncSession,
    *,
    content: bytes,
    file_name: str,
    source_object_key: str = "",
    candidate_id: str = "",
) -> Optional[Tuple[str, str]]:
    """
    确保「简历」知识库存在，复用简历原件的 MinIO 对象启动 ingest。

    Args:
        candidate_id: 关联的候选人 ID。会写入 KnowledgeDocument.source_id，
            删除该文档时据此级联删除候选人及其下游数据。

    Returns:
        (doc_id, task_id) 或 None（扩展名不支持 / 失败时）。
        失败不抛出，仅记日志，避免阻断简历上传主流程。
    """
    try:
        if not check_extension(file_name):
            ext = os.path.splitext(file_name)[1].lower()
            logger.warning(
                "简历文件格式不支持 RAG 入库，已跳过: %s（支持 %s）",
                ext or "(无扩展名)",
                ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
            return None

        kb = await ensure_resume_knowledge_base(db)

        # 复用简历原件的 MinIO 对象：Candidate.resume_file 与本文档的 object_key
        # 指向同一个对象，避免两份文件各自失踪、不同步。删除时由 cascade_delete
        # 统一清理；入库去重分支已对 source_type='resume' 跳过删除，不会误删原件。
        if not source_object_key:
            logger.warning(
                "简历入库缺少 source_object_key，跳过（candidate=%s file=%s）",
                candidate_id or "-", file_name,
            )
            return None
        rag_key = source_object_key

        doc_id, task_id = ingest_file_async(
            file_path=rag_key,
            file_name=file_name,
            kb_id=kb.id,
            file_size=len(content),
            source_type="resume" if candidate_id else "",
            source_id=str(candidate_id) if candidate_id else "",
        )
        logger.info(
            "简历已提交知识库入库 kb=%s doc=%s task=%s source=%s candidate=%s",
            kb.id,
            doc_id,
            task_id,
            source_object_key or "-",
            candidate_id or "-",
        )
        return doc_id, task_id
    except Exception as e:
        logger.warning("简历自动入库知识库失败（不影响简历上传）: %s", e)
        return None
