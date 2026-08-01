"""Document ingestion pipeline — 3-layer parsing with OCR fallback.

Blueprint: file → parse (fast → render → OCR) → clean → chunk → embed → store.

Matches the reference RAG system's 3-layer approach:
  1. Fast text extraction (PyMuPDF / python-docx)
  2. OCR fallback for images and low-text pages (RapidOCR)
  3. Merge results, preferring OCR when it produces more text

Runs in a background thread so the API returns immediately with a task_id.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.database import get_sync_db
from app.models.knowledge import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    IngestionTask,
    _now_ms,
    _short_uuid,
)
from app.services.rag.text_processor import clean_text
from app.services.rag.splitter import chunk_document
from app.services.rag.embedding import encode_batch
from app.services.rag.utils import get_file_type
from app.infrastructure.milvus_manager import insert_vectors

settings = get_settings()
logger = logging.getLogger(__name__)

# Per-thread ingest overrides (loaded from system_settings in worker)
_ingest_runtime = threading.local()


# ── File Parsing (multimodal, no local OCR) ─────────────────

def _extract_text(file_path: str) -> str:
    """Parse file to text via unified multimodal vision (no local OCR).

    PDF → render pages → MIMO vision. Images → MIMO vision directly.
    DOCX / plain text → native text read (not image-based; avoids token waste).
    """
    ext = Path(file_path).suffix.lower()

    # ── Plain text / DOCX: native read ──
    if ext in (".txt", ".md", ".markdown"):
        return _read_text_file(file_path)
    if ext == ".docx":
        return _extract_docx_text(file_path)

    # ── Images: direct multimodal vision ──
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        with open(file_path, "rb") as f:
            from app.services.ai.vision import extract_text_from_image_bytes_sync
            return extract_text_from_image_bytes_sync(f.read())

    # ── PDF: render to pages → multimodal vision ──
    if ext == ".pdf":
        from app.services.ai.vision import render_pdf_pages_to_png, extract_text_from_images_sync
        pages = render_pdf_pages_to_png(file_path)
        if not pages:
            logger.warning("PDF 渲染失败，无法多模态识别: %s", os.path.basename(file_path))
            return ""
        return extract_text_from_images_sync(pages)

    return ""


def _read_text_file(path: str) -> str:
    """Read text file with encoding detection."""
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def _extract_docx_text(path: str) -> str:
    """Extract text from DOCX using python-docx (native text, not image OCR)."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.debug("python-docx 解析失败: %s", e)
        return ""


# ── Background Ingestion ────────────────────────────────────

def _run_ingest_sync(
    doc_id: str,
    task_id: str,
    object_key: str,
    file_name: str,
    kb_id: str,
    file_size: int = 0,
    source_type: str = "",
    source_id: str = "",
):
    """Full ingestion pipeline (runs in background thread).

    ``object_key`` is a MinIO object key — the file is downloaded to a local
    temp file for parsing/extraction and removed when done. Uses a sync DB
    session to avoid event-loop conflicts with async FastAPI.
    """
    now = _now_ms()
    db = get_sync_db()
    doc = None
    task = None
    local_path = ""

    try:
        from app.services.system.kb_settings import get_kb_ingest_settings_sync

        ingest_cfg = get_kb_ingest_settings_sync(db)
        _ingest_runtime.ocr_enabled = ingest_cfg["ocr_enabled"]

        # Download the original file from MinIO to a temp file for parsing.
        ext = os.path.splitext(file_name)[1] or ".txt"
        from app.infrastructure import minio_storage
        try:
            local_path = minio_storage.download_to_temp(object_key, suffix=ext)
        except Exception as e:
            logger.error("从 MinIO 下载文件失败: %s (key=%s)", e, object_key)
            raise

        # Compute file hash for dedup
        file_hash = ""
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

        # 1. Create document + task records
        existing = (
            db.query(KnowledgeDocument.id)
            .filter(
                KnowledgeDocument.kb_id == kb_id,
                KnowledgeDocument.file_name == file_name,
            )
            .first()
        )
        if existing:
            logger.warning(
                "跳过重复文档入库 kb=%s file=%s existing=%s",
                kb_id, file_name, existing[0],
            )
            # 简历入库时 object_key 与 Candidate.resume_file 是同一个共享对象，
            # 删掉会连带毁掉简历原件，这里跳过清理（非简历来源才清理临时上传对象）。
            if source_type != "resume":
                try:
                    minio_storage.delete_object(object_key)
                except Exception as cleanup_err:
                    logger.debug("清理重复上传对象失败: %s", cleanup_err)
            return

        doc = KnowledgeDocument(
            id=doc_id,
            kb_id=kb_id,
            file_name=file_name,
            file_size=file_size,
            file_type=get_file_type(file_name),
            object_key=object_key,
            file_hash=file_hash,
            status="pending",
            uploaded_at=now,
            created_at=now,
            source_type=source_type or "",
            source_id=source_id or "",
        )
        db.add(doc)
        db.flush()

        task = IngestionTask(
            id=task_id,
            doc_id=doc_id,
            kb_id=kb_id,
            file_name=file_name,
            file_size=file_size,
            status="waiting",
            progress=0,
            created_at=now,
            updated_at=now,
        )
        db.add(task)
        db.commit()

        def _update(status: str, progress: int, message: str = ""):
            task.status = status
            task.progress = progress
            if message:
                task.message = message
            task.updated_at = _now_ms()
            # 同步文档状态，避免前端一直显示「等待中」导致「查看分片」灰掉
            if doc is not None:
                if status in ("completed", "failed"):
                    doc.status = status
                elif status in ("parsing", "encoding", "indexing"):
                    doc.status = status
            db.commit()

        # 2. Parse → raw text (3-layer pipeline)
        _update("parsing", 10)
        raw_text = _extract_text(local_path)
        if not raw_text or not raw_text.strip():
            doc.status = "failed"
            _update("failed", 0, "无法解析文件内容")
            return

        # 3. Clean text
        _update("parsing", 30)
        cleaned = clean_text(raw_text)

        # 4. Chunk
        _update("encoding", 50)
        chunks = chunk_document(cleaned)
        if not chunks:
            chunks = [cleaned[: settings.chunk_size]]

        # 5. Encode in batches — dual vectors (dense + sparse)
        _update("encoding", 60)
        from app.services.rag.embedding import encode_dense, encode_sparse

        all_dense = []
        all_sparse = []
        bsize = settings.ingest_batch_size
        for i in range(0, len(chunks), bsize):
            batch = chunks[i : i + bsize]
            all_dense.extend(encode_dense(batch))
            if settings.sparse_vector_enabled:
                try:
                    all_sparse.extend(encode_sparse(batch))
                except Exception:
                    all_sparse.extend([{} for _ in batch])
            else:
                all_sparse.extend([{} for _ in batch])
            pct = 60 + int(25 * (i + len(batch)) / max(len(chunks), 1))
            _update("encoding", min(pct, 85))

        # 6. Insert dual vectors → Milvus
        _update("indexing", 85)
        milvus_ids = insert_vectors(all_dense, all_sparse, [kb_id] * len(all_dense))

        # Milvus 写入失败（连接假死/重试耗尽）→ 明确失败，不落库残缺分片。
        # 否则会存下一堆 milvus_pk=0 的分片：能在「查看分片」里看到，却永远
        # 检索不到，形成静默脏数据。宁可整篇失败让用户重传。
        if all_dense and not milvus_ids:
            doc.status = "failed"
            _update("failed", 0, "向量写入失败（Milvus 不可用），请稍后重试上传")
            return

        # 7. Store chunks → PostgreSQL
        _update("indexing", 95)
        for i, text in enumerate(chunks):
            mpk = milvus_ids[i] if i < len(milvus_ids) else 0
            db.add(
                KnowledgeChunk(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    chunk_index=i,
                    chunk_text=text,
                    milvus_pk=mpk,
                    created_at=_now_ms(),
                )
            )

        # 8. Finalize counters
        doc.status = "completed"
        doc.chunk_count = len(chunks)

        kb = db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        ).scalar_one_or_none()
        if kb:
            kb.doc_count = (kb.doc_count or 0) + 1
            kb.chunk_count = (kb.chunk_count or 0) + len(chunks)
            kb.updated_at = _now_ms()

        _update("completed", 100, f"入库完成，共 {len(chunks)} 个分片")
        logger.info(
            "文档入库完成: doc_id=%s kb=%s chunks=%d",
            doc_id,
            kb_id,
            len(chunks),
        )

    except Exception as e:
        logger.exception("入库管道异常: doc_id=%s", doc_id)
        try:
            if doc:
                doc.status = "failed"
            if task:
                task.status = "failed"
                task.message = str(e)[:500]
                task.updated_at = _now_ms()
            db.commit()
        except Exception:
            pass
    finally:
        db.close()
        # Clean up the temp file downloaded from MinIO
        if local_path:
            try:
                os.remove(local_path)
            except OSError:
                pass


# ── Public API ──────────────────────────────────────────────

def ingest_file_async(
    file_path: str,
    file_name: str,
    kb_id: str,
    file_size: int = 0,
    source_type: str = "",
    source_id: str = "",
):
    """Launch ingestion in a background thread.

    ``file_path`` is a MinIO object key. The worker downloads it to a temp file
    for parsing and removes the temp file when done.

    Args:
        source_type / source_id: 来源追踪。简历自动入库时传
            ``source_type='resume'``、``source_id=<candidate_id>``，删除文档时
            据此级联删除候选人。

    Returns (doc_id, task_id) immediately.
    Client polls GET /api/v1/rag/ingest-tasks/{task_id} for progress.
    """
    if file_size <= 0:
        # Best-effort: stat the MinIO object to get its size if not provided.
        try:
            from app.infrastructure import minio_storage
            stat = minio_storage._get_client().stat_object(
                minio_storage.settings.minio_bucket, file_path
            )
            file_size = stat.size or 0
        except Exception:
            file_size = 0

    if file_size > settings.max_file_size:
        raise ValueError(f"文件过大: {file_size} > {settings.max_file_size}")

    doc_id = _short_uuid("doc")
    task_id = _short_uuid("task")

    t = threading.Thread(
        target=_run_ingest_sync,
        args=(doc_id, task_id, file_path, file_name, kb_id, file_size, source_type, source_id),
        daemon=True,
    )
    t.start()
    return doc_id, task_id
