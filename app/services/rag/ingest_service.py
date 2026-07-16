"""Document ingestion pipeline — 3-layer parsing with OCR fallback.

Blueprint: file → parse (fast → render → OCR) → clean → chunk → embed → store.

Matches the reference RAG system's 3-layer approach:
  1. Fast text extraction (PyPDF2 / python-docx)
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
from app.core.milvus_manager import insert_vectors

settings = get_settings()
logger = logging.getLogger(__name__)

# OCR fallback threshold: trigger OCR when fast extraction yields fewer chars
_OCR_FALLBACK_THRESHOLD = settings.ocr_fallback_threshold

# RapidOCR singleton
_ocr = None


# ── File Parsing (3-layer pipeline) ─────────────────────────

def _extract_text(file_path: str) -> str:
    """Parse file to text with format-specific extractors.

    Layer 1: Fast text extraction (PyPDF2, python-docx, raw read)
    Layer 2: OCR fallback for images and low-text documents
    Layer 3: Merge results
    """
    ext = Path(file_path).suffix.lower()

    # ── Plain text: direct decode ──
    if ext in (".txt", ".md", ".markdown"):
        return _read_text_file(file_path)

    # ── Images: directly to OCR ──
    if ext in (".png", ".jpg", ".jpeg"):
        with open(file_path, "rb") as f:
            return _ocr_image(f.read())

    # ── PDF / DOCX: fast path first, OCR fallback ──
    fast_text = _extract_text_fast(file_path, ext)
    if len(fast_text.strip()) >= _OCR_FALLBACK_THRESHOLD:
        return fast_text

    logger.info(
        "快速提取文本不足（%d 字符），触发 OCR 兜底: %s",
        len(fast_text.strip()),
        os.path.basename(file_path),
    )
    ocr_text = _ocr_document(file_path, ext)
    return _merge_texts(fast_text, ocr_text)


def _read_text_file(path: str) -> str:
    """Read text file with encoding detection."""
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def _extract_text_fast(path: str, ext: str) -> str:
    """Layer 1: Fast native text extraction."""
    if ext == ".pdf":
        return _extract_pdf_text(path)
    elif ext == ".docx":
        return _extract_docx_text(path)
    return ""


def _extract_pdf_text(path: str) -> str:
    """Extract text from PDF using PyPDF2 (fast path)."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    texts.append(t.strip())
            except Exception:
                pass
        return "\n\n".join(texts)
    except Exception as e:
        logger.debug("PyPDF2 解析失败: %s", e)
        return ""


def _extract_docx_text(path: str) -> str:
    """Extract text from DOCX using python-docx (fast path)."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.debug("python-docx 解析失败: %s", e)
        return ""


# ── Layer 2: OCR fallback ──────────────────────────────────

def _ocr_document(path: str, ext: str) -> str:
    """OCR for documents with insufficient text."""
    if ext == ".pdf":
        return _ocr_pdf(path)
    elif ext == ".docx":
        return _ocr_docx_images(path)
    return ""


def _ocr_pdf(path: str) -> str:
    """Render PDF pages to images, then OCR each page."""
    if not settings.ocr_enabled:
        return ""

    try:
        from PyPDF2 import PdfReader
        import fitz  # PyMuPDF for rendering
        reader = PdfReader(path)
        doc = fitz.open(path)
        texts = []

        for i, page in enumerate(reader.pages):
            # Check if page already has good text
            page_text = (page.extract_text() or "").strip()
            if len(page_text) >= _OCR_FALLBACK_THRESHOLD:
                texts.append(page_text)
                continue

            # Render page to image → OCR
            try:
                pix = doc[i].get_pixmap(dpi=200)
                ocr_text = _ocr_image(pix.tobytes("png"))
                if ocr_text.strip():
                    texts.append(ocr_text)
                elif page_text:
                    texts.append(page_text)
            except Exception:
                if page_text:
                    texts.append(page_text)

        doc.close()
        return "\n\n".join(texts)
    except ImportError:
        logger.debug("PyMuPDF (fitz) 不可用，跳过 PDF OCR")
        return ""
    except Exception as e:
        logger.debug("PDF OCR 失败: %s", e)
        return ""


def _ocr_docx_images(path: str) -> str:
    """Extract embedded images from DOCX and OCR them."""
    if not settings.ocr_enabled:
        return ""

    try:
        from docx import Document
        doc = Document(path)
        texts = []

        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                img_bytes = rel.target_part.blob
                ocr_text = _ocr_image(img_bytes)
                if ocr_text.strip():
                    texts.append(ocr_text.strip())
            except Exception:
                pass

        return "\n\n".join(texts)
    except Exception as e:
        logger.debug("DOCX 图片 OCR 失败: %s", e)
        return ""


def _ocr_image(img_data: bytes) -> str:
    """OCR a single image using RapidOCR (PP-OCR v4)."""
    if not settings.ocr_enabled:
        return ""

    ocr = _get_ocr()
    if ocr is None:
        return ""

    try:
        result, _ = ocr(img_data)
        if not result:
            return ""
        lines = [item[1].strip() for item in result if item[1] and item[1].strip()]
        return "\n".join(lines)
    except Exception as e:
        logger.debug("OCR 识别失败: %s", e)
        return ""


def _get_ocr():
    """RapidOCR global singleton, lazy-loaded on first call."""
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
            logger.info("RapidOCR 模型加载完成")
        except ImportError:
            logger.warning("rapidocr-onnxruntime 未安装，OCR 功能不可用")
            return None
    return _ocr


def _merge_texts(fast_text: str, ocr_text: str) -> str:
    """Merge fast extraction and OCR results, preferring OCR when richer."""
    ocr = ocr_text.strip()
    if len(ocr) >= _OCR_FALLBACK_THRESHOLD:
        return ocr
    fast = fast_text.strip()
    if fast:
        return fast
    return ocr  # fallback: whatever we have


# ── Background Ingestion ────────────────────────────────────

def _run_ingest_sync(
    doc_id: str,
    task_id: str,
    file_path: str,
    file_name: str,
    kb_id: str,
    file_size: int = 0,
):
    """Full ingestion pipeline (runs in background thread).

    Uses sync DB session to avoid event-loop conflicts with async FastAPI.
    """
    now = _now_ms()
    db = get_sync_db()
    doc = None
    task = None

    try:
        # Compute file hash for dedup
        file_hash = ""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

        # 1. Create document + task records
        doc = KnowledgeDocument(
            id=doc_id,
            kb_id=kb_id,
            file_name=file_name,
            file_size=file_size,
            file_type=get_file_type(file_path),
            object_key=file_path,
            file_hash=file_hash,
            status="pending",
            uploaded_at=now,
            created_at=now,
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
            db.commit()

        # 2. Parse → raw text (3-layer pipeline)
        _update("parsing", 10)
        raw_text = _extract_text(file_path)
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

        # 5. Encode in batches with real embeddings
        _update("encoding", 60)
        all_vectors = []
        bsize = settings.ingest_batch_size
        for i in range(0, len(chunks), bsize):
            batch = chunks[i : i + bsize]
            all_vectors.extend(encode_batch(batch))
            pct = 60 + int(25 * (i + len(batch)) / max(len(chunks), 1))
            _update("encoding", min(pct, 85))

        # 6. Insert vectors → Milvus
        _update("indexing", 85)
        milvus_ids = insert_vectors(all_vectors, [kb_id] * len(all_vectors))

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


# ── Public API ──────────────────────────────────────────────

def ingest_file_async(file_path: str, file_name: str, kb_id: str):
    """Launch ingestion in a background thread.

    Returns (doc_id, task_id) immediately.
    Client polls GET /api/v1/rag/ingest-tasks/{task_id} for progress.
    """
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    if file_size > settings.max_file_size:
        raise ValueError(f"文件过大: {file_size} > {settings.max_file_size}")

    doc_id = _short_uuid("doc")
    task_id = _short_uuid("task")

    t = threading.Thread(
        target=_run_ingest_sync,
        args=(doc_id, task_id, file_path, file_name, kb_id, file_size),
        daemon=True,
    )
    t.start()
    return doc_id, task_id
