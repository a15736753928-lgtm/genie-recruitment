"""RAG (Retrieval-Augmented Generation) API routes.

Blueprint alignment: Section 10
Endpoints for knowledge base management, document ingestion,
semantic search/retrieval, and dashboard statistics.
"""

import asyncio
import os
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, delete

from app.database import get_db
from app.config import get_settings
from app.models.knowledge import (
    KnowledgeBase, KnowledgeDocument, KnowledgeChunk, IngestionTask, _now_ms, _short_uuid,
)
from app.services.rag.search_service import search, get_chunk_content
from app.services.rag.ingest_service import ingest_file_async
from app.services.rag.utils import (
    validate_kb_id, check_extension, check_size, SUPPORTED_EXTENSIONS,
)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG 知识库"])
settings = get_settings()

os.makedirs(settings.upload_dir, exist_ok=True)


# ── Knowledge Base CRUD ─────────────────────────────────

@router.get("/knowledge-bases")
async def list_knowledge_bases(
    keyword: str = Query(""),
    page: int = Query(1),
    page_size: int = Query(20),
    db: AsyncSession = Depends(get_db),
):
    """List all knowledge bases with pagination and optional search."""
    query = select(KnowledgeBase)
    count_query = select(func.count()).select_from(KnowledgeBase)

    if keyword:
        like = f"%{keyword}%"
        query = query.where(KnowledgeBase.name.ilike(like))
        count_query = count_query.where(KnowledgeBase.name.ilike(like))

    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(desc(KnowledgeBase.updated_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    kbs = result.scalars().all()

    return {
        "code": 200,
        "message": "success",
        "data": {
            "list": [
                {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description or "",
                    "docCount": kb.doc_count or 0,
                    "chunkCount": kb.chunk_count or 0,
                    "createdAt": kb.created_at,
                    "updatedAt": kb.updated_at,
                }
                for kb in kbs
            ],
            "total": total,
            "page": page,
            "pageSize": page_size,
        },
    }


@router.post("/knowledge-bases")
async def create_knowledge_base(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge base."""
    name = body.get("name", "").strip()
    if not name:
        return {"code": 400, "message": "知识库名称不能为空", "data": None}

    # Check uniqueness
    existing = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.name == name)
    )
    if existing.scalar_one_or_none():
        return {"code": 409, "message": "知识库名称已存在", "data": None}

    kb = KnowledgeBase(
        id=_short_uuid("kb"),
        name=name,
        description=body.get("description", ""),
        owner_id=None,
        created_at=_now_ms(),
        updated_at=_now_ms(),
    )
    db.add(kb)
    await db.flush()
    await db.refresh(kb)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description or "",
            "docCount": 0,
            "chunkCount": 0,
            "createdAt": kb.created_at,
            "updatedAt": kb.updated_at,
        },
    }


@router.get("/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get knowledge base details."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        return {"code": 404, "message": "知识库不存在", "data": None}

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description or "",
            "docCount": kb.doc_count or 0,
            "chunkCount": kb.chunk_count or 0,
            "createdAt": kb.created_at,
            "updatedAt": kb.updated_at,
        },
    }


@router.put("/knowledge-bases/{kb_id}")
async def update_knowledge_base(
    kb_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Update knowledge base name/description."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        return {"code": 404, "message": "知识库不存在", "data": None}

    if "name" in body:
        kb.name = body["name"]
    if "description" in body:
        kb.description = body["description"]
    kb.updated_at = _now_ms()

    await db.flush()
    await db.refresh(kb)

    return {"code": 200, "message": "已更新", "data": None}


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge base and all its documents/chunks (cascade).

    Also cleans up Milvus vectors and uploaded files.
    """
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        return {"code": 404, "message": "知识库不存在", "data": None}

    # Get milvus PKs for all chunks in this KB
    from app.infrastructure.milvus_manager import delete_by_ids
    chunk_result = await db.execute(
        select(KnowledgeChunk.milvus_pk).where(KnowledgeChunk.kb_id == kb_id)
    )
    milvus_pks = [r[0] for r in chunk_result.fetchall() if r[0] > 0]

    # Delete files for documents in this KB (stored in MinIO)
    doc_result = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.kb_id == kb_id)
    )
    from app.infrastructure import minio_storage
    for doc in doc_result.scalars().all():
        if doc.object_key:
            await asyncio.to_thread(minio_storage.delete_object, doc.object_key)

    # Cascade delete (PG handles via FK ON DELETE CASCADE)
    await db.delete(kb)
    await db.flush()

    # Clean up Milvus vectors (run blocking call off the event loop)
    if milvus_pks:
        await asyncio.to_thread(delete_by_ids, milvus_pks)

    return {"code": 200, "message": "已删除", "data": None}


# ── Document Management ─────────────────────────────────

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
):
    """Upload a file and start async ingestion.

    Step 1: Validate file type and size
    Step 2: Save file to disk
    Step 3: Launch background ingestion thread
    Step 4: Return immediately with (doc_id, task_id)

    The client should poll GET /api/v1/rag/ingest-tasks/{task_id}
    for ingestion progress.
    """
    # Validate kb_id format
    if not validate_kb_id(kb_id):
        return {
            "code": 400,
            "message": f"无效的知识库 ID 格式: {kb_id[:20]}...",
            "data": None,
        }

    # Validate file extension
    if file.filename and not check_extension(file.filename):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return {
            "code": 400,
            "message": f"不支持的文件格式。支持: {supported}",
            "data": None,
        }

    # Validate file size
    content = await file.read()
    if not check_size(len(content)):
        return {
            "code": 413,
            "message": f"文件过大: {len(content)} > {settings.max_file_size} 字节",
            "data": None,
        }

    # Save file to MinIO (object key: rag/<uuid>.<ext>)
    file_ext = os.path.splitext(file.filename or "document")[1] or ".txt"
    object_key = f"rag/{uuid.uuid4()}{file_ext}"
    try:
        from app.infrastructure import minio_storage
        await asyncio.to_thread(
            minio_storage.upload_bytes,
            object_key,
            content,
            "application/octet-stream",
        )
    except Exception as e:
        return {"code": 500, "message": f"文件存储失败: {e}", "data": None}

    # Start async ingestion (downloads from MinIO inside the worker thread)
    try:
        doc_id, task_id = ingest_file_async(
            file_path=object_key,
            file_name=file.filename or "unknown",
            kb_id=kb_id,
            file_size=len(content),
        )
    except ValueError as e:
        return {"code": 400, "message": str(e), "data": None}

    return {
        "code": 200,
        "message": "上传成功，正在后台处理",
        "data": {
            "docId": doc_id,
            "taskId": task_id,
            "fileName": file.filename,
            "fileSize": len(content),
        },
    }


@router.get("/documents")
async def list_documents(
    kb_id: str = Query(""),
    status: str = Query(""),
    page: int = Query(1),
    page_size: int = Query(20),
    db: AsyncSession = Depends(get_db),
):
    """List documents with optional KB filter."""
    query = select(KnowledgeDocument)
    count_query = select(func.count()).select_from(KnowledgeDocument)

    if kb_id:
        query = query.where(KnowledgeDocument.kb_id == kb_id)
        count_query = count_query.where(KnowledgeDocument.kb_id == kb_id)

    if status:
        query = query.where(KnowledgeDocument.status == status)
        count_query = count_query.where(KnowledgeDocument.status == status)

    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(desc(KnowledgeDocument.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    docs = result.scalars().all()

    # Fetch KB names
    kb_ids = list({d.kb_id for d in docs})
    kb_map = {}
    if kb_ids:
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))
        )
        kb_map = {k.id: k.name for k in kb_result.scalars().all()}

    def _format_ts(ms: int) -> str:
        if not ms:
            return ""
        dt = datetime.utcfromtimestamp(ms / 1000)
        return dt.strftime("%Y-%m-%d %H:%M")

    return {
        "code": 200,
        "message": "success",
        "data": {
            "list": [
                {
                    "id": d.id,
                    "kbId": d.kb_id,
                    "kbName": kb_map.get(d.kb_id, ""),
                    "fileName": d.file_name,
                    "fileType": d.file_type,
                    "fileSize": d.file_size,
                    "chunkCount": d.chunk_count or 0,
                    "status": d.status,
                    "ingestedAt": _format_ts(d.uploaded_at),
                    "createdAt": d.created_at,
                }
                for d in docs
            ],
            "total": total,
            "page": page,
            "pageSize": page_size,
        },
    }


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get document details."""
    result = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return {"code": 404, "message": "文档不存在", "data": None}

    kb_result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
    )
    kb = kb_result.scalar_one_or_none()

    def _format_ts(ms: int) -> str:
        if not ms:
            return ""
        return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": doc.id,
            "kbId": doc.kb_id,
            "kbName": kb.name if kb else "",
            "fileName": doc.file_name,
            "fileType": doc.file_type,
            "fileSize": doc.file_size,
            "fileHash": doc.file_hash,
            "chunkCount": doc.chunk_count or 0,
            "status": doc.status,
            "ingestedAt": _format_ts(doc.uploaded_at),
        },
    }


@router.get("/documents/{doc_id}/file")
async def get_document_file(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stream a document's original file content (preview / download).

    Supports both MinIO object keys and legacy absolute local file paths stored
    in ``object_key`` before the MinIO migration.
    """
    result = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
    )
    doc = result.scalar_one_or_none()
    if not doc or not doc.object_key:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "文档或原始文件不存在", "data": None},
        )

    from starlette.responses import StreamingResponse, FileResponse

    ext = os.path.splitext(doc.file_name or "")[1].lower()
    media_type = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")

    filename = doc.file_name or f"{doc.id}{ext}"
    stored = doc.object_key
    from app.services.rag.utils import content_disposition_header
    cd_header = content_disposition_header(filename, "inline")

    # 1) Legacy absolute local path that still exists on disk → stream from disk.
    if os.path.isabs(stored) and os.path.isfile(stored):
        return FileResponse(
            path=stored,
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": cd_header},
        )

    # 2) Otherwise treat ``stored`` as a MinIO object key and stream from MinIO.
    from app.infrastructure import minio_storage

    # Verify the object actually exists in MinIO before opening a stream —
    # otherwise the StreamingResponse would start, fail mid-stream, and the
    # client would receive a truncated/empty body with no clear error.
    if not minio_storage.object_exists(stored):
        return JSONResponse(
            status_code=404,
            content={
                "code": 404,
                "message": "原始文件不存在或已被清理，请重新上传该文档",
                "data": None,
            },
        )

    try:
        response = await asyncio.to_thread(minio_storage.get_object_stream, stored)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"读取文件失败: {e}", "data": None},
        )

    def _iter():
        try:
            for chunk in response.stream(amt=64 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    return StreamingResponse(
        _iter(),
        media_type=media_type,
        headers={"Content-Disposition": cd_header},
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document, its chunks (PG + Milvus), and its file.

    若该文档属于「简历」知识库，会一并级联删除对应候选人及其所有下游数据
    （面试 / 试用期 / 绩效 / 人才库 / 简历原件）。删除逻辑与「简历管理」
    的删除共用同一套服务（app.services.cascade_delete），保证两边行为一致。
    """
    import logging
    log = logging.getLogger("genie.rag.delete")
    log.info("删除文档开始 doc_id=%s", doc_id)

    result = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        log.warning("删除文档：文档不存在 doc_id=%s", doc_id)
        return {"code": 404, "message": "文档不存在", "data": None}

    log.info(
        "删除文档 doc_id=%s file=%s kb=%s source_type=%s source_id=%s",
        doc_id, doc.file_name, doc.kb_id,
        getattr(doc, "source_type", "") or "",
        getattr(doc, "source_id", "") or "",
    )

    from app.services.system.cascade_delete import cascade_delete_by_kb_document
    try:
        await cascade_delete_by_kb_document(db, doc)
    except Exception as e:
        log.warning("级联删除失败（继续）: %s doc_id=%s", e, doc_id)
        # 兜底：确保文档本身被删
        await db.delete(doc)
        await db.flush()

    log.info("删除文档完成 doc_id=%s", doc_id)
    return {"code": 200, "message": "已删除", "data": None}


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1),
    page_size: int = Query(50),
    db: AsyncSession = Depends(get_db),
):
    """Get chunks for a document (paginated preview)."""
    # Verify document exists and check ownership
    doc_result = await db.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        return {"code": 404, "message": "文档不存在", "data": None}

    count_result = await db.execute(
        select(func.count()).select_from(KnowledgeChunk)
        .where(KnowledgeChunk.doc_id == doc_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(KnowledgeChunk)
        .where(KnowledgeChunk.doc_id == doc_id)
        .order_by(KnowledgeChunk.chunk_index)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    chunks = result.scalars().all()

    return {
        "code": 200,
        "message": "success",
        "data": {
            "list": [
                {
                    "id": c.milvus_pk,
                    "chunkIndex": c.chunk_index,
                    "text": c.chunk_text,
                    "charCount": len(c.chunk_text),
                }
                for c in chunks
            ],
            "total": total,
            "page": page,
            "pageSize": page_size,
        },
    }


# ── Ingestion Task Tracking ─────────────────────────────

@router.get("/ingest-tasks/{task_id}")
async def get_ingest_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll ingestion progress."""
    result = await db.execute(
        select(IngestionTask).where(IngestionTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        return {"code": 404, "message": "任务不存在", "data": None}

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": task.id,
            "docId": task.doc_id,
            "kbId": task.kb_id,
            "fileName": task.file_name,
            "fileSize": task.file_size,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "createdAt": task.created_at,
            "updatedAt": task.updated_at,
        },
    }


# ── Search / Retrieval ──────────────────────────────────

@router.post("/search")
async def search_query(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Core RAG search endpoint.

    Request body:
    {
        "query": "BGE-M3 模型的嵌入维度是多少？",
        "kb_ids": ["kb_abc12345"],     // optional, null = all KBs
        "top_k": 10,                    // optional, 0 = unlimited (max 5000)
        "min_similarity": 0.0           // optional
    }

    Response: list of matched chunks with metadata and highlights.
    """
    query_text = body.get("query", "")
    if not query_text or not query_text.strip():
        return {"code": 400, "message": "请提供查询文本", "data": []}

    kb_ids = body.get("kb_ids")
    top_k = body.get("top_k", settings.default_top_k)
    if top_k == 0 or top_k is None:
        top_k = min(settings.max_search_recall, 50)

    # 未传 min_similarity 时默认读系统设置 recallThreshold
    if "min_similarity" in body:
        min_sim = float(body.get("min_similarity") or 0.0)
    else:
        from app.services.system.system_settings import get_system_setting
        min_sim = float(await get_system_setting(db, "recallThreshold", 0.75) or 0.75)

    try:
        results = await search(
            query=query_text,
            kb_ids=kb_ids,
            top_k=top_k,
            min_similarity=min_sim,
        )
    except Exception as e:
        print(f"[Search] Error: {e}")
        return {"code": 500, "message": f"检索失败: {str(e)}", "data": []}

    return {
        "code": 200,
        "message": "success",
        "data": results,
    }


@router.get("/search/chunk-content")
async def get_chunk(
    milvus_pk: int = Query(...),
):
    """Get full chunk text by its Milvus primary key."""
    result = await get_chunk_content(milvus_pk)
    if not result:
        return {"code": 404, "message": "分片不存在", "data": None}
    return {"code": 200, "message": "success", "data": result}


# ── Dashboard Stats ─────────────────────────────────────

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
):
    """Get RAG dashboard overview stats."""
    # KB count
    kb_total = (await db.execute(
        select(func.count()).select_from(KnowledgeBase)
    )).scalar() or 0

    # Document count
    doc_total = (await db.execute(
        select(func.count()).select_from(KnowledgeDocument)
    )).scalar() or 0

    # Chunk count
    chunk_total = (await db.execute(
        select(func.count()).select_from(KnowledgeChunk)
    )).scalar() or 0

    # Vector count (from Milvus). get_collection_stats() is a blocking
    # synchronous Milvus Lite call — running it directly in the async handler
    # would block the event loop and freeze the whole server (including
    # unrelated endpoints). Run it in a worker thread with a timeout so a
    # stuck Milvus can never deadlock the API.
    from app.infrastructure.milvus_manager import get_collection_stats
    try:
        milvus_stats = await asyncio.wait_for(
            asyncio.to_thread(get_collection_stats), timeout=5.0
        )
        vector_count = milvus_stats.get("row_count", 0)
    except asyncio.TimeoutError:
        vector_count = 0
    except Exception:
        vector_count = 0

    # Recent documents
    recent_result = await db.execute(
        select(KnowledgeDocument)
        .order_by(desc(KnowledgeDocument.created_at))
        .limit(10)
    )
    recent_docs = recent_result.scalars().all()

    # KB names map
    kb_ids = list({d.kb_id for d in recent_docs})
    kb_map = {}
    if kb_ids:
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))
        )
        kb_map = {k.id: k.name for k in kb_result.scalars().all()}

    def _format_ts(ms: int) -> str:
        if not ms:
            return ""
        return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")

    return {
        "code": 200,
        "message": "success",
        "data": {
            "kbCount": kb_total,
            "documentCount": doc_total,
            "chunkCount": chunk_total,
            "vectorCount": vector_count,
            "recentDocuments": [
                {
                    "id": d.id,
                    "fileName": d.file_name,
                    "fileType": d.file_type,
                    "kbName": kb_map.get(d.kb_id, ""),
                    "chunkCount": d.chunk_count or 0,
                    "status": d.status,
                    "ingestedAt": _format_ts(d.uploaded_at),
                }
                for d in recent_docs
            ],
        },
    }


@router.get("/dashboard/distribution")
async def get_distribution(
    db: AsyncSession = Depends(get_db),
):
    """Get document distribution across knowledge bases."""
    kbs_result = await db.execute(select(KnowledgeBase))
    kbs = kbs_result.scalars().all()

    distribution = []
    for kb in kbs:
        doc_count = (await db.execute(
            select(func.count()).select_from(KnowledgeDocument)
            .where(KnowledgeDocument.kb_id == kb.id)
        )).scalar() or 0
        distribution.append({
            "kbId": kb.id,
            "kbName": kb.name,
            "docCount": doc_count,
            "chunkCount": kb.chunk_count or 0,
        })

    distribution.sort(key=lambda x: x["docCount"], reverse=True)
    return {"code": 200, "message": "success", "data": distribution}
