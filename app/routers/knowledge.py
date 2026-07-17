import os
import uuid
import json
import asyncio
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import OpenAI
from app.database import get_db
from app.models.knowledge import KnowledgeCategory, KnowledgeItem
from app.config import get_settings

router = APIRouter(tags=["知识库"])
settings = get_settings()

llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

os.makedirs(settings.upload_dir, exist_ok=True)

# ── Milvus Lite setup ───────────────────────────────────
try:
    from milvus import default_server
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

MILVUS_COLLECTION = "hr_knowledge_chunks"
_milvus_connected = False


def ensure_milvus():
    global _milvus_connected
    if not MILVUS_AVAILABLE:
        return False
    if _milvus_connected:
        return True
    try:
        if not default_server.is_running():
            default_server.start()
        connections.connect(host="127.0.0.1", port=default_server.listen_port)
        _milvus_connected = True
        _init_collection()
        return True
    except Exception as e:
        print(f"Milvus connection error: {e}")
        return False


def _init_collection():
    if utility.has_collection(MILVUS_COLLECTION):
        return
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="knowledge_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="category_key", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
    ]
    schema = CollectionSchema(fields, "HR knowledge chunks")
    Collection(MILVUS_COLLECTION, schema)


def text_to_chunks(text: str, chunk_size: int = 512) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - 50  # 50 char overlap
        if start >= len(text):
            break
    return chunks


def get_embedding(text: str) -> List[float]:
    """Get embedding from DeepSeek API."""
    try:
        response = llm_client.embeddings.create(
            model="deepseek-chat",
            input=text[:8000],  # Truncate
        )
        return response.data[0].embedding
    except Exception:
        # Fallback: zero vector
        return [0.0] * settings.embedding_dim


def store_in_milvus(knowledge_id: str, category_key: str, type: str, content: str) -> List[str]:
    """Store content chunks in Milvus. Returns list of chunk IDs."""
    if not ensure_milvus():
        return []

    chunks = text_to_chunks(content)
    milvus_ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{knowledge_id}_{i}"
        try:
            embedding = get_embedding(chunk)
            collection = Collection(MILVUS_COLLECTION)
            collection.insert([{
                "id": chunk_id,
                "knowledge_id": knowledge_id,
                "category_key": category_key,
                "type": type,
                "content": chunk,
                "embedding": embedding,
            }])
            milvus_ids.append(chunk_id)
        except Exception as e:
            print(f"Milvus insert error: {e}")

    if milvus_ids:
        try:
            collection = Collection(MILVUS_COLLECTION)
            collection.flush()
        except Exception:
            pass

    return milvus_ids


def delete_from_milvus(knowledge_id: str):
    """Delete all chunks for a knowledge item from Milvus."""
    if not ensure_milvus():
        return
    try:
        collection = Collection(MILVUS_COLLECTION)
        collection.delete(f'knowledge_id == "{knowledge_id}"')
        collection.flush()
    except Exception as e:
        print(f"Milvus delete error: {e}")


def search_milvus(query: str, top_k: int = 5, category_key: str = "") -> List[dict]:
    """Search Milvus for relevant chunks."""
    if not ensure_milvus():
        return []

    try:
        embedding = get_embedding(query)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        expr = f'category_key == "{category_key}"' if category_key else None

        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "knowledge_id", "content", "category_key"],
        )

        output = []
        for hits in results:
            for hit in hits:
                output.append({
                    "id": hit.entity.get("id"),
                    "content": hit.entity.get("content"),
                    "score": round(hit.score, 4),
                    "source": hit.entity.get("knowledge_id"),
                })
        return output
    except Exception as e:
        print(f"Milvus search error: {e}")
        return []


# ── Helpers ─────────────────────────────────────────────

def extract_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        from PyPDF2 import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    elif ext in ('.docx', '.doc'):
        from docx import Document
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    elif ext in ('.txt', '.md'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return ""


def build_category_tree(categories: List[KnowledgeCategory], parent_key: Optional[str] = None) -> List[dict]:
    children = [c for c in categories if c.parent_key == parent_key]
    result = []
    for child in children:
        item_count = len(child.items) if child.items else 0
        node = {
            "key": child.key,
            "title": child.title,
            "count": item_count,
            "children": build_category_tree(categories, child.key),
        }
        result.append(node)
    return result


# ── Endpoints ───────────────────────────────────────────

@router.get("/knowledge/stats")
async def get_knowledge_stats(db: AsyncSession = Depends(get_db)):
    total_result = await db.execute(select(func.count()).select_from(KnowledgeItem))
    total = total_result.scalar() or 0

    # New this month
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0)
    new_result = await db.execute(
        select(func.count()).select_from(KnowledgeItem)
        .where(KnowledgeItem.created_at >= month_start)
    )
    new_count = new_result.scalar() or 0

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "total": total,
            "newThisMonth": new_count,
        },
    }


@router.get("/knowledge/categories")
async def get_categories(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(KnowledgeCategory).options(selectinload(KnowledgeCategory.items))
    )
    categories = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": build_category_tree(categories),
    }


@router.get("/knowledge")
async def list_knowledge(
    categoryKey: str = Query("all"),
    keyword: str = Query(""),
    sortBy: str = Query("updateTime"),
    page: int = Query(1),
    pageSize: int = Query(10),
    db: AsyncSession = Depends(get_db),
):
    query = select(KnowledgeItem)

    if categoryKey and categoryKey != "all":
        query = query.where(KnowledgeItem.category_key == categoryKey)
    if keyword:
        query = query.where(KnowledgeItem.name.ilike(f"%{keyword}%"))

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    sort_col = KnowledgeItem.updated_at if sortBy == "updateTime" else KnowledgeItem.recall_count
    query = query.order_by(desc(sort_col))
    query = query.offset((page - 1) * pageSize).limit(pageSize)

    result = await db.execute(query)
    items = result.scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "list": [
                {
                    "id": str(item.id),
                    "name": item.name,
                    "category": item.category_key or "",
                    "categoryPath": item.category_path or "",
                    "type": item.type,
                    "recallCount": item.recall_count or 0,
                    "updateTime": item.updated_at.strftime("%Y-%m-%d %H:%M") if item.updated_at else "",
                    "content": item.content,
                }
                for item in items
            ],
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.post("/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename or "knowledge")[1] or ".pdf"
    saved_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.upload_dir, saved_name)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "fileId": saved_name,
            "fileName": file.filename or "unknown",
            "size": len(content),
        },
    }


@router.post("/knowledge")
async def create_knowledge_item(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    file_id = body.get("fileId")
    file_path = ""
    content_text = ""

    if file_id:
        file_path = os.path.join(settings.upload_dir, file_id)
        if os.path.exists(file_path):
            content_text = extract_file_text(file_path)

    item = KnowledgeItem(
        name=body.get("name", "未命名素材"),
        category_key=body.get("category", ""),
        category_path=body.get("categoryPath", ""),
        type=body.get("type", "interview"),
        file_path=file_path,
        content=content_text,
    )
    db.add(item)
    await db.flush()

    if content_text:
        item.milvus_ids = await asyncio.to_thread(
            store_in_milvus,
            str(item.id),
            item.category_key or "",
            item.type or "interview",
            content_text,
        )

    await db.refresh(item)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(item.id),
            "name": item.name,
            "category": item.category_key or "",
            "categoryPath": item.category_path or "",
            "type": item.type,
            "recallCount": item.recall_count or 0,
            "updateTime": item.updated_at.strftime("%Y-%m-%d %H:%M") if item.updated_at else "",
        },
    }


@router.put("/knowledge/{item_id}")
async def update_knowledge_item(
    item_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(KnowledgeItem).where(KnowledgeItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        return {"code": 404, "message": "素材不存在", "data": None}

    for field in ["name", "categoryPath", "type"]:
        if field in body:
            setattr(item, field, body[field])

    if "category" in body:
        item.category_key = body["category"]

    await db.flush()
    await db.refresh(item)
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(item.id),
            "name": item.name,
            "category": item.category_key or "",
            "categoryPath": item.category_path or "",
            "type": item.type,
            "recallCount": item.recall_count or 0,
            "updateTime": item.updated_at.strftime("%Y-%m-%d %H:%M") if item.updated_at else "",
        },
    }


@router.delete("/knowledge/{item_id}")
async def delete_knowledge_item(
    item_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(KnowledgeItem).where(KnowledgeItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        return {"code": 404, "message": "素材不存在", "data": None}

    # Delete from Milvus
    delete_from_milvus(str(item.id))

    # Delete file
    if item.file_path and os.path.exists(item.file_path):
        try:
            os.remove(item.file_path)
        except Exception:
            pass

    await db.delete(item)
    return {"code": 0, "message": "ok", "data": None}


@router.post("/knowledge/recall-test")
async def recall_test(body: dict, db: AsyncSession = Depends(get_db)):
    query = body.get("query", "")
    if not query:
        return {"code": 400, "message": "请提供查询文本", "data": []}

    # Search Milvus
    results = await asyncio.to_thread(search_milvus, query, 5)

    # Enrich with source names
    for r in results:
        if r.get("source"):
            item_result = await db.execute(
                select(KnowledgeItem).where(KnowledgeItem.id == r["source"])
            )
            item = item_result.scalar_one_or_none()
            if item:
                r["source"] = item.name
                # Update recall count
                item.recall_count = (item.recall_count or 0) + 1

    return {"code": 0, "message": "ok", "data": results}
