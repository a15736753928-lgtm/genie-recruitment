"""Knowledge / RAG tool handlers."""

from __future__ import annotations

import os
import io
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure import minio_storage


async def _rag_search(params: dict, db: AsyncSession) -> str:
    try:
        from app.services.rag.search_service import search as rag_search_fn
        from app.services.system.system_settings import get_system_setting
        top_k = params.get("topK", 5)
        recall_threshold = float(await get_system_setting(db, "recallThreshold", 0.75) or 0.75)
        results = await rag_search_fn(
            query=params["query"],
            kb_ids=None,
            top_k=top_k,
            min_similarity=recall_threshold,
        )
        if results:
            summaries = []
            for r in results[:3]:
                summaries.append(
                    f"[{r['kb_name']}] {r['file_name']}: {r['content'][:300]}"
                )
            return f"检索到 {len(results)} 条相关知识:\n" + "\n---\n".join(summaries)
        return "未检索到相关知识"
    except Exception:
        return "知识库检索暂不可用"


async def _list_knowledge(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import list_knowledge as fn
    result = await fn(categoryKey=params.get("categoryKey", "all"), keyword=params.get("keyword", ""), db=db)
    data = result.get("data", {})
    if not isinstance(data, dict):
        return "查询完成"
    total = data.get("total", 0)
    items = data.get("list") or data.get("items") or []
    if not items:
        return f"知识库：共 {total} 条素材（无明细）"
    limit = int(params.get("limit", 15) or 15)
    lines = [f"知识库：共 {total} 条，前 {min(limit, len(items))} 条："]
    for it in items[:limit]:
        lines.append(
            f"  [{it.get('id','')}] {it.get('name') or it.get('title','')} | "
            f"分类:{it.get('category') or it.get('categoryKey','—')} | "
            f"类型:{it.get('type','—')}"
        )
    return "\n".join(lines)


async def _get_knowledge_stats(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import get_knowledge_stats as fn
    result = await fn(db=db)
    d = result.get("data", {})
    return f"知识库统计：共 {d.get('total', 0)} 条，本月新增 {d.get('newThisMonth', 0)}"


async def _get_knowledge_categories(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import get_categories as fn
    result = await fn(db=db)
    data = result.get("data", [])
    return f"知识库分类共 {len(data)} 个根分类"


async def _upload_knowledge_file(params: dict, db: AsyncSession) -> str:
    object_key = params["fileKey"]
    file_name = params["fileName"]
    return f"知识库素材文件「{file_name}」已就绪，object key={object_key}。请接着调用 create_knowledge_item 完成入库。"


async def _create_knowledge_item(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import create_knowledge_item as fn
    result = await fn(body=params, db=db)
    if result["code"] == 0:
        return f"已创建知识库素材「{params.get('name', '')}」"
    return f"创建失败：{result.get('message', '')}"


async def _update_knowledge_item(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import update_knowledge_item as fn
    result = await fn(item_id=params["id"], body=params.get("fields", {}), db=db)
    return f"已更新素材 {params['id']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"


async def _delete_knowledge_item(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import delete_knowledge_item as fn
    result = await fn(item_id=params["id"], db=db)
    return f"已删除素材 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"


async def _recall_test(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.knowledge_base import recall_test as fn
    result = await fn(body={"query": params["query"]}, db=db)
    data = result.get("data", [])
    return f"召回测试命中 {len(data)} 条"


async def _list_knowledge_bases(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import list_knowledge_bases as fn
    result = await fn(keyword=params.get("keyword", ""), db=db)
    d = result.get("data", {})
    return f"RAG知识库共 {d.get('total', 0)} 个"


async def _create_knowledge_base(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import create_knowledge_base as fn
    result = await fn(body={"name": params["name"], "description": params.get("description", "")}, db=db)
    return f"已创建RAG知识库「{params['name']}」" if result.get("code") == 200 or result.get("code") == 0 else f"创建失败：{result.get('message', '')}"


async def _update_knowledge_base(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import update_knowledge_base as fn
    result = await fn(kb_id=params["id"], body=params.get("fields", {}), db=db)
    return f"已更新RAG知识库 {params['id']}" if result.get("code") in (0, 200) else f"更新失败：{result.get('message', '')}"


async def _delete_knowledge_base(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import delete_knowledge_base as fn
    result = await fn(kb_id=params["id"], db=db)
    return f"已删除RAG知识库 {params['id']}" if result.get("code") in (0, 200) else f"删除失败：{result.get('message', '')}"


async def _upload_document(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import upload_document as fn
    from fastapi import UploadFile
    object_key = params["fileKey"]
    file_name = params["fileName"]
    kb_id = params["kbId"]
    tmp_path = await asyncio.to_thread(minio_storage.download_to_temp, object_key)
    try:
        with open(tmp_path, "rb") as f:
            content = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    upload_file = UploadFile(file=io.BytesIO(content), filename=file_name)
    result = await fn(file=upload_file, kb_id=kb_id)
    if result.get("code") in (0, 200):
        d = result.get("data", {})
        return f"已上传文档「{file_name}」到知识库 {kb_id}，文档ID={d.get('docId', '')}，任务ID={d.get('taskId', '')}。入库异步进行中。"
    return f"上传失败：{result.get('message', '')}"


async def _list_documents(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import list_documents as fn
    result = await fn(kb_id=params.get("kbId", ""), db=db)
    d = result.get("data", {})
    return f"文档列表共 {d.get('total', 0)} 个"


async def _delete_document(params: dict, db: AsyncSession) -> str:
    from app.api.knowledge.rag import delete_document as fn
    result = await fn(doc_id=params["id"], db=db)
    return f"已删除文档 {params['id']}" if result.get("code") in (0, 200) else f"删除失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("rag_search", _rag_search)
    registry.register("list_knowledge", _list_knowledge)
    registry.register("get_knowledge_stats", _get_knowledge_stats)
    registry.register("get_knowledge_categories", _get_knowledge_categories)
    registry.register("upload_knowledge_file", _upload_knowledge_file)
    registry.register("create_knowledge_item", _create_knowledge_item)
    registry.register("update_knowledge_item", _update_knowledge_item)
    registry.register("delete_knowledge_item", _delete_knowledge_item)
    registry.register("recall_test", _recall_test)
    registry.register("list_knowledge_bases", _list_knowledge_bases)
    registry.register("create_knowledge_base", _create_knowledge_base)
    registry.register("update_knowledge_base", _update_knowledge_base)
    registry.register("delete_knowledge_base", _delete_knowledge_base)
    registry.register("upload_document", _upload_document)
    registry.register("list_documents", _list_documents)
    registry.register("delete_document", _delete_document)
