"""Search/retrieval pipeline.

Pipeline:
  query → normalize → embed → dense search → enrich → rerank → highlight → respond

Improvements over original:
  - Query normalization matching document cleaning pipeline
  - CrossEncoder reranking for precision (BGE-Reranker)
  - Empty-result fallback with expanded search
  - HTML-safe highlighting
"""

from __future__ import annotations

from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import async_session_factory
from app.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeBase
from app.services.rag.embedding import encode_query
from app.services.rag.text_processor import (
    clean_text,
    normalize_query,
    extract_highlight_terms,
    highlighter,
)
from app.services.rag.reranker import rerank
from app.core.milvus_manager import search_dense

settings = get_settings()


async def search(
    query: str,
    kb_ids: Optional[list[str]] = None,
    top_k: Optional[int] = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Main search pipeline.

    Args:
        query: Search query text
        kb_ids: Optional list of KB IDs to search in (None = all)
        top_k: Max results to return (default from settings)
        min_similarity: Minimum similarity threshold (0.0-1.0)

    Returns:
        List of result dicts:
        {
            id: milvus_pk (str),
            content: chunk_text,
            file_name: source document name,
            kb_id: knowledge base id,
            kb_name: knowledge base name,
            chunk_index: int,
            similarity: float,
            highlights: [str, ...],
        }
    """
    top_k = top_k or settings.default_top_k
    rerank_top_k = settings.rerank_top_k

    # 1. Normalize query (match document cleaning pipeline)
    query_normalized = normalize_query(clean_text(query))

    # 2. Embed query
    query_vec = encode_query(query_normalized)

    # 3. Dense search — fetch more candidates for reranking
    all_hits = []
    if kb_ids:
        for kb_id in kb_ids:
            hits = search_dense(query_vec, rerank_top_k, kb_id=kb_id)
            all_hits.extend(hits)
    else:
        all_hits = search_dense(query_vec, rerank_top_k)

    # 4. Empty result fallback — expand search
    if not all_hits:
        fallback_k = min(rerank_top_k * 2, settings.max_search_recall)
        if kb_ids:
            for kb_id in kb_ids:
                hits = search_dense(query_vec, fallback_k, kb_id=kb_id)
                all_hits.extend(hits)
        else:
            all_hits = search_dense(query_vec, fallback_k)

    if not all_hits:
        return []

    # 5. Enrich from PostgreSQL
    enriched = await _enrich_results(all_hits)

    # 6. Rerank with CrossEncoder for precision
    if settings.rerank_enabled and len(enriched) > 1:
        texts = [r["content"] for r in enriched]
        ranked = rerank(query_normalized, texts, top_k=top_k)
        # Reorder enriched by reranker scores
        reranked = []
        for idx, score in ranked:
            item = enriched[idx].copy()
            item["similarity"] = round(score, 4)
            item["match_type"] = "reranked"
            reranked.append(item)
        enriched = reranked
    else:
        # Sort by vector similarity descending
        enriched.sort(key=lambda r: r["similarity"], reverse=True)

    # 7. Filter by min similarity
    enriched = [r for r in enriched if r["similarity"] >= min_similarity]

    # 8. Limit to top_k
    enriched = enriched[:top_k]

    # 9. Extract highlights
    for r in enriched:
        r["highlights"] = extract_highlight_terms(query_normalized, r["content"])

    return enriched


async def _enrich_results(milvus_hits: list[dict]) -> list[dict]:
    """Join Milvus results with PG chunk/document/KB metadata."""
    if not milvus_hits:
        return []

    milvus_pks = [h["id"] for h in milvus_hits]

    async with async_session_factory() as db:
        # Fetch chunks
        chunk_result = await db.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.milvus_pk.in_(milvus_pks))
        )
        chunks = {c.milvus_pk: c for c in chunk_result.scalars().all()}

        # Fetch documents
        doc_ids = list({c.doc_id for c in chunks.values()})
        doc_result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id.in_(doc_ids))
        )
        docs = {d.id: d for d in doc_result.scalars().all()}

        # Fetch KBs
        kb_ids_set = list({c.kb_id for c in chunks.values()})
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids_set))
        )
        kbs = {k.id: k for k in kb_result.scalars().all()}

    enriched = []
    for hit in milvus_hits:
        chunk = chunks.get(hit["id"])
        if not chunk:
            continue

        doc = docs.get(chunk.doc_id)
        kb = kbs.get(chunk.kb_id)

        enriched.append(
            {
                "id": str(hit["id"]),
                "content": chunk.chunk_text,
                "file_name": doc.file_name if doc else "",
                "kb_id": chunk.kb_id,
                "kb_name": kb.name if kb else "",
                "chunk_index": chunk.chunk_index,
                "similarity": round(hit["distance"], 4),
                "match_type": "vector",
            }
        )

    return enriched


async def get_chunk_content(milvus_pk: int) -> Optional[dict]:
    """Get a single chunk's full text."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.milvus_pk == milvus_pk)
        )
        chunk = result.scalar_one_or_none()
        if not chunk:
            return None

        doc_result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == chunk.doc_id)
        )
        doc = doc_result.scalar_one_or_none()

        return {
            "id": str(chunk.milvus_pk),
            "content": chunk.chunk_text,
            "file_name": doc.file_name if doc else "",
            "chunk_index": chunk.chunk_index,
            "kb_id": chunk.kb_id,
        }
