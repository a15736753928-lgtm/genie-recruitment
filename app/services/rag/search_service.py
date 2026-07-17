"""
Search/retrieval pipeline — three-way hybrid (Dense + Sparse + Graph) + RRF fusion.

Blueprint alignment: Section 9

Pipeline:
  query → normalize → [dense recall + sparse recall + graph recall]
       → RRF three-way fusion → PG enrich → empty fallback
       → rerank → highlight → community attach → respond
"""

from __future__ import annotations

import json
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import async_session_factory
from app.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeBase, GraphCommunity
from app.services.rag.embedding import encode_query_dense, encode_query_sparse
from app.services.rag.text_processor import (
    clean_text,
    normalize_query,
    extract_highlight_terms,
    highlighter,
)
from app.services.rag.reranker import rerank
from app.core.milvus_manager import search_dense, search_sparse

settings = get_settings()


async def search(
    query: str,
    kb_ids: Optional[list[str]] = None,
    top_k: Optional[int] = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Main three-way hybrid search pipeline.

    Args:
        query: Search query text.
        kb_ids: Optional list of KB IDs (None = all).
        top_k: Max results (default from settings).
        min_similarity: Minimum similarity threshold.

    Returns:
        List of result dicts with id, content, file_name, kb_id, kb_name,
        chunk_index, similarity, highlights, community_id, community_name, match_type.
    """
    top_k = top_k or settings.default_top_k
    rerank_top_k = settings.rerank_top_k

    # 1. Normalize query
    query_normalized = normalize_query(clean_text(query))

    # 2. Dense recall
    query_dense = encode_query_dense(query_normalized)
    all_dense = []
    if kb_ids:
        for kb_id in kb_ids:
            hits = search_dense(query_dense, rerank_top_k, kb_id=kb_id)
            for h in hits:
                h["source"] = "dense"
            all_dense.extend(hits)
    else:
        hits = search_dense(query_dense, rerank_top_k)
        for h in hits:
            h["source"] = "dense"
        all_dense = hits

    # 3. Sparse recall
    all_sparse = []
    if settings.sparse_vector_enabled:
        try:
            query_sparse = encode_query_sparse(query_normalized)
            if query_sparse:
                if kb_ids:
                    for kb_id in kb_ids:
                        hits = search_sparse(query_sparse, rerank_top_k, kb_id=kb_id)
                        for h in hits:
                            h["source"] = "sparse"
                        all_sparse.extend(hits)
                else:
                    hits = search_sparse(query_sparse, rerank_top_k)
                    for h in hits:
                        h["source"] = "sparse"
                    all_sparse = hits
        except Exception:
            pass

    # 4. Graph recall (async)
    all_graph = []
    if settings.kuzu_enabled:
        try:
            from app.services.rag.graph_search_service import graph_search
            graph_result = await graph_search(query, kb_ids, rerank_top_k)
            for h in graph_result.get("chunk_hits", []):
                h["source"] = "graph"
            all_graph = graph_result.get("chunk_hits", [])
        except Exception:
            pass

    # 5. RRF three-way fusion
    fused = _rrf_fuse(all_dense, all_sparse, all_graph, top_k=rerank_top_k)

    # 6. Empty result fallback — expand dense recall
    if not fused:
        fallback_k = min(rerank_top_k * 2, settings.max_search_recall)
        fallback = []
        if kb_ids:
            for kb_id in kb_ids:
                hits = search_dense(query_dense, fallback_k, kb_id=kb_id)
                for h in hits:
                    h["source"] = "dense_fallback"
                fallback.extend(hits)
        else:
            hits = search_dense(query_dense, fallback_k)
            for h in hits:
                h["source"] = "dense_fallback"
            fallback = hits
        fused = _rrf_fuse(fallback, [], [], top_k=rerank_top_k)

    if not fused:
        return []

    # 7. Enrich from PostgreSQL
    enriched = await _enrich_results(fused)

    # 8. Rerank
    if settings.rerank_enabled and len(enriched) > 1:
        texts = [r["content"] for r in enriched]
        ranked = rerank(query_normalized, texts, top_k=top_k)
        reranked = []
        for idx, score in ranked:
            item = enriched[idx].copy()
            item["similarity"] = round(score, 4)
            item["match_type"] = "reranked"
            reranked.append(item)
        enriched = reranked
    else:
        enriched.sort(key=lambda r: r["similarity"], reverse=True)

    # 9. Filter by min similarity
    enriched = [r for r in enriched if r["similarity"] >= min_similarity]

    # 10. Limit to top_k
    enriched = enriched[:top_k]

    # 11. Extract highlights
    for r in enriched:
        r["highlights"] = extract_highlight_terms(query_normalized, r["content"])

    # 12. Attach community info
    await _attach_community_info(enriched)

    return enriched


# ── RRF Fusion ─────────────────────────────────────────

def _rrf_fuse(
    dense: list[dict],
    sparse: list[dict],
    graph: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Reciprocal Rank Fusion — blueprint Section 9.3.

    score(chunk) = dense_w / (k + rank_dense + 1)
                 + sparse_w / (k + rank_sparse + 1)
                 + graph_w / (k + rank_graph + 1)

    Returns:
        List of {id, distance: rrf_score, kb_id, sources: [...], match_type}
    """
    k = settings.rrf_k
    w_dense = settings.rrf_dense_weight
    w_sparse = settings.rrf_sparse_weight
    w_graph = settings.rrf_graph_weight

    fusion_map: dict[int, dict] = {}

    for rank, hit in enumerate(dense):
        pk = hit["id"]
        if pk not in fusion_map:
            fusion_map[pk] = {"id": pk, "kb_id": hit.get("kb_id", ""), "score": 0.0, "sources": [], "distance": hit.get("distance", 0.0)}
        fusion_map[pk]["score"] += w_dense / (k + rank + 1)
        fusion_map[pk]["sources"].append("dense")

    for rank, hit in enumerate(sparse):
        pk = hit["id"]
        if pk not in fusion_map:
            fusion_map[pk] = {"id": pk, "kb_id": hit.get("kb_id", ""), "score": 0.0, "sources": [], "distance": hit.get("distance", 0.0)}
        fusion_map[pk]["score"] += w_sparse / (k + rank + 1)
        fusion_map[pk]["sources"].append("sparse")

    for rank, hit in enumerate(graph):
        pk = hit["id"]
        if pk not in fusion_map:
            fusion_map[pk] = {"id": pk, "kb_id": hit.get("kb_id", ""), "score": 0.0, "sources": [], "distance": hit.get("distance", 0.0)}
        graph_weight = w_graph * hit.get("distance", 0.25)  # graph uses overlap rate
        fusion_map[pk]["score"] += graph_weight / (k + rank + 1)
        fusion_map[pk]["sources"].append("graph")

    results = list(fusion_map.values())
    for r in results:
        r["distance"] = r["score"]  # RRF score becomes the distance
        r["match_type"] = "+".join(r.get("sources", []))

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── PG Enrichment ──────────────────────────────────────

async def _enrich_results(combined: list[dict]) -> list[dict]:
    """Join Milvus results with PG chunk/document/KB metadata."""
    if not combined:
        return []

    milvus_pks = [h["id"] for h in combined]

    async with async_session_factory() as db:
        chunk_result = await db.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.milvus_pk.in_(milvus_pks))
        )
        chunks = {c.milvus_pk: c for c in chunk_result.scalars().all()}

        doc_ids = list({c.doc_id for c in chunks.values()})
        doc_result = await db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id.in_(doc_ids))
        )
        docs = {d.id: d for d in doc_result.scalars().all()}

        kb_ids_set = list({c.kb_id for c in chunks.values()})
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids_set))
        )
        kbs = {k.id: k for k in kb_result.scalars().all()}

    enriched = []
    for hit in combined:
        chunk = chunks.get(hit["id"])
        if not chunk:
            continue

        doc = docs.get(chunk.doc_id)
        kb = kbs.get(chunk.kb_id)

        enriched.append({
            "id": str(hit["id"]),
            "content": chunk.chunk_text,
            "file_name": doc.file_name if doc else "",
            "kb_id": chunk.kb_id,
            "kb_name": kb.name if kb else "",
            "chunk_index": chunk.chunk_index,
            "similarity": round(hit.get("distance", 0.0), 4),
            "match_type": hit.get("match_type", "vector"),
        })

    return enriched


# ── Community Attachment ───────────────────────────────

async def _attach_community_info(results: list[dict]) -> None:
    """Attach community_id and community_name to results."""
    if not results or not settings.community_enabled:
        return

    pks = [int(r["id"]) for r in results if r["id"].isdigit()]
    if not pks:
        return

    try:
        async with async_session_factory() as db:
            com_result = await db.execute(select(GraphCommunity))
            communities = com_result.scalars().all()

            for r in results:
                pk = int(r["id"]) if r["id"].isdigit() else 0
                for com in communities:
                    try:
                        com_chunk_ids = json.loads(com.chunk_ids) if com.chunk_ids else []
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if pk in com_chunk_ids:
                        r["community_id"] = com.id
                        r["community_name"] = com.name
                        break
    except Exception:
        pass


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
