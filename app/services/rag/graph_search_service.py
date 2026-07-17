"""
Graph-based retrieval service — LLM entity extraction + Kuzu fuzzy match + 1-hop.

Blueprint alignment: Section 9.2.3

Pipeline:
  1. LLM extract query entities
  2. Kuzu fuzzy match entities (CONTAINS)
  3. 1-hop expand (Entity → MENTIONS → Chunk)
  4. Score by entity overlap rate
  5. Attach community summaries
"""

from __future__ import annotations

import json
import logging
from typing import Optional
from openai import AsyncOpenAI
from app.config import get_settings
from app.database import async_session_factory
from app.models.knowledge import GraphCommunity
from app.core.graph_store import (
    query_entities,
    expand_from_entities,
    is_available as kuzu_available,
)
from sqlalchemy import select

settings = get_settings()
logger = logging.getLogger(__name__)

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

_ENTITY_EXTRACT_SYSTEM = """你是一个查询分析专家。从用户问题中提取关键概念/术语/实体名称。

要求：
1. 只提取问题中明确提到的具体概念
2. 每个提取结果是一个简短的实体名称（不超过10个字）
3. 不要提取过于宽泛的词（如"系统""方法""技术"）
4. 返回 JSON: {"entities": ["实体1", "实体2", ...]}"""


async def _extract_entities_from_query(query: str) -> list[str]:
    """Use LLM to extract key entities from a search query."""
    try:
        response = await llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": _ENTITY_EXTRACT_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from code fence
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                data = json.loads(match.group(0))
            else:
                return []

        return data.get("entities", [])

    except Exception as e:
        logger.debug("查询实体提取失败: %s", e)
        return []


async def graph_search(
    query: str,
    kb_ids: Optional[list[str]] = None,
    top_k: int = 10,
) -> dict:
    """Graph-based retrieval: entity extraction + fuzzy match + 1-hop expansion.

    Args:
        query: Search query.
        kb_ids: Optional KB ID filter list.
        top_k: Max results.

    Returns:
        {chunk_hits: [{id, score, kb_id}], community_summaries: [...]}
    """
    if not kuzu_available():
        return {"chunk_hits": [], "community_summaries": []}

    # 1. LLM extract query entities
    query_entities = await _extract_entities_from_query(query)
    if not query_entities:
        return {"chunk_hits": [], "community_summaries": []}

    # 2. For each KB, fuzzy match + expand
    kb_list = kb_ids if kb_ids else [""]
    chunk_entity_hits: dict[int, int] = {}
    chunk_kb_map: dict[int, str] = {}

    for kb_id in kb_list:
        kb = kb_id if kb_id else ""

        # Fuzzy match entities
        matched = query_entities(matched_names, kb_id=kb) if kb else query_entities(matched_names)
        matched_names = [e["name"] for e in matched]

        # 1-hop expand
        for name in matched_names:
            expanded = expand_from_entities([name], hops=1, kb_id=kb)
            for cid in expanded.get("chunk_ids", []):
                chunk_entity_hits[cid] = chunk_entity_hits.get(cid, 0) + 1
                if cid not in chunk_kb_map:
                    chunk_kb_map[cid] = kb

    if not chunk_entity_hits:
        return {"chunk_hits": [], "community_summaries": []}

    # 3. Score by entity overlap rate
    total_query_entities = len(query_entities)
    chunk_hits = []
    for chunk_id, hit_count in chunk_entity_hits.items():
        score = min(hit_count / total_query_entities, 1.0)
        chunk_hits.append({
            "id": chunk_id,
            "distance": score,
            "kb_id": chunk_kb_map.get(chunk_id, ""),
            "hit_count": hit_count,
        })

    chunk_hits.sort(key=lambda x: x["distance"], reverse=True)
    chunk_hits = chunk_hits[:top_k]

    # 4. Find related community summaries
    community_summaries = await _get_communities_by_chunk_pks(
        [h["id"] for h in chunk_hits]
    )

    return {
        "chunk_hits": chunk_hits,
        "community_summaries": community_summaries,
    }


async def _get_communities_by_chunk_pks(pks: list[int]) -> list[dict]:
    """Find communities that contain these chunk IDs."""
    if not pks:
        return []

    async with async_session_factory() as db:
        try:
            result = await db.execute(
                select(GraphCommunity)
            )
            communities = result.scalars().all()

            summaries = []
            for com in communities:
                try:
                    com_chunk_ids = json.loads(com.chunk_ids) if com.chunk_ids else []
                except (json.JSONDecodeError, TypeError):
                    com_chunk_ids = []

                # Check overlap with query chunks
                overlap = set(com_chunk_ids) & set(pks)
                if overlap:
                    summaries.append({
                        "id": com.id,
                        "name": com.name,
                        "summary": com.summary,
                        "overlap_count": len(overlap),
                    })

            summaries.sort(key=lambda x: x["overlap_count"], reverse=True)
            return summaries[:5]  # Top 5 communities

        except Exception as e:
            logger.debug("社区查询失败: %s", e)
            return []
