"""
Community detection via Louvain algorithm + LLM summaries.

Blueprint alignment: Section 8.4

Pipeline:
  1. Load entities + relations from Kuzu
  2. Build networkx undirected graph
  3. Louvain community detection (seed=42)
  4. LLM summary per community (150-250 chars)
  5. Write communities to PostgreSQL
"""

from __future__ import annotations

import json
import logging
from app.config import get_settings
from app.database import get_sync_db
from app.models.knowledge import GraphCommunity, _now_ms, _short_uuid
from app.infrastructure.graph_store import (
    get_all_entities_for_kb,
    get_all_relations_for_kb,
    is_available as kuzu_available,
)
from app.services.ai import get_sync_llm_client

settings = get_settings()
logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """你是一个技术文档摘要专家。请根据一组相关实体的信息，生成该知识社区的自然语言摘要。

要求：
1. 摘要长度 150-250 字
2. 概括该社区涵盖的主要主题、关键概念和技术
3. 使用简洁专业的中文
4. 不要编造实体列表中不存在的信息"""


def build_communities(kb_id: str) -> list[dict]:
    """Run Louvain community detection on a KB's knowledge graph.

    Args:
        kb_id: Knowledge base ID.

    Returns:
        List of community dicts with id, name, summary, entity_ids, chunk_ids.
    """
    if not kuzu_available():
        logger.warning("Kuzu 不可用，无法执行社区发现")
        return []

    # 1. Load graph data from Kuzu
    entities = get_all_entities_for_kb(kb_id)
    relations = get_all_relations_for_kb(kb_id)

    if not entities:
        logger.info("KB %s 无实体，跳过社区发现", kb_id)
        return []

    # 2. Build networkx graph
    try:
        import networkx as nx

        G = nx.Graph()
        for e in entities:
            G.add_node(e["name"], **e)

        for r in relations:
            G.add_edge(r["from"], r["to"], relation_type=r.get("relation_type", ""))

        # 3. Louvain community detection
        try:
            raw_communities = nx.community.louvain_communities(G, seed=42)
        except Exception:
            # Fallback to connected components
            raw_communities = list(nx.connected_components(G))

        if not raw_communities:
            raw_communities = [{e["name"]} for e in entities]

        # 4. Build community objects + LLM summaries
        communities = []
        for i, member_names in enumerate(raw_communities):
            if len(member_names) < 2:
                continue

            com_entities = [e for e in entities if e["name"] in member_names]
            chunk_ids = set()
            for e in com_entities:
                src = e.get("source_chunks", "")
                if src:
                    try:
                        chunk_ids.update(json.loads(src) if isinstance(src, str) else src)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # LLM summary
            entity_descriptions = "\n".join(
                f"- {e['name']} ({e.get('type', '')}): {e.get('description', '')}"
                for e in com_entities[:20]
            )
            summary = _generate_summary(entity_descriptions)

            # Extract community name from first line of summary
            name = summary.split("。")[0][:30] if summary else f"社区{i+1}"

            communities.append({
                "id": _short_uuid("com"),
                "kb_id": kb_id,
                "name": name,
                "summary": summary,
                "entity_ids": json.dumps(list(member_names), ensure_ascii=False),
                "chunk_ids": json.dumps(list(chunk_ids), ensure_ascii=False),
                "created_at": _now_ms(),
            })

        # 5. Write to PostgreSQL
        db = get_sync_db()
        try:
            # Clear old communities for this KB
            from sqlalchemy import delete
            db.execute(delete(GraphCommunity).where(GraphCommunity.kb_id == kb_id))

            for com in communities:
                db.add(GraphCommunity(
                    id=com["id"],
                    kb_id=com["kb_id"],
                    name=com["name"],
                    summary=com["summary"],
                    entity_ids=com["entity_ids"],
                    chunk_ids=com["chunk_ids"],
                    created_at=com["created_at"],
                ))
            db.commit()
            logger.info("社区发现完成: kb_id=%s communities=%d", kb_id, len(communities))
        except Exception as e:
            db.rollback()
            logger.error("社区写入失败: %s", e)
        finally:
            db.close()

        return communities

    except ImportError:
        logger.warning("networkx 未安装，无法执行社区发现")
        return []
    except Exception as e:
        logger.error("社区发现失败: %s", e)
        return []


def _generate_summary(entity_descriptions: str) -> str:
    """Generate a community summary via LLM."""
    try:
        response = get_sync_llm_client().chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": f"实体列表：\n{entity_descriptions}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("社区摘要生成失败: %s", e)
        return ""
