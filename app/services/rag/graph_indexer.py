"""
Knowledge graph indexer — LLM entity/relation extraction → Kuzu.

Blueprint alignment: Section 8.2, 8.3

Pipeline:
  chunks → [LLM entity extraction] → Kuzu Entity nodes + MENTIONS edges
                                   → [LLM relation extraction] → Kuzu RELATED edges
"""

from __future__ import annotations

import json
import logging
from openai import OpenAI
from app.config import get_settings
from app.core.graph_store import (
    upsert_entity,
    upsert_chunk,
    link_entity_to_chunk,
    upsert_relation,
    is_available as kuzu_available,
)

settings = get_settings()
logger = logging.getLogger(__name__)

llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
)

_EXTRACT_ENTITIES_SYSTEM = """你是一个知识图谱实体抽取专家。从给定的文本片段中提取关键实体。

要求：
1. 每个实体包含: name（名称）、type（类型）、description（一句话描述）
2. 只提取文本中明确提到的实体，不要推测
3. 实体名称要简洁规范（不超过20字）
4. 忽略过于宽泛或没有信息量的词
5. 返回 JSON: {"entities": [{"name": "...", "type": "...", "description": "..."}, ...]}"""

_EXTRACT_RELATIONS_SYSTEM = """你是一个知识图谱关系抽取专家。已知实体列表和文本上下文，判断哪些实体之间存在语义关系。

要求：
1. 关系方向: from → to
2. relation_type 使用简洁动词短语（"使用""属于""调用""依赖""基于""部署在""包含"）
3. 只输出存在明确语义关系且在同一上下文中被同时讨论的实体对
4. 返回 JSON: {"relations": [{"from": "...", "to": "...", "type": "..."}, ...]}"""


def index_document_chunks(
    doc_id: str,
    kb_id: str,
    chunks: list[str],
    milvus_ids: list[int],
    max_chunks_per_batch: int = 20,
    max_chunk_chars: int = 800,
):
    """Build graph index for a document's chunks.

    Args:
        doc_id: Document ID.
        kb_id: Knowledge base ID.
        chunks: List of chunk texts.
        milvus_ids: Corresponding Milvus primary keys.
    """
    if not kuzu_available():
        logger.info("Kuzu 不可用，跳过图谱索引")
        return

    # Register chunk nodes in Kuzu
    for chunk_id, text in zip(milvus_ids, chunks):
        upsert_chunk(chunk_id=int(chunk_id), kb_id=kb_id, doc_id=doc_id)

    # Extract entities in batches
    all_entities = []
    for batch_start in range(0, len(chunks), max_chunks_per_batch):
        batch = chunks[batch_start:batch_start + max_chunks_per_batch]
        batch_ids = milvus_ids[batch_start:batch_start + max_chunks_per_batch]

        entities = _extract_entities_batch(batch, batch_ids)
        all_entities.extend(entities)

    if not all_entities:
        return

    # Write entities + MENTIONS edges
    for ent in all_entities:
        upsert_entity(
            name=ent["name"],
            entity_type=ent.get("type", ""),
            description=ent.get("description", ""),
            source_chunks=json.dumps(ent.get("source_chunk_ids", [])),
            kb_id=kb_id,
        )
        for chunk_id in ent.get("source_chunk_ids", []):
            link_entity_to_chunk(ent["name"], int(chunk_id))

    # Extract relations
    relations = _extract_relations(all_entities, chunks[:max_chunks_per_batch])
    for rel in relations:
        upsert_relation(
            from_entity=rel["from"],
            to_entity=rel["to"],
            relation_type=rel.get("type", ""),
        )

    logger.info(
        "图谱索引完成: doc_id=%s entities=%d relations=%d",
        doc_id, len(all_entities), len(relations),
    )


def _extract_entities_batch(chunks: list[str], chunk_ids: list[int]) -> list[dict]:
    """Extract entities from a batch of chunks using LLM."""
    # Build context: truncated chunk texts with indices
    context_lines = []
    for i, text in enumerate(chunks):
        truncated = text[:max_chunk_chars] if len(text) > settings.chunk_size else text
        context_lines.append(f"[片段{i}] {truncated}")

    context = "\n\n---\n\n".join(context_lines)
    prompt = f"以下文本片段来自同一份文档，请提取其中的关键实体：\n\n{context}"

    try:
        response = llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": _EXTRACT_ENTITIES_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON
        data = _parse_json_response(content)
        entities = data.get("entities", [])

        # If response uses "片段N" as key
        if not entities:
            entities = _parse_fragmented_response(data, chunks)

        # Attach source chunk IDs
        for ent in entities:
            ent.setdefault("source_chunk_ids", chunk_ids)

        return entities

    except Exception as e:
        logger.warning("实体抽取失败: %s", e)
        return []


def _extract_relations(entities: list[dict], context_chunks: list[str]) -> list[dict]:
    """Extract relationships between extracted entities."""
    if len(entities) < 2:
        return []

    entity_list = [f"- {e['name']} ({e.get('type', '')}): {e.get('description', '')}" for e in entities[:50]]
    context = "\n".join(context_chunks[:3])[:3000]

    prompt = f"已知实体：\n" + "\n".join(entity_list) + f"\n\n文本上下文：\n{context}"

    try:
        response = llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": _EXTRACT_RELATIONS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        content = response.choices[0].message.content.strip()
        data = _parse_json_response(content)
        return data.get("relations", [])

    except Exception as e:
        logger.warning("关系抽取失败: %s", e)
        return []


def _parse_json_response(content: str) -> dict:
    """Parse potentially malformed JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from code fences
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding outermost {}
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _parse_fragmented_response(data: dict, chunks: list[str]) -> list[dict]:
    """Parse responses where entities are keyed by chunk index."""
    entities = []
    for key in data:
        if key.startswith("片段") or key.isdigit():
            chunk_entities = data[key]
            if isinstance(chunk_entities, list):
                entities.extend(chunk_entities)
    return entities
