"""
Kuzu embedded graph database manager.

Blueprint alignment: Section 5.3, 8.5

Schema:
  Nodes: Entity(name PK, type, description, source_chunks, kb_id)
         Chunk(chunk_id PK, kb_id, doc_id)
  Edges: MENTIONS(Entity → Chunk)
         RELATED(Entity → Entity, relation_type)

All queries use $param parameterization for injection prevention.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Global singletons ──────────────────────────────────

_db = None
_conn = None
_lock = threading.Lock()
_schema_ready = False


def _get_conn():
    """Get or create Kuzu connection (thread-safe)."""
    global _db, _conn, _schema_ready

    if _conn is not None:
        return _conn

    with _lock:
        if _conn is not None:
            return _conn

        try:
            import kuzu

            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                settings.kuzu_data_dir,
            )
            os.makedirs(db_path, exist_ok=True)

            logger.info("初始化 Kuzu 图数据库: %s", db_path)
            _db = kuzu.Database(db_path)
            _conn = kuzu.Connection(_db)

            _ensure_schema()
            _schema_ready = True
            logger.info("Kuzu 图数据库就绪")

        except ImportError:
            logger.warning("kuzu 包未安装，图数据库功能不可用")
            return None
        except Exception as e:
            logger.error("Kuzu 初始化失败: %s", e)
            return None

    return _conn


def _ensure_schema():
    """Create node/edge tables if they don't exist."""
    conn = _conn
    if conn is None:
        return

    try:
        # Entity node table
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                name STRING,
                type STRING,
                description STRING,
                source_chunks STRING,
                kb_id STRING,
                PRIMARY KEY (name)
            )
        """)

        # Chunk node table
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                chunk_id INT64,
                kb_id STRING,
                doc_id STRING,
                PRIMARY KEY (chunk_id)
            )
        """)

        # MENTIONS edge: Entity → Chunk
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS MENTIONS (
                FROM Entity TO Chunk
            )
        """)

        # RELATED edge: Entity → Entity
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATED (
                FROM Entity TO Entity,
                relation_type STRING
            )
        """)

    except Exception as e:
        logger.warning("Kuzu schema 创建警告 (可能已存在): %s", e)


# ── Write Operations ───────────────────────────────────

def upsert_entity(
    name: str,
    entity_type: str = "",
    description: str = "",
    source_chunks: str = "",
    kb_id: str = "",
) -> bool:
    """Insert or update an entity node."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        conn.execute(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.description = $description,
                e.source_chunks = $source_chunks,
                e.kb_id = $kb_id
            """,
            {
                "name": name,
                "type": entity_type,
                "description": description,
                "source_chunks": source_chunks,
                "kb_id": kb_id,
            },
        )
        return True
    except Exception as e:
        logger.error("upsert_entity 失败: %s", e)
        return False


def upsert_chunk(chunk_id: int, kb_id: str = "", doc_id: str = "") -> bool:
    """Insert or update a chunk node."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        conn.execute(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.kb_id = $kb_id,
                c.doc_id = $doc_id
            """,
            {
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
            },
        )
        return True
    except Exception as e:
        logger.error("upsert_chunk 失败: %s", e)
        return False


def link_entity_to_chunk(entity_name: str, chunk_id: int) -> bool:
    """Create MENTIONS edge from entity to chunk."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        conn.execute(
            """
            MATCH (e:Entity {name: $entity_name})
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (e)-[:MENTIONS]->(c)
            """,
            {
                "entity_name": entity_name,
                "chunk_id": chunk_id,
            },
        )
        return True
    except Exception as e:
        logger.error("link_entity_to_chunk 失败: %s", e)
        return False


def upsert_relation(
    from_entity: str,
    to_entity: str,
    relation_type: str = "",
) -> bool:
    """Create or update a RELATED edge between two entities."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        conn.execute(
            """
            MATCH (a:Entity {name: $from_entity})
            MATCH (b:Entity {name: $to_entity})
            MERGE (a)-[r:RELATED]->(b)
            SET r.relation_type = $relation_type
            """,
            {
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relation_type": relation_type,
            },
        )
        return True
    except Exception as e:
        logger.error("upsert_relation 失败: %s", e)
        return False


# ── Read / Query Operations ────────────────────────────

def query_entities(
    query_terms: list[str],
    kb_id: str = "",
    top_k: int = 10,
) -> list[dict]:
    """Fuzzy match entities by name (CONTAINS).

    Args:
        query_terms: Terms to search in entity names.
        kb_id: Optional KB filter.
        top_k: Max results.

    Returns:
        List of {name, type, description, source_chunks, kb_id}.
    """
    conn = _get_conn()
    if conn is None:
        return []

    results = []
    seen = set()

    for term in query_terms:
        try:
            kb_filter = "e.kb_id = $kb_id AND " if kb_id else ""
            query_str = (
                f"MATCH (e:Entity) "
                f"WHERE {kb_filter} CONTAINS(LOWER(e.name), LOWER($term)) "
                f"RETURN e.name, e.type, e.description, e.source_chunks, e.kb_id "
                f"LIMIT {int(top_k)}"
            )
            params = {"term": term}
            if kb_id:
                params["kb_id"] = kb_id

            result = conn.execute(query_str, params)
            while result.has_next():
                row = result.get_next()
                name = row[0]
                if name not in seen:
                    seen.add(name)
                    results.append({
                        "name": name,
                        "type": row[1],
                        "description": row[2],
                        "source_chunks": row[3],
                        "kb_id": row[4],
                    })
        except Exception as e:
            logger.debug("query_entities 查询失败 (term=%s): %s", term, e)

    return results[:top_k]


def expand_from_entities(
    entity_names: list[str],
    hops: int = 1,
    kb_id: str = "",
) -> dict:
    """1-hop expansion from entities to connected chunks.

    Returns:
        {entity_names: [...], chunk_ids: [...]}
    """
    conn = _get_conn()
    if conn is None:
        return {"entity_names": entity_names, "chunk_ids": []}

    chunk_ids = set()

    for name in entity_names:
        try:
            kb_filter = "e.kb_id = $kb_id AND " if kb_id else ""
            query_str = (
                f"MATCH (e:Entity {{name: $name}})-[:MENTIONS]->(c:Chunk) "
                f"WHERE {kb_filter} TRUE "
                f"RETURN c.chunk_id"
            )
            params = {"name": name}
            if kb_id:
                params["kb_id"] = kb_id

            result = conn.execute(query_str, params)
            while result.has_next():
                row = result.get_next()
                chunk_ids.add(row[0])
        except Exception as e:
            logger.debug("expand_from_entities 失败 (name=%s): %s", name, e)

    return {
        "entity_names": entity_names,
        "chunk_ids": list(chunk_ids),
    }


def get_all_entities_for_kb(kb_id: str) -> list[dict]:
    """Get all entities in a knowledge base."""
    conn = _get_conn()
    if conn is None:
        return []

    entities = []
    try:
        result = conn.execute(
            "MATCH (e:Entity {kb_id: $kb_id}) RETURN e.name, e.type, e.description, e.source_chunks",
            {"kb_id": kb_id},
        )
        while result.has_next():
            row = result.get_next()
            entities.append({
                "name": row[0],
                "type": row[1],
                "description": row[2],
                "source_chunks": row[3],
            })
    except Exception as e:
        logger.error("get_all_entities_for_kb 失败: %s", e)

    return entities


def get_all_relations_for_kb(kb_id: str) -> list[dict]:
    """Get all relations (edges between entities) in a knowledge base."""
    conn = _get_conn()
    if conn is None:
        return []

    relations = []
    try:
        result = conn.execute(
            """
            MATCH (a:Entity {kb_id: $kb_id})-[r:RELATED]->(b:Entity {kb_id: $kb_id})
            RETURN a.name, b.name, r.relation_type
            """,
            {"kb_id": kb_id},
        )
        while result.has_next():
            row = result.get_next()
            relations.append({
                "from": row[0],
                "to": row[1],
                "relation_type": row[2],
            })
    except Exception as e:
        logger.error("get_all_relations_for_kb 失败: %s", e)

    return relations


def clear_kb(kb_id: str) -> bool:
    """Delete all entities and chunks belonging to a knowledge base."""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        conn.execute("MATCH (e:Entity {kb_id: $kb_id}) DETACH DELETE e", {"kb_id": kb_id})
        conn.execute("MATCH (c:Chunk {kb_id: $kb_id}) DELETE c", {"kb_id": kb_id})
        return True
    except Exception as e:
        logger.error("clear_kb 失败: %s", e)
        return False


def is_available() -> bool:
    """Check if Kuzu graph database is ready."""
    return _get_conn() is not None
