"""Milvus Lite manager — embedded vector database.

Blueprint alignment: Section 5.2 / 5.3
- Collection: single collection with dense_vector (1024-dim COSINE)
- Schema: id (auto), kb_id (varchar), dense_vector (float_vector)
- Text stored in PostgreSQL chunks table, linked via milvus_pk
- HNSW index: M=16, efConstruction=256, metric=COSINE

Uses milvus-lite 3.x (embedded, no server process needed).
"""

import re
import threading
import numpy as np
from typing import Optional

from pymilvus import MilvusClient
from app.config import get_settings

settings = get_settings()

# Validate kb_id format to prevent filter expression injection
# Format: kb_ followed by exactly 8 hex characters
_KB_ID_PATTERN = re.compile(r'^kb_[a-f0-9]{8}$')

_lock = threading.Lock()
_client: Optional[MilvusClient] = None
_collection_ready: bool = False


def _get_client() -> MilvusClient:
    """Get or create the MilvusClient (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = MilvusClient(settings.milvus_db_path)
    return _client


def ensure_collection() -> bool:
    """Ensure the collection exists with proper schema. Returns True if ready."""
    global _collection_ready
    if _collection_ready:
        return True

    with _lock:
        if _collection_ready:
            return True
        try:
            client = _get_client()
            coll_name = settings.milvus_collection_name

            if client.has_collection(coll_name):
                _collection_ready = True
                return True

            # Create collection with auto-ID and dense vector
            client.create_collection(
                collection_name=coll_name,
                dimension=settings.embedding_dim,
                metric_type="COSINE",
                auto_id=True,
                # enable dynamic field for kb_id
            )

            # Create HNSW index
            client.create_index(
                collection_name=coll_name,
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )

            _collection_ready = True
            return True
        except Exception as e:
            print(f"[Milvus] Failed to init collection: {e}")
            return False


def insert_vectors(
    vectors: list[list[float]],
    kb_ids: list[str],
) -> list[int]:
    """Insert dense vectors into Milvus. Returns list of auto-generated IDs.

    Args:
        vectors: List of dense vectors, each [dim] float list
        kb_ids: kb_id per vector for partition/filtering

    Returns:
        List of auto-generated integer IDs (milvus_pk)
    """
    if not vectors:
        return []

    if not ensure_collection():
        return []

    client = _get_client()
    coll_name = settings.milvus_collection_name

    data = []
    for vec, kb_id in zip(vectors, kb_ids):
        data.append({
            "vector": vec,
            "kb_id": kb_id,
        })

    try:
        result = client.insert(collection_name=coll_name, data=data)
        # result is a dict with 'ids' key (list of ints)
        ids = result.get("ids", [])
        return ids if isinstance(ids, list) else list(ids)
    except Exception as e:
        print(f"[Milvus] Insert error: {e}")
        return []


def search_dense(
    query_vector: list[float],
    top_k: int = 10,
    kb_id: Optional[str] = None,
) -> list[dict]:
    """Dense vector search (COSINE similarity).

    Args:
        query_vector: Normalized dense vector
        top_k: Number of results
        kb_id: Optional knowledge base filter

    Returns:
        List of {id: milvus_pk, distance: float, kb_id: str}
    """
    if not ensure_collection():
        return []

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        # Build filter expression for kb_id
        filter_expr = None
        if kb_id:
            # Validate kb_id format to prevent expression injection
            if not _KB_ID_PATTERN.match(kb_id):
                raise ValueError(f"Invalid kb_id format: {kb_id[:20]}...")
            filter_expr = f'kb_id == "{kb_id}"'

        results = client.search(
            collection_name=coll_name,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr,
            output_fields=["kb_id"],
            search_params={"ef": settings.search_ef if hasattr(settings, 'search_ef') else 64},
        )

        # results is list[list[dict]] — one inner list per query
        if not results or not results[0]:
            return []

        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit["id"],          # milvus auto-id
                "distance": hit["distance"],
                "kb_id": hit.get("entity", {}).get("kb_id", ""),
            })
        return hits

    except Exception as e:
        print(f"[Milvus] Search error: {e}")
        return []


def delete_by_ids(ids: list[int]) -> bool:
    """Delete vectors by their Milvus primary keys."""
    if not ids or not ensure_collection():
        return False

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        # Build expression: id in [1, 2, 3, ...]
        id_list = ", ".join(str(i) for i in ids)
        client.delete(collection_name=coll_name, filter=f"id in [{id_list}]")
        return True
    except Exception as e:
        print(f"[Milvus] Delete error: {e}")
        return False


def get_collection_stats() -> dict:
    """Get collection statistics."""
    if not ensure_collection():
        return {"num_entities": 0}

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        stats = client.get_collection_stats(coll_name)
        return stats
    except Exception as e:
        print(f"[Milvus] Stats error: {e}")
        return {"num_entities": 0}
