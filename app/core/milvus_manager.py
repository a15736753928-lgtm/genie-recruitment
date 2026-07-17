"""
Milvus Lite manager — dual-vector embedded database.

Blueprint alignment: Section 5.2
- Dense vector: 1024-dim FLOAT_VECTOR, HNSW COSINE index
- Sparse vector: SPARSE_FLOAT_VECTOR, SPARSE_INVERTED_INDEX IP
- Both in a single collection with kb_id filter field
- Text stored in PostgreSQL chunks table, linked via milvus_pk

IMPORTANT: After upgrading from the old 512-dim single-vector schema,
existing collection must be dropped and documents re-ingested.
"""

import re
import threading
import numpy as np
from typing import Optional

from pymilvus import MilvusClient, DataType
from app.config import get_settings

settings = get_settings()

_KB_ID_PATTERN = re.compile(r'^kb_[a-f0-9]{8}$')

_lock = threading.Lock()
_client: Optional[MilvusClient] = None
_collection_ready: bool = False


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = MilvusClient(settings.milvus_db_path)
    return _client


def ensure_collection() -> bool:
    """Create dual-vector collection if not exists. Returns True if ready."""
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

            # Create schema with dual vector fields
            schema = client.create_schema(
                auto_id=True,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            )
            schema.add_field(
                field_name="kb_id",
                datatype=DataType.VARCHAR,
                max_length=128,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=settings.embedding_dim,  # 1024
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            # Create collection
            client.create_collection(
                collection_name=coll_name,
                schema=schema,
            )

            # Dense HNSW index (COSINE)
            client.create_index(
                collection_name=coll_name,
                field_name="dense_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )

            # Sparse inverted index (IP)
            client.create_index(
                collection_name=coll_name,
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            _collection_ready = True
            return True

        except Exception as e:
            print(f"[Milvus] Failed to init collection: {e}")
            return False


def insert_vectors(
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
    kb_ids: list[str],
) -> list[int]:
    """Insert dual vectors (dense + sparse) into Milvus.

    Args:
        dense_vectors: List of 1024-dim float lists
        sparse_vectors: List of {token_id: weight} dicts
        kb_ids: kb_id per vector

    Returns:
        List of auto-generated integer IDs (milvus_pk)
    """
    if not dense_vectors:
        return []

    if not ensure_collection():
        return []

    client = _get_client()
    coll_name = settings.milvus_collection_name

    data = []
    for d_vec, s_vec, kb_id in zip(dense_vectors, sparse_vectors, kb_ids):
        data.append({
            "dense_vector": d_vec,
            "sparse_vector": s_vec,
            "kb_id": kb_id,
        })

    try:
        result = client.insert(collection_name=coll_name, data=data)
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

    Returns:
        [{id: milvus_pk, distance: float, kb_id: str}, ...]
    """
    if not ensure_collection():
        return []

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        filter_expr = None
        if kb_id:
            if not _KB_ID_PATTERN.match(kb_id):
                raise ValueError(f"Invalid kb_id format: {kb_id[:20]}...")
            filter_expr = f'kb_id == "{kb_id}"'

        results = client.search(
            collection_name=coll_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["kb_id"],
            search_params={"ef": settings.search_ef},
        )

        if not results or not results[0]:
            return []

        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit["id"],
                "distance": hit["distance"],
                "kb_id": hit.get("entity", {}).get("kb_id", ""),
            })
        return hits

    except Exception as e:
        print(f"[Milvus] Dense search error: {e}")
        return []


def search_sparse(
    sparse_vector: dict[int, float],
    top_k: int = 10,
    kb_id: Optional[str] = None,
) -> list[dict]:
    """Sparse vector search (IP — inner product).

    Args:
        sparse_vector: {token_id: weight} dict
        top_k: Number of results
        kb_id: Optional KB filter

    Returns:
        [{id: milvus_pk, distance: float, kb_id: str}, ...]
    """
    if not ensure_collection():
        return []

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        filter_expr = None
        if kb_id:
            if not _KB_ID_PATTERN.match(kb_id):
                raise ValueError(f"Invalid kb_id format: {kb_id[:20]}...")
            filter_expr = f'kb_id == "{kb_id}"'

        results = client.search(
            collection_name=coll_name,
            data=[sparse_vector],
            anns_field="sparse_vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["kb_id"],
        )

        if not results or not results[0]:
            return []

        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit["id"],
                "distance": hit["distance"],
                "kb_id": hit.get("entity", {}).get("kb_id", ""),
            })
        return hits

    except Exception as e:
        print(f"[Milvus] Sparse search error: {e}")
        return []


def delete_by_ids(ids: list[int]) -> bool:
    """Delete vectors by Milvus primary keys."""
    if not ids or not ensure_collection():
        return False

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        id_list = ", ".join(str(i) for i in ids)
        client.delete(collection_name=coll_name, filter=f"id in [{id_list}]")
        return True
    except Exception as e:
        print(f"[Milvus] Delete error: {e}")
        return False


def get_collection_stats() -> dict:
    """Get collection statistics."""
    if not ensure_collection():
        return {"row_count": 0}

    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        stats = client.get_collection_stats(coll_name)
        return stats
    except Exception as e:
        print(f"[Milvus] Stats error: {e}")
        return {"row_count": 0}


def drop_collection() -> bool:
    """Drop and recreate the collection (for schema migration)."""
    global _collection_ready
    client = _get_client()
    coll_name = settings.milvus_collection_name

    try:
        if client.has_collection(coll_name):
            client.drop_collection(coll_name)
        _collection_ready = False
        return ensure_collection()
    except Exception as e:
        print(f"[Milvus] Drop error: {e}")
        return False
