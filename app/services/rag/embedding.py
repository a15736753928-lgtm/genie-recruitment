"""
Embedding service — BGE-M3 ONNX INT8 dual-vector encoding.

Blueprint alignment: Section 6

Uses BGE-M3 (gpahal/bge-m3-onnx-int8) via ONNX Runtime for:
  - encode_dense(texts)  → list[list[float]]  (N × 1024, L2-normalized for COSINE)
  - encode_sparse(texts) → list[dict[int, float]]  (token_id → weight, for IP)

Delegates model loading to app.core.model_loader.

IMPORTANT: After upgrading from bge-small (512-dim), existing Milvus vectors
are incompatible. Drop the old collection and re-ingest all documents.
"""

from __future__ import annotations

from app.core.model_loader import (
    encode_dense as _encode_dense,
    encode_sparse as _encode_sparse,
    encode_query_dense as _encode_query_dense,
    encode_query_sparse as _encode_query_sparse,
    preload_models,
)


def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts into dense vectors (1024-dim, L2-norm).

    Alias for encode_dense — used by the ingestion pipeline.
    """
    return _encode_dense(texts)


def encode_query(text: str) -> list[float]:
    """Encode a single query text into a dense vector."""
    return _encode_query_dense(text)


# Re-export for convenience
encode_dense = _encode_dense
encode_sparse = _encode_sparse
encode_query_dense = _encode_query_dense
encode_query_sparse = _encode_query_sparse
