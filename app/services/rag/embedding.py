"""Embedding service — sentence-transformers with BGE model.

Replaces the old hash-based feature hashing with real semantic embeddings.
Uses BAAI/bge-small-zh-v1.5 (512-dim, CPU-friendly) via sentence-transformers.

IMPORTANT: After upgrading this module, existing Milvus vectors are NOT compatible.
Re-ingest all documents to rebuild the vector index with semantic embeddings.

Design:
  - Global model singleton with thread lock (like reference system)
  - L2-normalized vectors for COSINE similarity in Milvus
  - Batch encoding for ingestion, single-query encoding for search
  - CPU by default; set EMBEDDING_DEVICE=cuda for GPU
"""

from __future__ import annotations

import logging
import threading
from typing import List

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Global model singleton ──────────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the sentence-transformers model (thread-safe singleton)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        from sentence_transformers import SentenceTransformer

        model_name = settings.embedding_model
        device = settings.embedding_device

        logger.info("正在加载嵌入模型: %s (device=%s)", model_name, device)
        _model = SentenceTransformer(model_name, device=device)
        # Verify dimension matches config
        actual_dim = _model.get_sentence_embedding_dimension()
        if actual_dim != settings.embedding_dim:
            logger.warning(
                "模型输出维度 (%d) 与配置维度 (%d) 不一致，请更新 EMBEDDING_DIM",
                actual_dim,
                settings.embedding_dim,
            )
        logger.info("嵌入模型加载完成: dim=%d", actual_dim)
        return _model


# ── Public API ──────────────────────────────────────────────

def encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts into dense vectors.

    Args:
        texts: List of text strings to encode

    Returns:
        List of vectors, each [dim] float list, L2-normalized for COSINE
    """
    if not texts:
        return []

    dim = settings.embedding_dim
    # Handle empty/whitespace-only texts
    empty = [0.0] * dim
    results = []
    real_texts = []
    real_indices = []

    for i, t in enumerate(texts):
        if not t or not t.strip():
            results.append((i, empty))
        else:
            real_indices.append(i)
            real_texts.append(t)

    if not real_texts:
        return [empty] * len(texts)

    model = _get_model()
    embeddings = model.encode(
        real_texts,
        batch_size=settings.ingest_batch_size,
        normalize_embeddings=True,  # L2-normalize for COSINE
        show_progress_bar=False,
    )

    # Reconstruct original order
    all_embeddings = [None] * len(texts)
    for i, emb in zip(real_indices, embeddings.tolist()):
        all_embeddings[i] = emb
    for i, emb in results:
        all_embeddings[i] = emb

    return all_embeddings


def encode_query(text: str) -> list[float]:
    """Encode a single query text into a dense vector.

    Args:
        text: Query text

    Returns:
        Single [dim] float list, L2-normalized
    """
    if not text or not text.strip():
        return [0.0] * settings.embedding_dim

    model = _get_model()
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0].tolist()
