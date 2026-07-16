"""Reranker service — CrossEncoder for precision retrieval.

Uses BGE-Reranker via sentence-transformers CrossEncoder to re-score
candidate documents against the query, producing a relevance-ranked result.

CPU by default; set EMBEDDING_DEVICE=cuda in .env for GPU acceleration.
"""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global singleton
_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """Lazy-load the CrossEncoder reranker (thread-safe singleton)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker

        from sentence_transformers import CrossEncoder

        model_name = settings.reranker_model
        device = settings.embedding_device

        logger.info("正在加载重排序模型: %s (device=%s)", model_name, device)
        _reranker = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        logger.info("重排序模型加载完成")
        return _reranker


def rerank(
    query: str,
    texts: list[str],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Cross-encode query against candidate texts and return ranked indices.

    Args:
        query: Search query
        texts: Candidate text list
        top_k: Max results to return (default: all)

    Returns:
        [(original_index, score), ...] sorted by score descending.
        Scores are clamped to [0.0, 1.0].
    """
    if not texts:
        return []

    if not settings.rerank_enabled:
        # Pass-through: return all with score 0
        return [(i, 0.0) for i in range(len(texts))]

    top_k = top_k or len(texts)

    try:
        model = _get_reranker()
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs, show_progress_bar=False)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [float(scores)]

        indexed = [
            (i, max(0.0, min(1.0, float(scores[i]))))
            for i in range(len(texts))
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    except Exception as e:
        logger.warning("重排序失败，回退到原始排序: %s", e)
        # Fallback: return in original order
        return [(i, 0.0) for i in range(len(texts))][:top_k]
