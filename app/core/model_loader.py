"""
BGE-M3 ONNX INT8 model loader — thread-safe singleton.

Blueprint alignment: Section 6.1

Loads the ONNX-quantized BGE-M3 model from HuggingFace
(gpahal/bge-m3-onnx-int8) and provides:

  - encode_dense(texts) → list[list[float]]  (N × 1024, L2-normalized)
  - encode_sparse(texts) → list[dict[int, float]]  (token_id → weight)

Uses ONNX Runtime for inference. Falls back to sentence-transformers
PyTorch model if ONNX is unavailable, with a warning.

Both dense and sparse outputs come from a single BGE-M3 forward pass.
"""

from __future__ import annotations

import logging
import threading
import numpy as np
from typing import List

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Global singletons ──────────────────────────────────

_model = None
_tokenizer = None
_onnx_session = None
_lock = threading.Lock()
_use_onnx = True


def _load_onnx_model():
    """Download and load the ONNX INT8 quantized BGE-M3 model."""
    global _onnx_session, _tokenizer

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        logger.info("加载 BGE-M3 tokenizer: %s", settings.embedding_model)
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.embedding_model,
            cache_dir=settings.model_cache_dir,
        )

        logger.info("下载 BGE-M3 ONNX 模型: %s", settings.bge_onnx_model_name)
        model_path = hf_hub_download(
            repo_id=settings.bge_onnx_model_name,
            filename=settings.bge_onnx_filename,
            cache_dir=settings.model_cache_dir,
        )

        # GPU if available, else CPU
        providers = []
        if settings.embedding_device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        logger.info("创建 ONNX Runtime 会话 (providers=%s)", providers)
        _onnx_session = ort.InferenceSession(model_path, providers=providers)
        logger.info("BGE-M3 ONNX 模型加载完成")

    except ImportError as e:
        logger.warning("ONNX Runtime 不可用，回退到 PyTorch: %s", e)
        return False
    except Exception as e:
        logger.warning("ONNX 模型加载失败，回退到 PyTorch: %s", e)
        return False
    return True


def _load_pytorch_model():
    """Fallback: load BGE-M3 via sentence-transformers."""
    global _model

    from sentence_transformers import SentenceTransformer

    logger.info("加载 BGE-M3 PyTorch 模型: %s (device=%s)",
                settings.embedding_model, settings.embedding_device)
    _model = SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
        cache_folder=settings.model_cache_dir,
    )
    actual_dim = _model.get_sentence_embedding_dimension()
    logger.info("BGE-M3 PyTorch 模型加载完成: dim=%d", actual_dim)


def _get_model():
    """Lazy-load the model (thread-safe singleton).

    Tries ONNX first, falls back to PyTorch sentence-transformers.
    """
    global _model, _use_onnx

    if _model is not None or _onnx_session is not None:
        return

    with _lock:
        if _model is not None or _onnx_session is not None:
            return

        if _use_onnx:
            if _load_onnx_model():
                return
            _use_onnx = False

        _load_pytorch_model()


# ── Public API ─────────────────────────────────────────

def encode_dense(texts: list[str]) -> list[list[float]]:
    """Encode texts into dense 1024-dim vectors (L2-normalized for COSINE).

    Args:
        texts: List of text strings to encode.

    Returns:
        List of [1024] float lists, each L2-normalized.
    """
    if not texts:
        return []

    _get_model()
    dim = settings.embedding_dim  # 1024

    empty_vec = [0.0] * dim
    real_texts = []
    real_indices = []
    results_map = {}

    for i, t in enumerate(texts):
        if not t or not t.strip():
            results_map[i] = empty_vec
        else:
            real_indices.append(i)
            real_texts.append(t)

    if not real_texts:
        return [empty_vec] * len(texts)

    if _onnx_session is not None:
        vectors = _encode_dense_onnx(real_texts)
    else:
        vectors = _encode_dense_pt(real_texts)

    for idx, vec in zip(real_indices, vectors):
        results_map[idx] = vec

    return [results_map[i] for i in range(len(texts))]


def encode_sparse(texts: list[str]) -> list[dict[int, float]]:
    """Encode texts into sparse lexical weight vectors (BGE-M3).

    Each sparse vector is {token_id: weight}, usable with IP (inner product)
    metric in Milvus.

    Args:
        texts: List of text strings.

    Returns:
        List of {token_id: float_weight} dicts.
    """
    if not texts:
        return []

    _get_model()

    if _onnx_session is not None:
        return _encode_sparse_onnx(texts)
    else:
        return _encode_sparse_pt(texts)


def encode_query_dense(text: str) -> list[float]:
    """Single-query dense encoding."""
    results = encode_dense([text])
    return results[0] if results else [0.0] * settings.embedding_dim


def encode_query_sparse(text: str) -> dict[int, float]:
    """Single-query sparse encoding."""
    results = encode_sparse([text])
    return results[0] if results else {}


def preload_models():
    """Preload all models at startup (called from lifespan)."""
    logger.info("预加载 BGE-M3 模型...")
    _get_model()
    logger.info("BGE-M3 模型预加载完成")


# ── ONNX encode internals ─────────────────────────────

def _encode_dense_onnx(texts: list[str]) -> list[list[float]]:
    """Dense encoding via ONNX Runtime."""
    # Tokenize
    encoded = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="np",
    )

    # ONNX expects specific input names; BGE-M3 uses "input_ids", "attention_mask"
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    # Run inference
    outputs = _onnx_session.run(None, ort_inputs)
    # Outputs: [dense_embeddings, sparse_lexical_weights] — depends on model export
    # For gpahal/bge-m3-onnx-int8, output 0 is dense_emb
    dense = outputs[0]  # shape: (N, 1024)

    # L2 normalize
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    dense = dense / norms

    return dense.tolist()


def _encode_sparse_onnx(texts: list[str]) -> list[dict[int, float]]:
    """Sparse encoding via ONNX Runtime."""
    encoded = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="np",
    )

    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    outputs = _onnx_session.run(None, ort_inputs)

    # gpahal/bge-m3-onnx-int8: output[1] is sparse lexical weights
    if len(outputs) > 1:
        sparse_output = outputs[1]  # shape: (N, vocab_size) or (N, num_tokens)
    else:
        # Fallback: use dense output with threshold as pseudo-sparse
        logger.warning("ONNX model 未输出稀疏向量，使用阈值截断回退")
        sparse_output = outputs[0]

    # Convert to list of {token_id: weight} dicts
    results = []
    for i in range(sparse_output.shape[0]):
        row = sparse_output[i]
        # Get non-zero indices and values
        if hasattr(row, "toarray"):
            row = row.toarray().flatten()
        nonzero = np.nonzero(row)[0]
        sparse_dict = {int(idx): float(row[idx]) for idx in nonzero}
        results.append(sparse_dict)

    return results


# ── PyTorch fallback encode internals ─────────────────

def _encode_dense_pt(texts: list[str]) -> list[list[float]]:
    """Dense encoding via sentence-transformers (PyTorch fallback)."""
    embeddings = _model.encode(
        texts,
        batch_size=settings.ingest_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _encode_sparse_pt(texts: list[str]) -> list[dict[int, float]]:
    """Sparse encoding via BGE-M3 PyTorch (BGEM3FlagModel)."""
    try:
        from FlagEmbedding import BGEM3FlagModel

        # The PyTorch model needs to be loaded as BGEM3FlagModel for sparse output
        # If _model is SentenceTransformer, reload as BGEM3FlagModel
        flag_model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=True,
            device=settings.embedding_device,
        )
        outputs = flag_model.encode(
            texts,
            batch_size=settings.ingest_batch_size,
            return_dense=False,
            return_sparse=True,
        )
        sparse_lexical = outputs.get("lexical_weights", [])
        return sparse_lexical

    except ImportError:
        logger.warning("FlagEmbedding 不可用，稀疏向量返回空字典")
        return [{} for _ in texts]
    except Exception as e:
        logger.warning("稀疏编码失败: %s", e)
        return [{} for _ in texts]
