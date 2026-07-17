"""
BGE-M3 ONNX INT8 model loader — thread-safe singleton.

Loads the ONNX-quantized BGE-M3 model from HuggingFace
(gpahal/bge-m3-onnx-int8, ~500 MB) and provides:

  - encode_dense(texts) → list[list[float]]  (N × 1024, L2-normalized)
  - encode_sparse(texts) → list[dict[int, float]]  (token_id → weight)

Tokenizer is loaded from BAAI/bge-m3 (config files only, ~tens of MB).
ONNX Runtime is used for inference. No PyTorch model fallback.
"""

from __future__ import annotations

import logging
import threading
import numpy as np
from typing import List

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Global singletons ──────────────────────────────────

_tokenizer = None
_onnx_session = None
_lock = threading.Lock()


def _load_onnx_model():
    """Download and load the ONNX INT8 quantized BGE-M3 model."""
    global _onnx_session, _tokenizer

    logger.info("加载 BGE-M3 tokenizer: %s", settings.embedding_model)
    _tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model,
        local_files_only=True,
    )

    logger.info("加载 BGE-M3 ONNX 模型: %s / %s",
                settings.bge_onnx_model_name, settings.bge_onnx_filename)
    model_path = hf_hub_download(
        repo_id=settings.bge_onnx_model_name,
        filename=settings.bge_onnx_filename,
        local_files_only=True,
    )

    providers = []
    if settings.embedding_device == "cuda":
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    logger.info("创建 ONNX Runtime 会话 (providers=%s)", providers)
    _onnx_session = ort.InferenceSession(model_path, providers=providers)
    logger.info("BGE-M3 ONNX 模型加载完成")


def _get_model():
    """Lazy-load the model (thread-safe singleton)."""
    if _onnx_session is not None:
        return

    with _lock:
        if _onnx_session is not None:
            return
        _load_onnx_model()


# ── Public API ─────────────────────────────────────────

def encode_dense(texts: list[str]) -> list[list[float]]:
    """Encode texts into dense 1024-dim vectors (L2-normalized for COSINE)."""
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

    vectors = _encode_dense_onnx(real_texts)

    for idx, vec in zip(real_indices, vectors):
        results_map[idx] = vec

    return [results_map[i] for i in range(len(texts))]


def encode_sparse(texts: list[str]) -> list[dict[int, float]]:
    """Encode texts into sparse lexical weight vectors (BGE-M3)."""
    if not texts:
        return []

    _get_model()
    return _encode_sparse_onnx(texts)


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
    logger.info("预加载 BGE-M3 ONNX 模型...")
    _get_model()
    logger.info("BGE-M3 ONNX 模型预加载完成")


# ── ONNX encode internals ─────────────────────────────

def _encode_dense_onnx(texts: list[str]) -> list[list[float]]:
    """Dense encoding via ONNX Runtime."""
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

    if len(outputs) > 1:
        sparse_output = outputs[1]
    else:
        logger.warning("ONNX model 未输出稀疏向量，使用阈值截断回退")
        sparse_output = outputs[0]

    results = []
    for i in range(sparse_output.shape[0]):
        row = sparse_output[i]
        if hasattr(row, "toarray"):
            row = row.toarray().flatten()
        nonzero = np.nonzero(row)[0]
        sparse_dict = {int(idx): float(row[idx]) for idx in nonzero}
        results.append(sparse_dict)

    return results
