"""Knowledge base runtime settings — system_settings layered over config defaults."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.system.system_settings import DEFAULT_SETTINGS, get_system_settings

_cfg = get_settings()

KB_SETTING_DEFAULTS: dict[str, Any] = {
    "recallThreshold": 0.75,
    "kbDefaultTopK": 10,
    "kbRerankEnabled": True,
    "kbOcrEnabled": True,
}


def _merged_from_dict(stored: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **KB_SETTING_DEFAULTS, **stored}


async def get_kb_runtime_settings(db: AsyncSession) -> dict[str, Any]:
    """Resolved KB settings for search APIs."""
    sys = await get_system_settings(db)
    merged = _merged_from_dict(sys)
    return {
        "recall_threshold": float(merged.get("recallThreshold") or 0.75),
        "default_top_k": int(merged.get("kbDefaultTopK") or 10),
        "rerank_enabled": bool(merged.get("kbRerankEnabled", True)),
    }


def get_kb_ingest_settings_sync(db) -> dict[str, Any]:
    """Sync variant for ingest worker threads."""
    from sqlalchemy import select

    from app.models.settings import SystemSetting

    result = db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()
    stored = setting.value if setting and isinstance(setting.value, dict) else {}
    merged = _merged_from_dict(stored)
    return {
        "ocr_enabled": bool(merged.get("kbOcrEnabled", True)),
    }


def get_kb_engine_info() -> dict[str, Any]:
    """Read-only engine metadata for settings UI."""
    return {
        "embeddingModel": _cfg.embedding_model,
        "embeddingDim": _cfg.embedding_dim,
        "rerankerModel": _cfg.reranker_model,
        "chunkSize": _cfg.chunk_size,
        "chunkOverlap": _cfg.chunk_overlap,
        "supportedExtensions": sorted({".txt", ".md", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}),
        "vectorStore": "Milvus Lite",
        "sparseVectorEnabled": _cfg.sparse_vector_enabled,
    }
