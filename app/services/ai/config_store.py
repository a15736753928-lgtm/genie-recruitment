"""LLM 配置的持久化与缓存。

- 存在 system_settings 表的 key="llm_config" 行中，与 global settings 隔离
- 内存缓存（60s TTL），保存时 invalidate
- DB 为空时从 .env 自动 bootstrap 一条 openai_compatible 默认提供商
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB

from app.config import get_settings
from app.models.settings import SystemSetting
from app.services.ai.crypto import (
    encrypt_key,
    decrypt_key,
    mask_key,
    SENTINEL_UNCHANGED,
)

logger = logging.getLogger(__name__)

# ── 类型定义 ──────────────────────────────────────────────

# Provider types
PROVIDER_TYPES = ("openai_compatible", "anthropic", "azure_openai")

# 默认 LLM 配置（空列表，由 env bootstrap 填充）
DEFAULT_LLM_CONFIG: dict = {"providers": [], "defaultProviderId": None}

# ── 内存缓存 ──────────────────────────────────────────────

_cache: Optional[dict] = None
_cache_expires: float = 0.0
_CACHE_TTL: float = 60.0


def invalidate_llm_cache() -> None:
    global _cache, _cache_expires
    _cache = None
    _cache_expires = 0.0


async def _get_stored_config(db: AsyncSession) -> Optional[dict]:
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "llm_config")
    )
    row = result.scalar_one_or_none()
    if row and isinstance(row.value, dict):
        return row.value
    return None


# ── 公共 API ──────────────────────────────────────────────

async def load_llm_config(db: AsyncSession) -> dict:
    """加载 LLM 配置（含缓存）。DB 为空时自动从 .env bootstrap。"""
    global _cache, _cache_expires
    now = time.monotonic()
    if _cache is not None and now < _cache_expires:
        return _cache  # type: ignore[return-value]

    stored = await _get_stored_config(db)
    if stored and stored.get("providers"):
        config = _normalize_config(stored)
    else:
        config = _bootstrap_from_env()

    _cache = config
    _cache_expires = now + _CACHE_TTL
    return config


async def save_llm_config(db: AsyncSession, incoming: dict) -> dict:
    """保存 LLM 配置（合并加密 Key 的占位值）。

    PUT 语义：incoming 中每个 provider.api_keys[].apiKey 为
    - 非空新明文 → 加密写入
    - SENTINEL_UNCHANGED → 从 DB 旧值保留密文
    - 空字符串 → 视为空（删除 Key 行）

    返回脱敏后的配置（Key 显示为 masked）。
    """
    old = await _get_stored_config(db) or {}
    merged = _merge_providers(old, incoming)

    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "llm_config")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = merged
    else:
        row = SystemSetting(key="llm_config", value=merged)
        db.add(row)
    await db.flush()
    invalidate_llm_cache()
    return _masked_config(merged)


# ── 内部 ──────────────────────────────────────────────────


def _normalize_config(raw: dict) -> dict:
    """给 config 补充缺失字段，保证结构完整。"""
    for p in raw.get("providers", []):
        p.setdefault("id", str(uuid.uuid4())[:8])
        p.setdefault("enabled", True)
        p.setdefault("priority", p.get("priority") or 10)
        p.setdefault("model", "")
        for k in p.get("apiKeys", []):
            k.setdefault("id", str(uuid.uuid4())[:8])
            k.setdefault("enabled", True)
            k.setdefault("priority", k.get("priority") or 10)
    return raw


def _merge_providers(old: dict, incoming: dict) -> dict:
    """合并 incoming → old，处理 Key 的 SENTINEL_UNCHANGED 占位。"""
    merged = dict(old)
    merged["providers"] = list(old.get("providers", []))
    merged["defaultProviderId"] = incoming.get("defaultProviderId")

    incoming_providers = incoming.get("providers", [])
    new_providers = []
    for i, inc_p in enumerate(incoming_providers):
        inc_id = inc_p.get("id", "")
        # 查找旧 provider
        old_p = None
        for op in merged["providers"]:
            if op.get("id") == inc_id and inc_id:
                old_p = op
                break
        if old_p is None and inc_id:
            # 更新已有
            old_p = merged["providers"][i] if i < len(merged["providers"]) else None

        # 合并 apiKeys
        merged_keys = []
        old_keys = {k.get("id"): k for k in (old_p.get("apiKeys", []) if old_p else [])}
        for inc_k in inc_p.get("apiKeys", []):
            raw_key = inc_k.get("apiKey", "")
            if raw_key == SENTINEL_UNCHANGED:
                # 保留旧密文
                old_k = old_keys.get(inc_k.get("id", ""))
                inc_k["apiKey"] = old_k.get("apiKey", "") if old_k else ""
            elif raw_key:
                # 新明文 → 加密
                inc_k["apiKey"] = encrypt_key(raw_key)
            else:
                inc_k["apiKey"] = ""
            merged_keys.append(inc_k)

        merged_p = {**inc_p, "apiKeys": merged_keys}
        new_providers.append(merged_p)

    merged["providers"] = new_providers
    return merged


def _bootstrap_from_env() -> dict:
    """从 .env 创建默认 openai_compatible provider（兼容现有部署）。"""
    settings = get_settings()
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    if not keys:
        logger.warning("LLM config bootstrap: .env 无有效 API Key")
        return {"providers": [], "defaultProviderId": None}

    provider_id = "default"
    provider = {
        "id": provider_id,
        "name": "DeepSeek（默认）",
        "type": "openai_compatible",
        "enabled": True,
        "priority": 1,
        "model": settings.deepseek_model,
        "baseUrl": settings.deepseek_base_url,
        "apiKeys": [
            {
                "id": f"key-{i+1}",
                "label": f"Key #{i+1}",
                "enabled": True,
                "priority": i + 1,
                "apiKey": encrypt_key(key),
            }
            for i, key in enumerate(keys)
        ],
    }
    logger.info("LLM config bootstrap：从 .env 创建 provider=%s keys=%d", provider_id, len(keys))
    return {"providers": [provider], "defaultProviderId": provider_id}


def _masked_config(config: dict) -> dict:
    """返回脱敏后的配置（Key 替换为 masked）。"""
    import copy
    masked = copy.deepcopy(config)
    for p in masked.get("providers", []):
        for k in p.get("apiKeys", []):
            plain = decrypt_key(k.get("apiKey", ""))
            k["apiKey"] = mask_key(plain) if plain else ""
    return masked


def resolve_api_keys(config: dict) -> list[dict]:
    """从配置中提取所有 enabled 的 (provider, key) 对，按 priority 排序。

    Returns list of {provider, key_plain, key_id}
    """
    pairs = []
    for p in config.get("providers", []):
        if not p.get("enabled", True):
            continue
        for k in p.get("apiKeys", []):
            if not k.get("enabled", True):
                continue
            plain = decrypt_key(k.get("apiKey", ""))
            if not plain:
                continue
            pairs.append({
                "provider": p,
                "key_plain": plain,
                "key_id": k.get("id", ""),
                "provider_priority": p.get("priority", 10),
                "key_priority": k.get("priority", 10),
                "provider_id": p.get("id", ""),
            })
    # 排序：先按 provider priority，再按 key priority
    pairs.sort(key=lambda x: (x["provider_priority"], x["key_priority"]))
    return pairs
