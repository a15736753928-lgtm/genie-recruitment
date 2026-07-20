"""LLM 配置管理 API。

独立于 /settings 的 CRUD 端点，避免 Key 明文被误写回 global JSONB。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.settings import AuditLog
from app.services.ai import reload_llm_config, get_router_stats
from app.services.ai.config_store import (
    load_llm_config,
    save_llm_config,
    resolve_api_keys,
    _bootstrap_from_env,
)
from app.services.ai.crypto import mask_key, decrypt_key, SENTINEL_UNCHANGED
from app.services.ai.providers.registry import get_adapter, supported_types

logger = logging.getLogger(__name__)
router = APIRouter(tags=["LLM 配置"])


def _audit(db, actor: str, action: str, section: str = "llm_config"):
    """写审计日志（不包含 Key 明文）。"""
    log = AuditLog(
        id=f"log-{uuid.uuid4().hex[:12]}",
        time=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        actor=actor,
        action=action,
        section=section,
    )
    db.add(log)


# ── 配置读写 ──────────────────────────────────────────────

@router.get("/api/settings/llm-config")
async def get_llm_config(db: AsyncSession = Depends(get_db)):
    """获取 LLM 配置（Key 脱敏）。"""
    config = await load_llm_config(db)
    # 脱敏
    import copy
    masked = copy.deepcopy(config)
    for p in masked.get("providers", []):
        for k in p.get("apiKeys", []):
            if k.get("apiKey"):
                plain = decrypt_key(k["apiKey"])
                k["apiKey"] = mask_key(plain) if plain else ""
    return {"code": 0, "message": "ok", "data": masked}


@router.put("/api/settings/llm-config")
async def update_llm_config(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """保存 LLM 配置。

    - Key 字段传明文 → 加密存储
    - Key 字段传 "__LLM_KEY_UNCHANGED__" → 保留 DB 旧值（不变更）
    - 保存后自动重载 Router，立即生效
    """
    masked = await save_llm_config(db, body)
    await reload_llm_config(db)
    return {"code": 0, "message": "ok", "data": masked}


# ── 测试 & 重载 ───────────────────────────────────────────

@router.post("/api/settings/llm-config/test")
async def test_llm_provider(body: dict, db: AsyncSession = Depends(get_db)):
    """测试指定 provider + key 的连通性。

    body: { providerType, baseUrl, model, apiKey, azureEndpoint?, azureDeployment?, azureApiVersion? }
    """
    provider_type = body.get("providerType", "openai_compatible")
    api_key = body.get("apiKey", "").strip()
    if not api_key:
        return {"code": 400, "message": "API Key 不能为空", "data": None}

    # 构建临时 provider dict
    provider = {
        "name": "测试",
        "type": provider_type,
        "model": body.get("model", ""),
        "baseUrl": body.get("baseUrl", body.get("base_url", "")),
        "azureEndpoint": body.get("azureEndpoint", body.get("azure_endpoint", "")),
        "azureDeployment": body.get("azureDeployment", body.get("azure_deployment", "")),
        "azureApiVersion": body.get("azureApiVersion", body.get("azure_api_version", "")),
    }

    try:
        adapter = get_adapter(provider_type)
        ok, msg = await adapter.test_connection(provider, api_key)
        return {"code": 0 if ok else 1, "message": msg, "data": {"ok": ok}}
    except Exception as exc:
        return {"code": 1, "message": str(exc), "data": {"ok": False}}


@router.post("/api/settings/llm-config/reload")
async def manual_reload_llm(db: AsyncSession = Depends(get_db)):
    """手动刷新 Router 缓存（调试用）。"""
    await reload_llm_config(db)
    stats = await get_router_stats()
    return {"code": 0, "message": "Router 已重载", "data": stats}


@router.get("/api/settings/llm-config/supported-types")
async def get_supported_types():
    """返回当前环境支持的 provider type 列表。"""
    return {"code": 0, "message": "ok", "data": supported_types()}


@router.get("/api/settings/llm-config/router-stats")
async def get_router_status():
    """返回 Router 当前状态（调试用）。"""
    return {"code": 0, "message": "ok", "data": await get_router_stats()}
