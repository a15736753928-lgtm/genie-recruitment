"""LlmRouter —— 优先级调度 + 轮询并发 + 跨 Key / 跨 Provider 兜底。

所有 LLM 调用通过此模块的两个核心函数：

- ``get_llm_client()`` → AsyncOpenAI（轮询返回，适合高级调用场景）
- ``llm_chat(messages)`` → str（推荐，带自动重试/兜底）

路由顺序：
  provider.priority ASC → key.priority ASC → 同 priority keys round-robin
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings, normalize_llm_model
from app.services.ai.config_store import load_llm_config, resolve_api_keys
from app.services.ai.providers.registry import get_adapter

logger = logging.getLogger(__name__)

# ── 连接池（openai_compatible 场景，复用 AsyncOpenAI 客户端） ──

# key_id → AsyncOpenAI client
_client_cache: dict[str, Any] = {}
# key_id → (provider, api_key_plain) 用于重建
_client_meta: dict[str, tuple[dict, str]] = {}

# 轮询索引
_round_robin_clients: list[str] = []  # 排序后的 key_id 列表
_round_robin_index: int = 0

# 配置版本号（用于检测变更）
_config_version: str = ""


async def reload_llm_config(db: AsyncSession) -> None:
    """重新加载 DB 配置并重建客户端池（系统设置保存后调用）。"""
    global _config_version, _client_cache, _client_meta, _round_robin_clients, _round_robin_index
    config = await load_llm_config(db)
    pairs = resolve_api_keys(config)

    # 检查配置是否变化
    new_version = _config_hash(config)
    if new_version == _config_version:
        return
    _config_version = new_version

    # 重建客户端池
    new_cache = {}
    new_meta = {}
    new_robin = []

    for pair in pairs:
        key_id = pair["key_id"]
        provider = pair["provider"]
        key_plain = pair["key_plain"]

        # 复用已有客户端（连接池不变）
        if key_id in _client_cache and key_id in _client_meta:
            old_provider, old_key = _client_meta[key_id]
            if old_provider == provider and old_key == key_plain:
                new_cache[key_id] = _client_cache[key_id]
                new_meta[key_id] = _client_meta[key_id]
                new_robin.append(key_id)
                continue

        # 新建客户端
        try:
            adapter = get_adapter(provider.get("type", "openai_compatible"))
            client = adapter.create_client(provider, key_plain)
            new_cache[key_id] = client
            new_meta[key_id] = (provider, key_plain)
            new_robin.append(key_id)
            logger.debug("Router: 初始化客户端 key_id=%s provider=%s", key_id, provider.get("name"))
        except Exception as exc:
            logger.warning("Router: 初始化客户端失败 key_id=%s: %s", key_id, exc)

    _client_cache = new_cache
    _client_meta = new_meta
    _round_robin_clients = new_robin
    _round_robin_index = 0
    logger.info("Router: 配置已重载，%d 个可用客户端", len(new_robin))


async def get_router_stats() -> dict:
    """返回当前 Router 状态（用于调试）。"""
    return {
        "version": _config_version,
        "clientCount": len(_client_cache),
        "activeKeyIds": _round_robin_clients,
        "nextIndex": _round_robin_index % len(_round_robin_clients) if _round_robin_clients else 0,
    }


def _config_hash(config: dict) -> str:
    import hashlib, json
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


# ── 公共 API ──────────────────────────────────────────────


def get_llm_client() -> Any:
    """轮询返回一个 AsyncOpenAI 客户端（openai_compatible 场景）。

    无可用客户端时回退到 env 配置。
    """
    global _round_robin_index
    if not _round_robin_clients:
        return _fallback_client()

    idx = _round_robin_index % len(_round_robin_clients)
    _round_robin_index = (idx + 1) % len(_round_robin_clients)
    key_id = _round_robin_clients[idx]
    return _client_cache.get(key_id, _fallback_client())


def get_sync_llm_client() -> Any:
    """返回同步 OpenAI 客户端（用于 sync 上下文，如社区检测、图谱索引）。

    直接使用第一个可用 Key，不做轮询（sync 调用频率低）。
    """
    from openai import OpenAI
    settings = get_settings()
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    key = keys[0] if keys else ""
    return OpenAI(
        api_key=key,
        base_url=settings.deepseek_base_url,
        timeout=60.0,
        max_retries=2,
    )


def _fallback_client() -> Any:
    """兜底：用 env 配置创建临时客户端。"""
    from openai import AsyncOpenAI
    settings = get_settings()
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    key = keys[0] if keys else ""
    return AsyncOpenAI(
        api_key=key,
        base_url=settings.deepseek_base_url,
        timeout=60.0,
        max_retries=2,
    )


async def llm_chat(
    messages: list,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    max_retries: int = 1,
) -> str:
    """调用 LLM 并返回 content 文本。

    自动按 priority 尝试所有 Key → Provider，带指数退避重试。
    默认 max_retries=1（只尝试一轮所有 Key），避免与 SDK 层
    max_retries=2 叠加导致 60+ 秒重试风暴。
    """
    settings = get_settings()
    model = normalize_llm_model(model or settings.deepseek_model)

    # 如果 Router 还没 warm-up，先兜底
    if not _round_robin_clients:
        return await _llm_chat_direct(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
        )

    last_error: Exception | None = None

    for attempt in range(max_retries):
        client_order = _clients_in_robin_order()
        for key_id, client in client_order:
            try:
                # DeepSeek v4 默认思考模式会吃满 max_tokens 导致 content 为空，
                # 对 deepseek 系列统一关闭思考（提速且避免空输出）。
                extra = ({"thinking": {"type": "disabled"}}
                         if model.lower().startswith("deepseek") else {})
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **(extra and {"extra_body": extra} or {}),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM 调用失败 key=%s attempt=%d: %s",
                    key_id, attempt + 1, exc,
                )
                # 立即换下一个 key
                continue

        # 所有 key 本轮都失败，退避
        if attempt < max_retries - 1:
            wait = 1.5 ** attempt
            logger.warning("所有 Key 本轮均失败，%0.1fs 后退避重试", wait)
            await asyncio.sleep(wait)

    raise last_error if last_error else RuntimeError("LLM 调用失败：无可用客户端")


async def _llm_chat_direct(
    messages: list,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """兜底直连（env 配置）。"""
    client = _fallback_client()
    extra = ({"thinking": {"type": "disabled"}}
             if model.lower().startswith("deepseek") else {})
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **(extra and {"extra_body": extra} or {}),
    )
    return (resp.choices[0].message.content or "").strip()


def _clients_in_robin_order() -> list[tuple[str, Any]]:
    """按当前轮询位排列客户端，优先尝试当前 key。"""
    global _round_robin_index
    if not _round_robin_clients:
        return []
    n = len(_round_robin_clients)
    start = _round_robin_index % n
    result = []
    for i in range(n):
        idx = (start + i) % n
        key_id = _round_robin_clients[idx]
        if key_id in _client_cache:
            result.append((key_id, _client_cache[key_id]))
    return result


def create_langchain_llm(
    *,
    streaming: bool = False,
    temperature: float = 0.7,
) -> Any:
    """创建一个 LangChain ChatModel（用于 Agent）。

    从当前 Router 配置的第一个可用 provider 创建。
    仅支持 openai_compatible provider type。
    """
    if not _round_robin_clients:
        # 回退
        from langchain_openai import ChatOpenAI
        settings = get_settings()
        return ChatOpenAI(
            model=normalize_llm_model(settings.deepseek_model),
            api_key=settings.deepseek_api_key.split(",")[0].strip(),
            base_url=settings.deepseek_base_url,
            temperature=temperature,
            streaming=streaming,
            timeout=60,
            max_retries=2,
            model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},  # deepseek-v4-flash 关闭思考
        )

    # 轮询选择下一个 key（与 llm_chat 共享 _round_robin_index）：
    # 让对话 agent 跨对话负载均衡，不再永远打第一个 provider。
    # 同轮内某 key 失败的 failover 由 graph.py 的 call_model 捕获异常后重建重试。
    global _round_robin_index
    idx = _round_robin_index % len(_round_robin_clients)
    _round_robin_index = (idx + 1) % len(_round_robin_clients)
    key_id = _round_robin_clients[idx]
    meta = _client_meta.get(key_id)
    if not meta:
        return _fallback_langchain(temperature, streaming)

    provider, key_plain = meta
    adapter = get_adapter(provider.get("type", "openai_compatible"))
    return adapter.create_langchain_llm(provider, key_plain, streaming=streaming, temperature=temperature)


def _fallback_langchain(temperature: float, streaming: bool) -> Any:
    from langchain_openai import ChatOpenAI
    settings = get_settings()
    return ChatOpenAI(
        model=normalize_llm_model(settings.deepseek_model),
        api_key=settings.deepseek_api_key.split(",")[0].strip(),
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=60,
        max_retries=2,
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},  # deepseek-v4-flash 关闭思考
    )
