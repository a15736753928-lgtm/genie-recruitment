"""AI 服务层 —— 与 LLM 直接交互的业务 Agent。

支持多个 API Key 负载均衡 + 互相兜底：一个 Key 故障时自动切换到另一个。
"""
from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── 连接池 ────────────────────────────────────────────────

_clients: list[AsyncOpenAI] = []
_index: int = 0


def _init_clients() -> None:
    global _clients
    if _clients:
        return
    settings = get_settings()
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("未配置 DeepSeek API Key（DEEPSEEK_API_KEY 为空）")
    _clients = [
        AsyncOpenAI(
            api_key=key,
            base_url=settings.deepseek_base_url,
            timeout=60.0,
            max_retries=1,
        )
        for key in keys
    ]
    logger.info("DeepSeek 客户端池已初始化：%d 个 API Key", len(_clients))


def _all_clients() -> list[AsyncOpenAI]:
    _init_clients()
    return _clients


def get_llm_client() -> AsyncOpenAI:
    """轮询返回一个 AsyncOpenAI 客户端（多 Key 负载均衡）。"""
    global _index
    _init_clients()
    client = _clients[_index % len(_clients)]
    _index = (_index + 1) % len(_clients)
    return client


# ── 带兜底的 LLM 调用 ─────────────────────────────────────

async def llm_call_with_retry(
    model: str | None = None,
    *,
    messages: list,
    temperature: float = 0.2,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    """调用 LLM 并返回 content 文本。

    - 轮询选 Key（负载均衡）
    - 网络/服务端错误 → 换下一个 Key 重试（兜底）
    - 所有 Key 都失败 → 退避等待后继续尝试
    """
    clients = _all_clients()
    model = model or get_settings().deepseek_model
    last_error: Exception | None = None

    for attempt in range(max_retries):
        # 每个 attempt 尝试所有 key
        for offset, client in enumerate(_cycle_from_current(clients)):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                key_idx = (_index + offset) % len(clients)
                if offset < len(clients) - 1:
                    logger.warning("Key #%d 调用失败，换下一个: %s", key_idx, exc)
                else:
                    wait = 1.5 ** attempt
                    logger.warning(
                        "所有 Key 本轮均失败 (attempt %d/%d)，%0.1fs 后退避重试",
                        attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)

    raise last_error  # type: ignore[misc]


def _cycle_from_current(items: list) -> list:
    """从当前轮询位开始排列 items，保证首次尝试的是当前轮询 key。"""
    global _index
    _init_clients()
    start = _index % len(items)
    return items[start:] + items[:start]
