"""AI 服务层 —— 统一 LLM 入口。

所有 LLM 调用必须通过此模块的公共 API，禁止在业务代码中直接创建
AsyncOpenAI / OpenAI / ChatOpenAI 实例。

公共 API：
- ``get_llm_client()``          → 轮询返回 AsyncOpenAI 客户端
- ``llm_chat(messages, ...)``    → 调用 LLM 并返回 text（带兜底重试）
- ``create_langchain_llm(...)``  → 创建 LangChain ChatModel（Agent 用）
- ``reload_llm_config(db)``      → 重载 DB 配置（设置保存后调用）
- ``get_router_stats()``         → Router 调试信息
"""
from __future__ import annotations

import logging

from app.services.ai.router import (
    get_llm_client,
    get_sync_llm_client,
    llm_chat,
    create_langchain_llm,
    reload_llm_config,
    get_router_stats,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_llm_client",
    "get_sync_llm_client",
    "llm_chat",
    "create_langchain_llm",
    "reload_llm_config",
    "get_router_stats",
]
