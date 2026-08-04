"""Anthropic 原生协议 Provider（Claude 系列）。

需要 ``pip install anthropic langchain-anthropic``。
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.ai.providers.base import LlmProviderAdapter, ProviderInitError

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Anthropic Messages API 适配器。"""

    @staticmethod
    def provider_type() -> str:
        return "anthropic"

    @staticmethod
    def create_client(
        provider: dict,
        api_key_plain: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> Any:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ProviderInitError("anthropic 包未安装，请运行 pip install anthropic")

        return AsyncAnthropic(
            api_key=api_key_plain,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def create_langchain_llm(
        provider: dict,
        api_key_plain: str,
        *,
        streaming: bool = False,
        temperature: float = 0.7,
    ) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ProviderInitError("langchain-anthropic 包未安装，请运行 pip install langchain-anthropic")

        return ChatAnthropic(
            model=provider.get("model", "claude-sonnet-4-20250514"),
            api_key=api_key_plain,
            temperature=temperature,
            streaming=streaming,
            # Anthropic API 强制要求显式 max_tokens，缺省会 400
            max_tokens=8192,
        )

    @staticmethod
    async def test_connection(provider: dict, api_key_plain: str) -> tuple[bool, str]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return False, "anthropic 包未安装"
        client = AsyncAnthropic(api_key=api_key_plain, timeout=10.0, max_retries=0)
        try:
            # 发一条极短消息验证连通性
            resp = await client.messages.create(
                model=provider.get("model", "claude-sonnet-4-20250514"),
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True, f"连通（model={resp.model}）"
        except Exception as exc:
            logger.warning("Anthropic provider [%s] 连通性测试失败: %s", provider.get("name"), exc)
            return False, str(exc)


_: LlmProviderAdapter = AnthropicAdapter  # type: ignore[assignment]
