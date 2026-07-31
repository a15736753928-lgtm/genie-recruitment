"""OpenAI 兼容协议 Provider（DeepSeek、OpenAI、Ollama 等）。"""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import normalize_llm_model
from app.services.ai.providers.base import LlmProviderAdapter

logger = logging.getLogger(__name__)


class OpenAICompatibleAdapter:
    """OpenAI 兼容协议适配器。"""

    @staticmethod
    def provider_type() -> str:
        return "openai_compatible"

    @staticmethod
    def create_client(
        provider: dict,
        api_key_plain: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key_plain,
            base_url=provider.get("baseUrl") or provider.get("base_url", ""),
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
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("langchain-openai 未安装，无法创建 LangChain LLM")

        return ChatOpenAI(
            model=normalize_llm_model(provider.get("model", "mimo-v2.5")),
            api_key=api_key_plain,
            base_url=provider.get("baseUrl") or provider.get("base_url", ""),
            temperature=temperature,
            streaming=streaming,
            timeout=60,
            max_retries=2,
        )

    @staticmethod
    async def test_connection(provider: dict, api_key_plain: str) -> tuple[bool, str]:
        client = OpenAICompatibleAdapter.create_client(provider, api_key_plain, timeout=10.0, max_retries=0)
        try:
            # 尝试列出模型（OpenAI 兼容端点通用探测方式）
            resp = await client.models.list()
            model_count = len(resp.data) if hasattr(resp, "data") else 0
            return True, f"连通（{model_count} 个模型可用）"
        except Exception as exc:
            logger.warning("Provider [%s] 连通性测试失败: %s", provider.get("name"), exc)
            return False, str(exc)


# 确保实现了 Protocol
_: LlmProviderAdapter = OpenAICompatibleAdapter  # type: ignore[assignment]
