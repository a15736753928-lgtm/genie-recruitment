"""Azure OpenAI Provider。

使用 openai 包的 AsyncAzureOpenAI。
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.ai.providers.base import LlmProviderAdapter, ProviderInitError

logger = logging.getLogger(__name__)


class AzureOpenAIAdapter:
    """Azure OpenAI 适配器。"""

    @staticmethod
    def provider_type() -> str:
        return "azure_openai"

    @staticmethod
    def create_client(
        provider: dict,
        api_key_plain: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> Any:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            raise ProviderInitError("openai 包版本过低，不支持 AsyncAzureOpenAI")

        endpoint = provider.get("azureEndpoint") or provider.get("azure_endpoint", "")
        if not endpoint:
            raise ProviderInitError("Azure provider 缺少 azureEndpoint")

        return AsyncAzureOpenAI(
            api_key=api_key_plain,
            azure_endpoint=endpoint,
            api_version=provider.get("azureApiVersion") or provider.get("azure_api_version", "2024-10-21"),
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
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ProviderInitError("langchain-openai 未安装")

        deployment = provider.get("azureDeployment") or provider.get("azure_deployment", "")
        endpoint = provider.get("azureEndpoint") or provider.get("azure_endpoint", "")

        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key_plain,
            api_version=provider.get("azureApiVersion") or provider.get("azure_api_version", "2024-10-21"),
            temperature=temperature,
            streaming=streaming,
            timeout=60,
            max_retries=2,
        )

    @staticmethod
    async def test_connection(provider: dict, api_key_plain: str) -> tuple[bool, str]:
        try:
            client = AzureOpenAIAdapter.create_client(provider, api_key_plain, timeout=10.0, max_retries=0)
            resp = await client.models.list()
            model_count = len(resp.data) if hasattr(resp, "data") else 0
            return True, f"连通（{model_count} 个模型可用）"
        except Exception as exc:
            logger.warning("Azure provider [%s] 连通性测试失败: %s", provider.get("name"), exc)
            return False, str(exc)


_: LlmProviderAdapter = AzureOpenAIAdapter  # type: ignore[assignment]
