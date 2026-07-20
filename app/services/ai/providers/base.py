"""Provider Adapter 协议定义。"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

__all__ = ["LlmProviderAdapter", "ProviderInitError"]


class ProviderInitError(Exception):
    """Provider 初始化失败（缺少依赖、配置无效等）。"""


@runtime_checkable
class LlmProviderAdapter(Protocol):
    """LLM Provider 适配器协议。

    每个 provider type 需实现此协议的方法。
    """

    @staticmethod
    def provider_type() -> str:
        """返回 provider type 标识（openai_compatible / anthropic / azure_openai）。"""
        ...

    @staticmethod
    def create_client(
        provider: dict,
        api_key_plain: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> Any:
        """创建原生 SDK 客户端（AsyncOpenAI / AsyncAnthropic / AsyncAzureOpenAI）。

        返回的客户端至少需支持 ``chat.completions.create()`` 或等价接口。
        """
        ...

    @staticmethod
    def create_langchain_llm(
        provider: dict,
        api_key_plain: str,
        *,
        streaming: bool = False,
        temperature: float = 0.7,
    ) -> Any:
        """创建 LangChain ChatModel 实例（用于 Agent）。"""
        ...

    @staticmethod
    async def test_connection(provider: dict, api_key_plain: str) -> tuple[bool, str]:
        """测试连通性。返回 (ok, message)。"""
        ...
