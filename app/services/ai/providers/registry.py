"""Provider type → Adapter 注册表。"""
from __future__ import annotations

import logging

from app.services.ai.providers.base import LlmProviderAdapter, ProviderInitError
from app.services.ai.providers.openai_compatible import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

# ── 注册的所有 adapter ─────────────────────────────────────
# 按 provider_type 索引
_REGISTRY: dict[str, LlmProviderAdapter] = {
    "openai_compatible": OpenAICompatibleAdapter,
}

# 可选 adapter：安装对应包后自动可用
try:
    from app.services.ai.providers.anthropic import AnthropicAdapter
    _REGISTRY["anthropic"] = AnthropicAdapter
except ImportError:
    pass

try:
    from app.services.ai.providers.azure_openai import AzureOpenAIAdapter
    _REGISTRY["azure_openai"] = AzureOpenAIAdapter
except Exception:
    pass


def get_adapter(provider_type: str) -> LlmProviderAdapter:
    """根据 provider type 获取对应的 adapter。

    Raises:
        ProviderInitError: 不支持的 provider type 或缺少依赖。
    """
    if provider_type not in _REGISTRY:
        raise ProviderInitError(
            f"不支持的 LLM 提供商类型：{provider_type}。"
            f"支持的类型：{', '.join(_REGISTRY.keys())}"
        )
    return _REGISTRY[provider_type]


def supported_types() -> list[str]:
    """返回当前已注册的所有 provider type。"""
    return list(_REGISTRY.keys())
