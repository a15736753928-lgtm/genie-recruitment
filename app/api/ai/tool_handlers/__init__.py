"""Tool handler registry — replaces the monolithic if/elif chain in tool_executor.py.

Each domain (recruitment, positions, interview, etc.) registers its handlers
via ``register_handlers(registry)``.  The centralized ``ToolHandlerRegistry``
dispatches tool calls to the correct handler by name.

Adding a new tool:
  1. Add a handler function in the appropriate domain module.
  2. Call ``registry.register("tool_name", handler_func)`` in that module's
     ``register_handlers``.
  3. Done — no need to touch tool_executor.py.
"""

from __future__ import annotations

from typing import Callable, Awaitable

from sqlalchemy.ext.asyncio import AsyncSession

# A tool handler receives (params: dict, db: AsyncSession) → str
ToolHandler = Callable[[dict, AsyncSession], Awaitable[str]]


class ToolHandlerRegistry:
    """Registry of named tool handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register a handler for a tool name."""
        self._handlers[name] = handler

    def get(self, name: str) -> ToolHandler | None:
        """Return the handler for *name*, or ``None``."""
        return self._handlers.get(name)

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._handlers.keys())


# ── Singleton registry ──────────────────────────────────────

_registry: ToolHandlerRegistry | None = None


def get_registry() -> ToolHandlerRegistry:
    """Return the global tool handler registry (lazy-init)."""
    global _registry
    if _registry is None:
        _registry = ToolHandlerRegistry()
        _bootstrap(_registry)
    return _registry


def _bootstrap(reg: ToolHandlerRegistry) -> None:
    """Import and register all handler modules."""
    from app.api.ai.tool_handlers import (
        recruitment,
        positions,
        interview,
        probation,
        performance,
        # knowledge 知识库/RAG handlers 已断开（2026-08-01），文件保留，恢复时加回
        dashboard,
        settings,
    )
    recruitment.register_handlers(reg)
    positions.register_handlers(reg)
    interview.register_handlers(reg)
    probation.register_handlers(reg)
    performance.register_handlers(reg)
    # knowledge.register_handlers(reg)  # 已断开（2026-08-01）
    dashboard.register_handlers(reg)
    settings.register_handlers(reg)
