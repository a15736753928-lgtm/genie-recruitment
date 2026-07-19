"""
Structured tool result types — replaces plain-text returns with typed data.

Mirrors Claude Code's pattern of returning structured, verifiable results
from tool calls instead of raw text strings that the LLM must parse.

Every tool result carries:
- ``success`` — whether the operation completed without error
- ``data`` — structured payload for the frontend to render as rich cards
- ``summary`` — one-line human-readable summary
- ``details`` — full context for the LLM to reason over
- ``display_hint`` — how the frontend should render this result
- ``error`` — structured error detail if success=False
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ErrorDetail:
    """Structured error information for LLM recovery and user display."""
    category: str          # "not_found" | "validation" | "permission" | "transient" | "system" | "business"
    message: str           # Human-readable error message (Chinese)
    llm_hint: str          # Recovery hint injected into LLM context
    retryable: bool = False
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Typed result from any tool execution.

    Serialization:
        ``to_llm_context()`` → compact text for the LLM's context window
        ``to_sse_dict()`` → structured dict for SSE ``tool_result`` event
        ``to_display()`` → rich dict for frontend card rendering
    """
    success: bool
    tool_name: str
    summary: str                          # One-line for progress display
    details: str                          # Full context for LLM reasoning
    data: Optional[dict[str, Any]] = None # Structured payload
    error: Optional[ErrorDetail] = None
    display_hint: str = "text"            # "card" | "list" | "table" | "text" | "score" | "questions"
    affected_ids: list[str] = field(default_factory=list)  # IDs of affected entities (for cache invalidation)

    def to_llm_context(self) -> str:
        """Format for injection into LLM conversation context.

        Keeps it compact so we don't waste token budget on verbose formatting.
        """
        if self.success:
            return self.details
        # Error path: give the LLM structured recovery information
        parts = [f"❌ {self.tool_name} 失败: {self.error.message}"]
        if self.error.llm_hint:
            parts.append(f"提示: {self.error.llm_hint}")
        if self.error.suggested_actions:
            parts.append(f"建议操作: {'; '.join(self.error.suggested_actions)}")
        return "\n".join(parts)

    def to_sse_dict(self) -> dict:
        """Serialize for SSE ``tool_result`` event.

        The frontend uses ``display_hint`` to decide which card component to render.
        """
        base: dict[str, Any] = {
            "success": self.success,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "display_hint": self.display_hint,
            "affected_ids": self.affected_ids,
        }
        if self.data is not None:
            base["data"] = self.data
        if self.error is not None:
            base["error"] = {
                "category": self.error.category,
                "message": self.error.message,
                "retryable": self.error.retryable,
                "suggested_actions": self.error.suggested_actions,
            }
        return base

    @classmethod
    def ok(
        cls,
        tool_name: str,
        summary: str = "",
        details: str = "",
        data: Optional[dict] = None,
        display_hint: str = "text",
        affected_ids: Optional[list[str]] = None,
    ) -> "ToolResult":
        """Factory for successful results."""
        return cls(
            success=True,
            tool_name=tool_name,
            summary=summary or details[:80],
            details=details or summary,
            data=data,
            display_hint=display_hint,
            affected_ids=affected_ids or [],
        )

    @classmethod
    def fail(
        cls,
        tool_name: str,
        error: ErrorDetail,
        summary: str = "",
    ) -> "ToolResult":
        """Factory for failed results."""
        return cls(
            success=False,
            tool_name=tool_name,
            summary=summary or f"{tool_name} 执行失败",
            details=error.message,
            error=error,
            display_hint="text",
        )

    @classmethod
    def from_legacy_string(cls, tool_name: str, result_text: str) -> "ToolResult":
        """Wrap a legacy plain-text result for backward compatibility.

        Used while gradually migrating individual tool branches to ToolResult.
        """
        return cls(
            success=True,
            tool_name=tool_name,
            summary=result_text[:80].replace("\n", " "),
            details=result_text,
            display_hint="text",
        )


# ── Display hint constants ───────────────────────────────

class DisplayHint:
    """Constants for ToolResult.display_hint."""
    CARD = "card"              # Single entity card (candidate, position)
    LIST = "list"              # List of items
    TABLE = "table"            # Tabular data
    TEXT = "text"              # Plain text (fallback)
    SCORE = "score"            # Score/evaluation result
    QUESTIONS = "questions"    # Interview question list
    STATS = "stats"            # Statistics summary
    CONFIRM = "confirm"        # Requires user confirmation
