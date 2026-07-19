"""
Structured output schemas — typed tool results and errors.

Replaces bare-string tool results with structured Pydantic models so the
frontend can render cards/tables/lists without parsing free-form text.
"""

from __future__ import annotations

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """Structured error from a tool execution."""

    category: Literal[
        "not_found",
        "permission_denied",
        "validation_error",
        "external_error",
        "timeout",
        "unknown",
    ] = "unknown"
    message: str = ""
    retryable: bool = False
    suggested_actions: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Unified tool result — used by all tools.

    The frontend uses ``display_hint`` to choose the render component
    (card, list, table, chart, etc.) and ``data`` for the payload.
    """

    success: bool = True
    tool_name: str = ""
    summary: str = ""                           # human-readable one-liner
    display_hint: Literal[
        "card", "list", "table", "text",
        "score", "questions", "stats", "confirm",
    ] = "text"
    data: Optional[dict[str, Any]] = None       # structured payload for frontend
    error: Optional[ToolError] = None
    tokens_used: int = 0

    # ── Factory methods ───────────────────────────────────

    @classmethod
    def ok(
        cls,
        tool_name: str,
        summary: str,
        hint: str = "text",
        data: Optional[dict] = None,
    ) -> "ToolResult":
        return cls(
            success=True,
            tool_name=tool_name,
            summary=summary,
            display_hint=hint,
            data=data,
        )

    @classmethod
    def fail(
        cls,
        tool_name: str,
        message: str,
        category: str = "unknown",
        retryable: bool = True,
        suggested_actions: Optional[list[str]] = None,
    ) -> "ToolResult":
        return cls(
            success=False,
            tool_name=tool_name,
            summary=message,
            display_hint="text",
            error=ToolError(
                category=category,
                message=message,
                retryable=retryable,
                suggested_actions=suggested_actions or [],
            ),
        )

    @classmethod
    def from_legacy_string(cls, tool_name: str, text: str) -> "ToolResult":
        """Convert a legacy string result into a ToolResult.

        Used for backward compatibility during the migration.
        Attempts to detect the display hint from the tool name.
        """
        hint = _infer_hint(tool_name)
        return cls(
            success=True,
            tool_name=tool_name,
            summary=text[:200],
            display_hint=hint,
            data={"raw_text": text},
        )

    # ── Serialization ─────────────────────────────────────

    def to_sse_dict(self) -> dict:
        """Render as a dict suitable for SSE ``tool_result`` events."""
        d = {
            "success": self.success,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "display_hint": self.display_hint,
        }
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error.model_dump()
        return d


# ── Hint inference ─────────────────────────────────────────

def _infer_hint(tool_name: str) -> str:
    """Guess the best display hint from the tool name."""
    HINT_MAP = {
        "get_resume": "card",
        "get_position": "card",
        "list_resumes": "list",
        "list_positions": "list",
        "get_questions": "questions",
        "generate_questions": "questions",
        "get_evaluation": "score",
        "save_evaluation": "score",
        "ai_score_question": "score",
        "get_leaderboard": "table",
        "get_rankings": "table",
        "get_probation_stats": "stats",
        "get_performance_stats": "stats",
        "get_dashboard_overview": "stats",
        "get_operations_dashboard": "stats",
    }
    return HINT_MAP.get(tool_name, "text")
