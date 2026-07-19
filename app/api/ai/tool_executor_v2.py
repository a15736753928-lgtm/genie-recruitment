"""
Structured tool executor — wraps the legacy string-based executor.

Provides ``execute_tool_call_structured`` which always returns a
``ToolResult`` object with proper display hints and data extraction.
Used by the v2 Agent OS chat endpoint for enhanced frontend rendering.

The legacy ``execute_tool_call`` in tool_executor.py is preserved as-is
for backward compatibility — this module adds the structured layer on top.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.ai.tool_executor import execute_tool_call as _legacy_execute
from app.agent_os.output.adapter import ToolResultAdapter
from app.agent_os.output.schemas import ToolResult

# Singleton adapter instance
_adapter = ToolResultAdapter()


async def execute_tool_call_structured(
    tool_name: str,
    params: dict,
    db: AsyncSession,
) -> ToolResult:
    """Execute a tool and return a structured ToolResult.

    Wraps the legacy string-based executor with automatic display hint
    detection, structured data extraction, and error classification.

    Args:
        tool_name: Tool name from TOOL_REGISTRY.
        params: Tool parameters.
        db: Active SQLAlchemy async session.

    Returns:
        ToolResult with success/error status, display hint, and
        structured data for frontend rendering.
    """
    return await _adapter.execute(
        tool_name=tool_name,
        params=params,
        legacy_executor=_legacy_execute,
        db=db,
    )
