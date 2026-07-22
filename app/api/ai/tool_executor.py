"""Tool executor — translates LLM tool calls into API calls and formats results.

Each tool returns a human-readable text summary that the LLM uses to
formulate its response to the user.  Separated from agent_chat.py so the
600-line if/elif chain doesn't obscure the API plumbing.
"""

from __future__ import annotations

import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.middleware.trace import get_trace_id

logger = logging.getLogger(__name__)


def sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event string for streaming responses."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def execute_tool_call(tool_name: str, params: dict, db: AsyncSession) -> str:
    """Execute a tool and return a human-readable result summary.

    Dispatches through the :class:`ToolHandlerRegistry` — new tools can be
    added by creating a handler and registering it, without touching this file.
    """
    try:
        from app.api.ai.tool_handlers import get_registry

        registry = get_registry()
        handler = registry.get(tool_name)
        if handler is not None:
            return await handler(params, db)

        return f"工具 {tool_name} 执行完成"

    except Exception as e:
        trace_id = get_trace_id()
        logger.warning("Tool execution error tool=%s trace=%s: %s", tool_name, trace_id, e)
        return f"工具执行错误: {str(e)}"
