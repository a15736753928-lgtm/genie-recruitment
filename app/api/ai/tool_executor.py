"""Tool executor — translates LLM tool calls into API calls and formats results.

Each tool returns a human-readable text summary that the LLM uses to
formulate its response to the user.  Separated from agent_chat.py so the
600-line if/elif chain doesn't obscure the API plumbing.
"""

from __future__ import annotations

import json
import logging
import contextvars
from sqlalchemy.ext.asyncio import AsyncSession
from app.middleware.trace import get_trace_id
from app.core.security import CurrentUser

logger = logging.getLogger(__name__)


# 当前请求的用户主体，供工具 handler 内做写操作二次权限校验（防御纵深）。
# 由 app.agent.tools._execute_tool_sync 在执行工具前 set，handler 内通过
# get_current_user_for_tools() 读取。走 quality_guard 验证路径
# （_verify_executor 不传 current_user）时为 None → 校验跳过（验证只读不改）。
_current_user_ctx: contextvars.ContextVar[CurrentUser | None] = contextvars.ContextVar(
    "ai_tool_current_user", default=None
)


def set_current_user_for_tools(user: CurrentUser | None):
    """注入当前用户到 contextvar，返回 reset token。调用方在 finally 用
    ``_current_user_ctx.reset(token)`` 还原，避免跨请求串号。"""
    return _current_user_ctx.set(user)


def get_current_user_for_tools() -> CurrentUser | None:
    """工具 handler 内读取当前用户。未注入（验证路径/旧调用）返回 None。"""
    return _current_user_ctx.get()


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

        # 防御纵深：即便可见集过滤（get_tools_for_agent）被绕过——比如未来新增
        # 工具忘在 TOOL_PERMISSIONS 登记、或边界 bug——写/敏感工具在此二次校验。
        # current_user 由 _execute_tool_sync 经 contextvar 注入；quality_guard
        # 验证路径不传 current_user → contextvar 为 None → 跳过（验证只读不改）。
        cu = get_current_user_for_tools()
        if cu is not None:
            from app.agent.tools import TOOL_PERMISSIONS
            perm = TOOL_PERMISSIONS.get(tool_name)
            if perm and not cu.has(perm):
                logger.info(
                    "工具权限拒绝 tool=%s user=%s perm=%s trace=%s",
                    tool_name, cu.username, perm, get_trace_id(),
                )
                return (
                    f"您没有执行该操作的权限（需要 {perm}）。"
                    "如需处理请联系有对应权限的同事。"
                )

        registry = get_registry()
        handler = registry.get(tool_name)
        if handler is not None:
            return await handler(params, db)

        return f"工具 {tool_name} 执行完成"

    except Exception as e:
        # 异常在这里被转成返回值，调用方 _execute_tool_sync 的 except 分支永远进不来，
        # 会无条件 commit 掉 handler 崩溃前写了一半的数据。必须在此就地回滚。
        await db.rollback()
        trace_id = get_trace_id()
        logger.warning("Tool execution error tool=%s trace=%s: %s", tool_name, trace_id, e)
        return f"工具执行失败: {str(e)}"
