"""LangGraph checkpointer 单例。

AI 对话工作台每次请求都重新 build_agent_graph（agent_chat.py），若每次 new
一个 saver，interrupt 挂起的 thread 在 resume 时找不到。因此必须用模块级单例。

当前用 InMemorySaver：项目单进程 uvicorn（main.py 无 workers），并发都在同一
进程内，零依赖立即可用。局限：进程重启（dev reload / prod 重启）会丢挂起状态，
由超时清理任务兜底（见 agent_chat 的 confirm 过期扫描）。需要持久化时替换为
AsyncPostgresSaver（asyncpg 已装），改动集中在本文件一处。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

_saver: "BaseCheckpointSaver | None" = None


def get_agent_checkpointer() -> "BaseCheckpointSaver":
    """返回模块级 checkpointer 单例（懒初始化）。"""
    global _saver
    if _saver is None:
        from langgraph.checkpoint.memory import InMemorySaver

        _saver = InMemorySaver()
    return _saver
