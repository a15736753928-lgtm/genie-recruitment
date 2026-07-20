"""
Agent Loop — async generator state machine (Claude Code architecture).

Replaces LangGraph ReAct with a native Python async generator loop.
The loop is driven by immutable state assignment (`state = next`),
not recursion — guaranteeing stack safety during long sessions.

Key components:
- ``LoopState``: immutable state object, full assignment at each continue
- ``ContinueReason``: typed enum for WHY the loop continues (7 variants)
- ``ExitReason``: typed enum for WHY the loop exits (12 variants)
- ``StreamingToolExecutor``: executes tools DURING model streaming
- ``query_loop``: the main async generator state machine
"""

from app.agent_os.loop.state import (
    LoopState,
    ContinueReason,
    ExitReason,
    QueryContext,
    LoopResult,
    TurnMetrics,
)
from app.agent_os.loop.streaming_executor import (
    StreamingToolExecutor,
    ToolExecutionState,
    ToolBlock,
)
from app.agent_os.loop.query_loop import (
    query_loop,
    query,
    QueryEngine,
)

__all__ = [
    "LoopState",
    "ContinueReason",
    "ExitReason",
    "QueryContext",
    "LoopResult",
    "TurnMetrics",
    "StreamingToolExecutor",
    "ToolExecutionState",
    "ToolBlock",
    "query_loop",
    "query",
    "QueryEngine",
]
