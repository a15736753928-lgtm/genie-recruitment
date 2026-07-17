"""
LangGraph ReAct agent for the Genie recruitment system.

Replaces the manual two-phase tool-calling pattern with a proper
LangGraph state graph: agent (LLM) ⇄ tools → END.

Streaming is exposed via an async generator that yields the same
SSE-formatted events the frontend expects.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from typing import TypedDict, Annotated

from app.config import get_settings

settings = get_settings()


# ── State ───────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Result container ────────────────────────────────────

@dataclass
class AgentResult:
    """Collected results populated during streaming."""
    tool_blocks: list[dict] = field(default_factory=list)
    full_content: str = ""


# ── LLM Factory ─────────────────────────────────────────

def _create_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Create a ChatOpenAI pointed at DeepSeek's OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=True,
        timeout=60,
        max_retries=0,
    )


# ── Graph Builder ───────────────────────────────────────

def build_agent_graph(
    tools: list[BaseTool],
    system_prompt: str,
) -> StateGraph:
    """Build a compiled LangGraph ReAct agent.

    Graph topology:
        START → agent ──[tool_calls?]──→ tools → agent
                     ──[no tools]─────→ END

    The agent node prepends the system prompt automatically.
    """
    llm = _create_llm()
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        # Must use ainvoke (not invoke) so the event loop is not blocked
        # while waiting for the LLM. A synchronous invoke here freezes
        # the entire async server, which is why the frontend stops
        # receiving data while any AI task is running.
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ── SSE Helpers ─────────────────────────────────────────

def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Streaming ───────────────────────────────────────────

async def stream_agent_response(
    graph,
    messages: list[BaseMessage],
    result: AgentResult,
) -> AsyncGenerator[str, None]:
    """Execute the agent graph, yielding SSE-formatted events.

    Populates ``result`` (an AgentResult) with ``tool_blocks`` and
    ``full_content`` as the agent runs.  Read them after the generator
    is exhausted.

    Args:
        graph: Compiled LangGraph graph.
        messages: Initial message list (system prompt NOT included —
                  the agent node will prepend it automatically).
        result: Mutable AgentResult to populate during streaming.

    Yields:
        SSE-formatted strings ready for ``StreamingResponse``.
    """
    async for event in graph.astream_events(
        {"messages": messages},
        version="v2",
    ):
        kind = event.get("event", "")

        # ── LLM token streaming ──
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                delta = chunk.content
                if isinstance(delta, str) and delta:
                    result.full_content += delta
                    yield _sse("content", {"delta": delta})

        # ── Tool call started ──
        elif kind == "on_tool_start":
            tool_name = event.get("name", "")
            raw_input = event.get("data", {}).get("input", {})
            tool_input = raw_input if isinstance(raw_input, dict) else {}

            tool_id = f"tool_{uuid.uuid4().hex[:8]}"
            result.tool_blocks.append({
                "id": tool_id,
                "name": tool_name,
                "params": tool_input,
                "result": "",
                "status": "running",
            })

            yield _sse("tool_call", {
                "id": tool_id,
                "name": tool_name,
                "params": tool_input,
            })

            yield _sse("task_progress", {
                "title": f"正在执行 {tool_name}",
                "description": str(tool_input)[:100],
                "progress": 30,
                "elapsed": "执行中...",
            })

        # ── Tool call completed ──
        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            result_text = str(output.content) if hasattr(output, "content") else str(output)

            # Update the most recent running block
            for block in reversed(result.tool_blocks):
                if block["status"] == "running":
                    block["result"] = result_text
                    block["status"] = "done"
                    yield _sse("tool_result", {
                        "id": block["id"],
                        "result": result_text,
                    })
                    break

    # Note: the caller (router) is responsible for yielding "done" and "phase_result"
    # after this generator is exhausted, so those events are emitted in the correct order.
