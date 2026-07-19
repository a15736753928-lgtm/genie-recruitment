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
from app.agent.tool_result import ToolResult, DisplayHint

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


# ── Tool Metadata ───────────────────────────────────────

# Progress percentages by tool category for differentiated feedback
_TOOL_PROGRESS: dict[str, int] = {
    # Fast reads
    "list_resumes": 70, "get_resume": 80, "list_positions": 80, "get_position": 80,
    "get_questions": 80, "get_evaluation": 80, "get_leaderboard": 80,
    "get_rankings": 80, "get_settings": 90, "get_operations_dashboard": 80,
    "get_dashboard_overview": 80, "get_knowledge_stats": 90,
    "get_knowledge_categories": 90, "get_probation_stats": 80,
    "get_performance_stats": 80, "get_department_performance": 80,
    "get_grade_distribution": 80, "get_bonus_info": 80,
    "get_quarter_trends": 80, "list_knowledge": 80, "list_knowledge_bases": 80,
    "list_documents": 80, "list_probation": 80, "list_performance": 80,
    "get_position_questions": 80, "get_probation_employee": 80,
    # Medium (writes / moderate latency)
    "update_resume": 40, "update_position": 40, "save_questions": 40,
    "save_evaluation": 40, "save_position_questions": 40,
    "update_settings": 50, "create_position": 50, "create_knowledge_item": 50,
    "create_knowledge_base": 50, "create_probation_employee": 50,
    "create_probation_task": 50, "update_knowledge_item": 50,
    "update_knowledge_base": 50, "update_probation_task": 50,
    # Slow (AI generation / file ops)
    "generate_questions": 15, "ai_score_question": 10,
    "ai_evaluate_probation": 10, "rag_search": 20,
    "upload_resume": 10, "upload_knowledge_file": 10,
    "upload_document": 10, "reanalyze_resume": 10,
    "batch_parse_resumes": 5, "recall_test": 20,
    "replace_question": 15,
    # Destructive (fast but needs care)
    "delete_resume": 90, "delete_position": 90, "delete_knowledge_item": 90,
    "delete_knowledge_base": 90, "delete_document": 90,
    "submit_evaluation": 80, "initiate_appraisal": 50,
}


def _tool_progress(tool_name: str) -> int:
    """Return a reasonable progress percentage for a tool's start event."""
    return _TOOL_PROGRESS.get(tool_name, 50)


def _display_hint_for(tool_name: str) -> str:
    """Map tool name to display hint for frontend card rendering."""
    if tool_name in ("list_resumes",):
        return "list"
    if tool_name in ("get_resume",):
        return "card"
    if tool_name in ("get_position", "list_positions"):
        return "card"
    if tool_name in ("get_questions", "generate_questions"):
        return "questions"
    if tool_name in ("get_evaluation", "save_evaluation", "ai_score_question"):
        return "score"
    if tool_name in ("get_leaderboard", "get_rankings"):
        return "table"
    if tool_name in ("rag_search", "recall_test"):
        return "list"
    return "text"


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
                "progress": _tool_progress(tool_name),
                "elapsed": "执行中...",
            })

        # ── Tool call completed ──
        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            result_text = str(output.content) if hasattr(output, "content") else str(output)

            # Wrap in ToolResult for structured SSE
            tr = ToolResult.from_legacy_string(tool_name, result_text)
            # Detect display hint from tool name
            tr.display_hint = _display_hint_for(tool_name)
            sse_data = tr.to_sse_dict()

            # Update the most recent running block
            for block in reversed(result.tool_blocks):
                if block["status"] == "running":
                    block["result"] = result_text
                    block["status"] = "done"
                    yield _sse("tool_result", {
                        "id": block["id"],
                        "result": result_text,
                        **sse_data,
                    })
                    break

    # Note: the caller (router) is responsible for yielding "done" and "phase_result"
    # after this generator is exhausted, so those events are emitted in the correct order.
