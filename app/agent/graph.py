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
from typing import Any, AsyncGenerator, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.tool_result import ToolResult, DisplayHint
from app.agent.checkpointer import get_agent_checkpointer
from app.agent_os.output import DISPLAY_HINTS
from app.services.ai import create_langchain_llm


# ── State ───────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 本轮 tools 节点的原始返回（含 ToolResult）。覆盖写、无 reducer——否则每轮
    # 累积，resume 后 confirm_gate 会撞到旧 confirmation 再次 interrupt。
    last_tool_results: list[Any]


# ── Result container ────────────────────────────────────

@dataclass
class AgentResult:
    """Collected results populated during streaming."""
    tool_blocks: list[dict] = field(default_factory=list)
    full_content: str = ""
    # 本轮流在 confirm_gate 处被 interrupt 挂起（等待用户决策）。
    interrupted: bool = False
    interrupt_payload: dict | None = None


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
    _model_temperature = 0.7
    llm = create_langchain_llm(streaming=True, temperature=_model_temperature)
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception:
            # Provider failover：当前 key 调用失败时，用轮询的下一个 key 重建 LLM
            # 重试一次（create_langchain_llm 内部会切到下一个 provider/key）。
            # 仅重试一次，避免在长时间断连场景下叠加 SDK 层重试导致 60+ 秒超时。
            retry_llm = create_langchain_llm(streaming=True, temperature=_model_temperature)
            response = await retry_llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    async def tools_node(state: AgentState) -> dict:
        """自定义工具节点：逐个执行工具，保留原始返回（含 ToolResult）供 confirm_gate 检测。

        复刻 ToolNode 的 ToolMessage 配对约定——tool_call_id 必须与 AIMessage 的
        tool_calls[].id 一致，否则 OpenAI 兼容接口把孤儿 ToolMessage 拒成 400。
        """
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        tool_by_name = {t.name: t for t in tools}
        raw_results: list[Any] = []
        tool_messages: list[ToolMessage] = []
        for tc in tool_calls:
            tool = tool_by_name.get(tc["name"])
            if tool is None:
                raw_results.append(None)
                tool_messages.append(ToolMessage(
                    content=f"工具 {tc['name']} 不存在",
                    tool_call_id=tc.get("id", ""),
                ))
                continue
            try:
                out = await tool.ainvoke(tc.get("args") or {})
            except Exception as e:
                out = f"工具执行失败: {str(e)}"
            # last_tool_results 只保留「需用户决策」的确认信息（纯 dict），不把
            # ToolResult 对象本身写进 state/checkpoint：msgpack 对未注册类型
            # 序列化会告警（未来版本直接 block），且减小 checkpoint 体积。
            raw_results.append(
                out.confirmation
                if isinstance(out, ToolResult) and out.confirmation
                else None
            )
            if isinstance(out, ToolResult):
                content = out.to_llm_context()
            else:
                content = str(out)
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc.get("id", "")))
        return {"messages": tool_messages, "last_tool_results": raw_results}

    async def confirm_gate(state: AgentState) -> dict:
        """检测本轮工具结果是否有「需用户决策」的确认；有则 interrupt 挂起，resume 后把
        用户选择注入为一条 HumanMessage，让模型据此继续（如先 update_resume 再出题）。"""
        for r in state.get("last_tool_results", []):
            if isinstance(r, dict) and r.get("kind"):
                confirm = r
                # 首次调用抛 GraphInterrupt 挂起；resume 时返回用户选择，从这行继续。
                choice = interrupt(confirm)
                candidate_id = confirm.get("candidate_id", "")
                candidate_name = confirm.get("candidate_name", "")
                round_label = confirm.get("round", "")
                return {"messages": [HumanMessage(content=(
                    f"[系统] 用户确认：{choice}。请按此继续执行："
                    f"若确认邀约一面，先调用 update_resume 把候选人 {candidate_id}"
                    f"（{candidate_name}）状态改为 round1，再调用 generate_questions 生成{round_label}"
                    f"面试题；若已执行成功则勿重复。若用户取消，则直接回复已取消，"
                    f"不要再调用任何写工具。"
                ))]}
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tools_node)
    workflow.add_node("confirm_gate", confirm_gate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "confirm_gate")
    workflow.add_edge("confirm_gate", "agent")

    return workflow.compile(checkpointer=get_agent_checkpointer())


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
    "get_probation_stats": 80,
    "get_performance_stats": 80, "get_department_performance": 80,
    "get_grade_distribution": 80, "get_bonus_info": 80,
    "get_quarter_trends": 80, "list_probation": 80, "list_performance": 80,
    "get_position_questions": 80, "get_probation_employee": 80,
    # Medium (writes / moderate latency)
    "update_resume": 40, "update_position": 40, "save_questions": 40,
    "save_evaluation": 40, "save_position_questions": 40,
    "update_settings": 50, "create_position": 50, "create_probation_employee": 50,
    "create_probation_task": 50, "update_probation_task": 50,
    # Slow (AI generation / file ops)
    "generate_questions": 15, "ai_score_question": 10,
    "ai_evaluate_probation": 10,
    "upload_resume": 10,
    "reanalyze_resume": 10,
    "batch_parse_resumes": 5,
    "replace_question": 15,
    # Destructive (fast but needs care)
    "delete_resume": 90, "delete_position": 90,
    "submit_evaluation": 80, "initiate_appraisal": 50,
}


def _tool_progress(tool_name: str) -> int:
    """Return a reasonable progress percentage for a tool's start event."""
    return _TOOL_PROGRESS.get(tool_name, 50)


def _display_hint_for(tool_name: str) -> str:
    """Map tool name to display hint for frontend card rendering."""
    return DISPLAY_HINTS.get(tool_name, "text")


# ── Streaming ───────────────────────────────────────────

async def stream_agent_response(
    graph,
    messages: list[BaseMessage] | None,
    result: AgentResult,
    thread_id: str,
    resume_value: Any = None,
) -> AsyncGenerator[str, None]:
    """Execute the agent graph, yielding SSE-formatted events.

    Populates ``result`` (an AgentResult) with ``tool_blocks`` and
    ``full_content`` as the agent runs.  Read them after the generator
    is exhausted.

    Args:
        graph: Compiled LangGraph graph.
        messages: Initial message list (system prompt NOT included —
                  the agent node will prepend it automatically).  For a
                  resume run, pass ``None`` — history is restored from the
                  checkpoint via ``thread_id``.
        result: Mutable AgentResult to populate during streaming.
        thread_id: LangGraph checkpoint thread id (= session id).  Required
                  so interrupt 挂起 / resume 能定位到同一线程状态。
        resume_value: 非空表示这是一次 resume（用户对挂起确认的选择），
                  从暂停点继续执行而非开启新对话。

    Yields:
        SSE-formatted strings ready for ``StreamingResponse``.
    """
    config = {
        "recursion_limit": 40,
        "configurable": {"thread_id": thread_id},
    }
    if resume_value is not None:
        inputs = Command(resume=resume_value)  # checkpoint 恢复历史，从暂停点继续
    else:
        inputs = {"messages": messages}

    async for event in graph.astream_events(inputs, version="v2", config=config):
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
                # run_id uniquely ties each on_tool_end back to the
                # on_tool_start that spawned it — without it, two in-flight
                # tools can get their results swapped.
                "_run_id": event.get("run_id"),
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
            raw_text = str(output.content) if hasattr(output, "content") else str(output)
            tool_name = event.get("name", "")

            # Try to wrap in structured ToolResult
            tr = None
            if isinstance(output, ToolResult):
                tr = output
            else:
                tr = ToolResult.from_legacy_string(tool_name, raw_text)

            # 「需用户决策」的确认结果保留 handler 设置的 CONFIRM，不被工具表覆盖。
            if tr.confirmation is None:
                tr.display_hint = _display_hint_for(tool_name)
            sse_data = tr.to_sse_dict()

            # Match this completion back to its own tool_start via run_id
            # (falls back to "most recent running block" only if run_id is
            # unavailable, e.g. an older LangGraph version).
            end_run_id = event.get("run_id")
            candidates = reversed(result.tool_blocks)
            if end_run_id is not None:
                matched = [b for b in result.tool_blocks if b.get("_run_id") == end_run_id and b["status"] == "running"]
                candidates = matched if matched else candidates
            for block in candidates:
                if block["status"] == "running":
                    block["result"] = raw_text
                    block["status"] = "done"
                    # Attach structured fields for frontend rendering
                    block["success"] = tr.success
                    block["summary"] = tr.summary
                    block["display_hint"] = tr.display_hint
                    if tr.data is not None:
                        block["data"] = tr.data
                    if tr.confirmation is not None:
                        block["confirmation"] = tr.confirmation
                    if tr.error:
                        # ErrorDetail 是 dataclass（无 model_dump），用 asdict 序列化
                        from dataclasses import asdict
                        block["error"] = asdict(tr.error)
                    yield _sse("tool_result", {
                        "id": block["id"],
                        "result": raw_text,
                        **sse_data,
                    })
                    break

        # ── 用户决策挂起（interrupt）──
        # interrupt 在根图的 on_chain_stream 事件以 __interrupt__ chunk 冒泡，
        # on_chain_end 拿不到。检测到即发 ask_user 事件并优雅结束生成器
        # （checkpoint 已保存挂起状态，resume 时从暂停点继续）。
        elif kind == "on_chain_stream" and event.get("name") == "LangGraph":
            chunk = event.get("data", {}).get("chunk")
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                interrupts = chunk.get("__interrupt__") or ()
                intr = interrupts[0] if interrupts else None
                if intr is not None:
                    payload = getattr(intr, "value", None) or {}
                    result.interrupted = True
                    result.interrupt_payload = payload
                    yield _sse("ask_user", {
                        "sessionId": thread_id,
                        "kind": payload.get("kind"),
                        "question": payload.get("question"),
                        "options": payload.get("options", []),
                        "interruptId": getattr(intr, "id", ""),
                        "context": payload,
                    })
                    return

    # Note: the caller (router) is responsible for yielding "done" and "phase_result"
    # after this generator is exhausted, so those events are emitted in the correct order.
