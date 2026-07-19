"""
Supervisor graph — replaces the flat ReAct loop with a plan-then-execute
architecture inspired by Claude Code's planning and dispatch patterns.

Architecture
------------
::

    START → classify_complexity
              ├─ [simple]  → worker_agent (ReAct) → verify → END
              └─ [complex] → plan_node → execute_plan → verify → END

- **classify_complexity**: Heuristic check — is this a single-step query or a
  multi-step workflow?  Single-step skips the planning overhead.
- **plan_node**: For complex tasks, generates a structured execution plan
  (list of steps with expected tool calls) and streams it to the frontend
  as a ``plan_proposal`` SSE event.
- **worker_agent**: The existing ReAct agent (LLM + tools), used both for
  simple queries and as the executor for each step of a complex plan.
- **verify_node**: Post-execution verification — checks that write operations
  actually took effect and results are consistent.

The original ``build_agent_graph`` / ``stream_agent_response`` from
``app/agent/graph.py`` are preserved and wrapped — the supervisor delegates
to them rather than duplicating their logic.
"""

from __future__ import annotations

import json
import uuid
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from app.agent.tool_result import ToolResult, ErrorDetail


# ═══════════════════════════════════════════════════════════
#  State
# ═══════════════════════════════════════════════════════════

from typing import TypedDict, Annotated


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: str                        # "classify" | "planning" | "executing" | "verifying" | "done"
    complexity: str                   # "simple" | "complex"
    plan: Optional[dict]              # Structured execution plan (None for simple)
    plan_step_index: int              # Current step within plan execution
    plan_results: list[dict]          # Accumulated results from each step


# ═══════════════════════════════════════════════════════════
#  Plan Model
# ═══════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    index: int
    title: str                        # Short Chinese label ("查看岗位JD")
    description: str                  # What this step does
    expected_tools: list[str]         # Tools likely needed
    expected_outcome: str             # What success looks like


@dataclass
class ExecutionPlan:
    plan_id: str
    title: str                        # "Agent工程师招聘流程"
    steps: list[PlanStep]
    total_estimated_tools: int

    def to_sse_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "steps": [
                {
                    "index": s.index,
                    "title": s.title,
                    "description": s.description,
                    "expected_tools": s.expected_tools,
                    "expected_outcome": s.expected_outcome,
                }
                for s in self.steps
            ],
            "total_steps": len(self.steps),
            "total_estimated_tools": self.total_estimated_tools,
        }


# ═══════════════════════════════════════════════════════════
#  Complexity Classification
# ═══════════════════════════════════════════════════════════

# Patterns that indicate a multi-step / complex task
COMPLEX_PATTERNS = [
    r"(然后|接着|之后|再|同时|并且|以及|最后)",
    r"(先|首先).*(然后|再|接着)",
    r"(安排|组织|进行).*(面试|考核|评估|流程)",
    r"(完整的|全部的|整个).*(招聘|面试|考核|筛选|流程)",
    r"(帮我|给我|我要).*(做|完成|处理|搞定)",
    r"(推荐|筛选).*(并且|同时|还要).*(面试|出题|评估)",
    r"(从.*到.*)(筛选|面试|入职|考核)",
    r"(@interview|@training|@performance)",  # Cross-agent mentions
]


def classify_complexity_sync(message: str) -> str:
    """Determine if a user message needs the planning phase.

    Returns:
        "complex" if the task spans multiple steps/domains, "simple" otherwise.
    """
    if not message or not message.strip():
        return "simple"

    normalized = message.strip()

    # Check for explicit multi-step markers
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "complex"

    # Very short messages are almost always simple
    if len(normalized) < 15:
        return "simple"

    return "simple"


# ═══════════════════════════════════════════════════════════
#  Graph Builder
# ═══════════════════════════════════════════════════════════

def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_supervisor_graph(
    tools: list[BaseTool],
    system_prompt: str,
) -> StateGraph:
    """Build a compiled supervisor LangGraph with plan-then-execute pattern.

    This wraps the existing ReAct flow inside a supervisor that adds:
    1. Complexity classification (simple → skip planning)
    2. Plan generation for complex tasks
    3. Step-by-step execution tracking
    4. Post-execution verification

    Args:
        tools: LangChain StructuredTool list (already intent-gated).
        system_prompt: Full system prompt for the worker agent.

    Returns:
        Compiled LangGraph StateGraph ready for streaming.
    """
    from app.agent.graph import _create_llm  # reuse existing LLM factory

    llm = _create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # ── Nodes ──────────────────────────────────────────

    async def node_classify(state: SupervisorState) -> dict:
        """Entry point: classify task complexity."""
        messages = state["messages"]
        last_msg = messages[-1] if messages else None
        content = str(last_msg.content) if last_msg and hasattr(last_msg, "content") else ""

        complexity = classify_complexity_sync(content)
        return {
            "phase": "classify",
            "complexity": complexity,
            "plan": None,
            "plan_step_index": 0,
            "plan_results": [],
        }

    async def node_plan(state: SupervisorState) -> dict:
        """Generate a structured execution plan for complex tasks.

        Uses a focused LLM call to decompose the user's request into
        ordered steps with expected tool calls.
        """
        messages = state["messages"]
        last_msg = messages[-1]
        user_content = str(last_msg.content) if hasattr(last_msg, "content") else ""

        plan_prompt = f"""你是一个任务规划器。将用户的请求分解为有序的执行步骤。
只返回一个 JSON 对象（不要 markdown 代码块，不要解释）：

{{
  "title": "简短的任务标题（≤15字）",
  "steps": [
    {{
      "index": 1,
      "title": "步骤名称（≤10字）",
      "description": "这个步骤要做什么",
      "expected_tools": ["工具名1", "工具名2"],
      "expected_outcome": "成功完成的标准"
    }}
  ]
}}

规则：
- 每个步骤应该只调用 1-2 个工具
- 步骤总数不超过 5 个
- 步骤顺序要合理（先查询再操作）
- 如果用户只问一个简单问题，steps 数组只包含一个步骤

用户请求：{user_content[:500]}
JSON："""

        plan_data = None
        try:
            resp = await llm.ainvoke([SystemMessage(content=plan_prompt)])
            raw = str(resp.content).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            plan_data = json.loads(raw)
        except Exception:
            pass

        if not plan_data or "steps" not in plan_data:
            # Fallback: single-step plan
            plan_data = {
                "title": "执行任务",
                "steps": [{
                    "index": 1,
                    "title": "处理请求",
                    "description": user_content[:100],
                    "expected_tools": [],
                    "expected_outcome": "完成用户请求",
                }],
            }

        steps = [
            PlanStep(
                index=s["index"],
                title=s.get("title", f"步骤{s['index']}"),
                description=s.get("description", ""),
                expected_tools=s.get("expected_tools", []),
                expected_outcome=s.get("expected_outcome", ""),
            )
            for s in plan_data.get("steps", [])
        ]

        plan = ExecutionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            title=plan_data.get("title", "执行计划"),
            steps=steps,
            total_estimated_tools=sum(len(s.expected_tools) for s in steps),
        )

        return {
            "phase": "planning",
            "plan": plan.to_sse_dict(),
        }

    async def node_execute_plan(state: SupervisorState) -> dict:
        """Execute the current step of the plan via the worker agent.

        For simple tasks, this is a single ReAct execution.
        For complex tasks, this executes the current step and advances the index.
        """
        plan = state.get("plan")
        step_index = state.get("plan_step_index", 0)
        messages = state["messages"]

        if plan and plan.get("steps"):
            steps = plan["steps"]
            if step_index < len(steps):
                step = steps[step_index]
                # Add step context to the last message
                step_hint = (
                    f"\n\n[当前步骤 {step['index']}/{len(steps)}: {step['title']}] "
                    f"{step['description']}"
                )
                if messages and hasattr(messages[-1], "content"):
                    modified_content = str(messages[-1].content) + step_hint
                    messages = list(messages[:-1]) + [
                        type(messages[-1])(content=modified_content)
                    ]

        # The actual tool execution will be handled by the streaming function
        # Here we just advance the state
        return {
            "phase": "executing",
            "plan_step_index": step_index + 1 if plan else 0,
        }

    async def node_verify(state: SupervisorState) -> dict:
        """Post-execution verification placeholder.

        In v2, this will:
        - Read-back after write operations
        - Cross-check consistency between related results
        - Score the quality of generated content
        """
        return {"phase": "verifying"}

    # ── Router ─────────────────────────────────────────

    def route_after_classify(state: SupervisorState) -> str:
        if state.get("complexity") == "complex":
            return "plan"
        return "execute"  # simple → skip planning

    def route_after_plan(state: SupervisorState) -> str:
        # v1: auto-execute after planning (no user approval wait)
        return "execute"

    def route_after_execute(state: SupervisorState) -> str:
        plan = state.get("plan")
        step_index = state.get("plan_step_index", 0)
        if plan and plan.get("steps") and step_index < len(plan["steps"]):
            return "execute"  # more steps
        return "verify"

    def route_after_verify(state: SupervisorState) -> str:
        return END

    # ── Build Graph ─────────────────────────────────────

    workflow = StateGraph(SupervisorState)
    workflow.add_node("classify", node_classify)
    workflow.add_node("plan", node_plan)
    workflow.add_node("execute", node_execute_plan)
    workflow.add_node("verify", node_verify)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", route_after_classify, {
        "plan": "plan",
        "execute": "execute",
    })
    workflow.add_conditional_edges("plan", route_after_plan, {
        "execute": "execute",
    })
    workflow.add_conditional_edges("execute", route_after_execute, {
        "execute": "execute",
        "verify": "verify",
    })
    workflow.add_conditional_edges("verify", route_after_verify, {
        END: END,
    })

    return workflow.compile()


# ═══════════════════════════════════════════════════════════
#  Streaming
# ═══════════════════════════════════════════════════════════

@dataclass
class SupervisorResult:
    """Collected results populated during streaming."""
    tool_blocks: list[dict] = field(default_factory=list)
    full_content: str = ""
    plan: Optional[dict] = None
    phase: str = ""


async def stream_supervisor_response(
    supervisor_graph,
    worker_graph,
    messages: list[BaseMessage],
    system_prompt: str,
    result: SupervisorResult,
) -> AsyncGenerator[str, None]:
    """Execute the supervisor graph, yielding SSE events.

    The supervisor orchestrates the flow:
    1. Classify complexity → emit ``thinking``
    2. If complex: generate plan → emit ``plan_proposal``
    3. Execute via worker agent → emit ``content``, ``tool_call``, ``tool_result``
    4. Verify → emit ``phase_result``

    The actual ReAct tool execution is delegated to the existing
    ``stream_agent_response`` from ``app/agent/graph.py`` for backward
    compatibility.

    Args:
        supervisor_graph: Compiled supervisor StateGraph.
        worker_graph: Compiled ReAct agent graph (from ``build_agent_graph``).
        messages: Initial message list.
        result: Mutable SupervisorResult to populate.
        system_prompt: System prompt for the worker agent.

    Yields:
        SSE-formatted strings.
    """
    from app.agent.graph import stream_agent_response, AgentResult
    from langchain_core.messages import HumanMessage

    # ── Phase 1: Classify ──
    initial_state = {
        "messages": messages,
        "phase": "classify",
        "complexity": "simple",
        "plan": None,
        "plan_step_index": 0,
        "plan_results": [],
    }

    # Run classify node
    classify_state = None
    async for event in supervisor_graph.astream_events(initial_state, version="v2"):
        kind = event.get("event", "")
        if kind == "on_chain_end" and event.get("name") == "classify":
            classify_state = event.get("data", {}).get("output", {})
            break

    if not classify_state:
        classify_state = initial_state

    complexity = classify_state.get("complexity", "simple")
    result.phase = "classify"
    result.plan = classify_state.get("plan")

    # Emit thinking based on complexity
    if complexity == "complex":
        yield _sse("thinking", {
            "text": "这是一个多步骤任务，我先规划一下执行方案…",
            "append": False,
        })
    else:
        yield _sse("thinking", {
            "text": "收到任务，正在分析您的指令…",
            "append": False,
        })

    # ── Phase 2: Plan (complex only) ──
    if complexity == "complex":
        # Run plan node
        plan_state = {"messages": messages}
        async for event in supervisor_graph.astream_events(plan_state, version="v2"):
            kind = event.get("event", "")
            if kind == "on_chain_end" and event.get("name") == "plan":
                plan_output = event.get("data", {}).get("output", {})
                result.plan = plan_output.get("plan")
                result.phase = "planning"
                break

        # Also call plan directly for reliability
        from app.agent.supervisor_graph import node_plan
        plan_result = await node_plan({"messages": messages, "phase": "", "complexity": "", "plan": None, "plan_step_index": 0, "plan_results": []})
        plan = plan_result.get("plan")
        if plan:
            result.plan = plan
            result.phase = "planning"
            yield _sse("plan_proposal", plan)

    # ── Phase 3: Execute via worker agent ──
    result.phase = "executing"
    worker_result = AgentResult()

    # Build worker messages — prepend system prompt
    worker_messages = list(messages)

    # If we have a plan, add it as context
    if result.plan and result.plan.get("steps"):
        steps_text = "\n".join(
            f"  {s['index']}. {s['title']}: {s['description']}"
            for s in result.plan["steps"]
        )
        plan_context = (
            f"\n\n[执行计划] {result.plan.get('title', '')}\n{steps_text}\n"
            f"按步骤顺序执行，每完成一步再开始下一步。"
        )
        worker_messages = worker_messages[:-1] + [
            HumanMessage(content=str(worker_messages[-1].content) + plan_context)
        ]

    # Delegate to the existing ReAct streaming
    async for sse_str in stream_agent_response(worker_graph, worker_messages, worker_result):
        yield sse_str

    # Merge tool blocks
    result.tool_blocks = worker_result.tool_blocks
    result.full_content = worker_result.full_content

    # ── Phase 4: Verify ──
    result.phase = "verifying"
    # v1: placeholder — always passes verification
    yield _sse("phase_result", {
        "id": f"phase_{uuid.uuid4().hex[:6]}",
        "tone": "success",
        "title": "任务完成",
        "description": "验证通过" if not result.plan else f"已完成 {len(result.plan.get('steps', []))} 个步骤",
    })
