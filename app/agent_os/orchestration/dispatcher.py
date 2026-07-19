"""
SubAgentDispatcher — spawn sub-agents with isolated contexts.

Each sub-agent runs in a completely independent context (no parent
conversation history).  Only the final structured summary returns to
the parent agent.  This is the "sidechain" pattern from Claude Code —
prevents context explosion and cross-contamination between parallel tasks.

Architecture::

    Main Agent                      Sub Agent 1
      [full context]                 [blank context + task]
          │                                │
          ├── dispatch ──────────────→  execute(task)
          │                                │
          │  ←── summary ────────────  return structured output
          │
      [ctx + summary]                 [context destroyed]
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@dataclass
class SubAgentTask:
    """A task dispatched to a sub-agent."""

    task_id: str = field(default_factory=lambda: f"subtask_{uuid.uuid4().hex[:8]}")
    agent_type: str = "genie"              # "recruit" | "interview" | "training" | "performance"
    task_description: str = ""             # what to do (becomes the user message)
    system_prompt_extra: str = ""          # additional instructions for this task
    tools: Optional[list[str]] = None      # restricted tool list (None = all for agent_type)
    structured_output_schema: Optional[dict] = None  # JSON schema the output must follow
    context_data: dict = field(default_factory=dict)  # extra data injected as context
    max_iterations: int = 10               # max ReAct loop iterations


@dataclass
class SubAgentResult:
    """Result returned by a sub-agent."""

    task_id: str
    success: bool
    summary: str                           # human-readable summary
    structured_data: Optional[dict] = None # if schema was provided
    tool_calls_made: int = 0
    error: Optional[str] = None
    tokens_used: int = 0                   # estimated


class SubAgentDispatcher:
    """Dispatches tasks to sub-agents with context isolation.

    Sub-agents are NOT full LangGraph instances — they use a lightweight
    async LLM loop to keep overhead low.  Each sub-agent:
    - Starts with a clean context (system prompt + task description only)
    - Can call a restricted set of tools
    - Returns a structured summary
    - Context is destroyed after completion
    """

    def __init__(self, max_concurrent: int = 10):
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ── Public API ────────────────────────────────────────

    async def dispatch(
        self,
        tasks: list[SubAgentTask],
        llm_client,
        tool_executor,           # async (tool_name, params, db) -> str
        db_factory,              # async_session_factory
        parallel: bool = True,
    ) -> list[SubAgentResult]:
        """Dispatch tasks to sub-agents.

        Args:
            tasks: List of SubAgentTask to execute.
            llm_client: AsyncOpenAI client.
            tool_executor: Tool execution function.
            db_factory: Async session factory.
            parallel: If True, execute tasks concurrently (up to max_concurrent).
                      If False, execute sequentially.

        Returns:
            List of SubAgentResult, one per task (same order).
        """
        if parallel and len(tasks) > 1:
            return await self._dispatch_parallel(
                tasks, llm_client, tool_executor, db_factory
            )
        else:
            return await self._dispatch_sequential(
                tasks, llm_client, tool_executor, db_factory
            )

    async def dispatch_single(
        self,
        task: SubAgentTask,
        llm_client,
        tool_executor,
        db_factory,
    ) -> SubAgentResult:
        """Dispatch a single task and return its result."""
        results = await self._dispatch_sequential(
            [task], llm_client, tool_executor, db_factory
        )
        return results[0]

    # ── Implementation ────────────────────────────────────

    async def _dispatch_parallel(
        self,
        tasks: list[SubAgentTask],
        llm_client,
        tool_executor,
        db_factory,
    ) -> list[SubAgentResult]:
        """Execute multiple tasks concurrently with a semaphore cap."""

        async def _run_one(task: SubAgentTask) -> SubAgentResult:
            async with self._semaphore:
                return await self._execute_subagent(
                    task, llm_client, tool_executor, db_factory
                )

        coros = [_run_one(t) for t in tasks]
        return await asyncio.gather(*coros)

    async def _dispatch_sequential(
        self,
        tasks: list[SubAgentTask],
        llm_client,
        tool_executor,
        db_factory,
    ) -> list[SubAgentResult]:
        """Execute tasks one at a time."""
        results = []
        for task in tasks:
            result = await self._execute_subagent(
                task, llm_client, tool_executor, db_factory
            )
            results.append(result)
        return results

    async def _execute_subagent(
        self,
        task: SubAgentTask,
        llm_client,
        tool_executor,
        db_factory,
    ) -> SubAgentResult:
        """Run a single sub-agent to completion.

        This is a simplified ReAct loop (not LangGraph) — we send the
        task + tools to the LLM, it returns either text or tool calls,
        we execute tools, feed results back, repeat until done.
        """
        from app.api.ai.prompt import AGENT_CONFIGS, build_system_prompt
        from app.agent.tools import create_langchain_tools, create_langchain_tools_from_defs
        from app.agent.intent_classifier import get_tool_defs_for_intent

        agent_config = AGENT_CONFIGS.get(task.agent_type, AGENT_CONFIGS["genie"])

        # ── Build sub-agent system prompt ──
        system_prompt = build_system_prompt(task.agent_type)
        system_prompt += f"\n\n---\n你是一个子 Agent，专门处理以下任务。完成后返回简洁的摘要。\n不要闲聊，直接执行任务并返回结果。"

        if task.system_prompt_extra:
            system_prompt += f"\n\n{task.system_prompt_extra}"

        if task.structured_output_schema:
            schema_json = json.dumps(task.structured_output_schema, ensure_ascii=False)
            system_prompt += (
                f"\n\n你的输出必须严格遵循以下 JSON Schema：\n```json\n{schema_json}\n```\n"
                f"只返回符合 Schema 的 JSON，不要包含解释文字。"
            )

        # ── Tool setup ──
        if task.tools is not None:
            from app.agent.tools import TOOL_REGISTRY
            tool_defs = [TOOL_REGISTRY[name] for name in task.tools if name in TOOL_REGISTRY]
            lc_tools = create_langchain_tools_from_defs(tool_defs)
        else:
            lc_tools = create_langchain_tools(task.agent_type)

        llm_with_tools = llm_client  # We manage tool calls manually

        # ── Build messages ──
        # Inject context data if provided
        task_msg = task.task_description
        if task.context_data:
            ctx_lines = ["\n\n参考数据："]
            for k, v in task.context_data.items():
                ctx_lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False)[:500]}")
            task_msg += "\n".join(ctx_lines)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_msg},
        ]

        # Build tool schema for API calls
        tool_schemas = []
        for t in lc_tools:
            schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.schema() if hasattr(t, "args_schema") else {},
                },
            }
            tool_schemas.append(schema)

        # ── ReAct loop ──
        tool_calls_made = 0
        total_tokens = 0

        for iteration in range(task.max_iterations):
            try:
                resp = await llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    temperature=0.3,
                    max_tokens=1024,
                )
            except Exception as e:
                return SubAgentResult(
                    task_id=task.task_id,
                    success=False,
                    summary="",
                    error=f"LLM call failed: {e}",
                    tool_calls_made=tool_calls_made,
                )

            choice = resp.choices[0]
            msg = choice.message
            total_tokens += resp.usage.total_tokens if resp.usage else 0

            # ── If the model returns content (no tool calls), we're done ──
            if msg.content and not msg.tool_calls:
                # Try to parse structured output if schema was provided
                structured = None
                if task.structured_output_schema and msg.content.strip():
                    try:
                        raw = msg.content.strip()
                        if raw.startswith("```"):
                            import re
                            raw = re.sub(r"^```(?:json)?\s*", "", raw)
                            raw = re.sub(r"\s*```$", "", raw)
                        structured = json.loads(raw)
                    except json.JSONDecodeError:
                        pass

                return SubAgentResult(
                    task_id=task.task_id,
                    success=True,
                    summary=msg.content[:1000],
                    structured_data=structured,
                    tool_calls_made=tool_calls_made,
                    tokens_used=total_tokens,
                )

            # ── Execute tool calls ──
            if msg.tool_calls:
                # Add assistant message with tool calls
                tool_call_blocks = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_call_blocks,
                })

                # Execute each tool
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        params = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        params = {}

                    async with db_factory() as db:
                        try:
                            result_text = await tool_executor(tool_name, params, db)
                            await db.commit()
                        except Exception as e:
                            result_text = f"工具执行错误: {e}"

                    tool_calls_made += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result_text)[:2000],  # cap per result
                    })

                continue  # Next iteration

            # No content and no tool calls — shouldn't happen, but handle gracefully
            return SubAgentResult(
                task_id=task.task_id,
                success=True,
                summary="任务完成（无输出）",
                tool_calls_made=tool_calls_made,
                tokens_used=total_tokens,
            )

        # Max iterations exceeded
        return SubAgentResult(
            task_id=task.task_id,
            success=False,
            summary="",
            error=f"超过最大迭代次数 ({task.max_iterations})",
            tool_calls_made=tool_calls_made,
            tokens_used=total_tokens,
        )
