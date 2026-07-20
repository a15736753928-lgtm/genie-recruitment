"""
query_loop — the main async generator state machine (Claude Code architecture).

This is the HEART of the refactored system.  It replaces LangGraph's
ReAct graph with a native Python ``async generator`` driven by immutable
state assignment.

Architecture (double-generator pattern)::

    QueryEngine (session lifecycle)
        │
        ├── for await (event of query(params)):
        │       yield event to SSE stream
        │
        └── query() ── yield* ──→ query_loop(state)
                                      │
                                      while True:
                                          compress → call_model → dispatch_tools
                                          state = next_state

Why generators instead of recursion:
    1. Backpressure — consumer controls production speed
    2. Cascading cancellation — .aclose() propagates through nested generators
    3. Stack safety — no recursion depth limits during long sessions
    4. Streaming composition — sub-agents nest directly into parent stream

Inspired by Claude Code's ``query.ts`` (~1,729 lines) and ``QueryEngine.ts`` (~1,295 lines).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional, Callable, Awaitable

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
    ToolBlock,
    ToolExecutionResult,
    is_concurrency_safe,
)

logger = logging.getLogger("genie.query_loop")


# ═══════════════════════════════════════════════════════════════
# SSE event helpers
# ═══════════════════════════════════════════════════════════════

def sse_event(event_type: str, data: dict) -> dict:
    """Create a typed SSE event dict (caller serializes to SSE format)."""
    return {"type": event_type, "data": data}


# ═══════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Conservative token count estimate.

    CJK characters ≈ 1 token each.  ASCII text ≈ 0.25 tokens per char.
    This is a heuristic — actual tokenization depends on the model.
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
    ascii_chars = len(text) - cjk
    return cjk + (ascii_chars // 4) + 1


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = ""
        if hasattr(msg, "content"):
            content = str(msg.content)
        elif isinstance(msg, dict):
            content = str(msg.get("content", ""))
        else:
            content = str(msg)
        total += estimate_tokens(content)
        # Tool calls overhead
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(str(tc))
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += estimate_tokens(str(tc))
    return total


# ═══════════════════════════════════════════════════════════════
# Compression pipeline (imported from compact module)
# ═══════════════════════════════════════════════════════════════

# Forward references — actual implementations in compact/ module.
# The loop imports them lazily to avoid circular dependencies.
_compaction_pipeline = None


async def _get_compaction_pipeline():
    """Lazy-load the compaction pipeline singleton."""
    global _compaction_pipeline
    if _compaction_pipeline is None:
        from app.agent_os.compact.pipeline import CompactionPipeline
        from app.agent_os.compact.pipeline import CompactionConfig
        _compaction_pipeline = CompactionPipeline(CompactionConfig())
    return _compaction_pipeline


# ═══════════════════════════════════════════════════════════════
# Main query loop
# ═══════════════════════════════════════════════════════════════

async def query_loop(
    initial_messages: list,
    system_prompt_static: str,
    system_prompt_dynamic: str,
    tools: list[dict],
    context: QueryContext,
    *,
    llm_client=None,
    tool_executor: Optional[Callable[..., Awaitable[str]]] = None,
    db_factory=None,
    hook_manager=None,
    permission_engine=None,
    on_turn_start: Optional[Callable[[int, LoopState], Awaitable[None]]] = None,
    on_turn_end: Optional[Callable[[int, TurnMetrics], Awaitable[None]]] = None,
) -> AsyncGenerator[dict, None]:
    """The main agent loop — async generator state machine.

    Yields SSE event dicts::

        {"type": "content", "data": {"delta": "..."}}
        {"type": "tool_call", "data": {"id": "...", "name": "...", "params": {...}}}
        {"type": "tool_result", "data": {"id": "...", "result": "..."}}
        {"type": "thinking", "data": {"text": "..."}}
        {"type": "compaction", "data": {"layer": 3, "savings": "45%"}}
        {"type": "error", "data": {"message": "..."}}
        {"type": "done", "data": {"exit_reason": "completed"}}

    Args:
        initial_messages: Starting conversation messages.
        system_prompt_static: Cacheable system prompt prefix.
        system_prompt_dynamic: Per-turn system prompt suffix.
        tools: Tool definitions (name, description, parameters).
        context: Immutable query configuration.
        llm_client: Async OpenAI-compatible client.
        tool_executor: async (tool_name, params, db) -> result_text.
        db_factory: Async session factory for database access.
        hook_manager: Optional HookManager for lifecycle events.
        permission_engine: Optional PermissionEngine for tool authorization.
        on_turn_start: Optional callback before each turn.
        on_turn_end: Optional callback after each turn.

    Yields:
        SSE event dicts.
    """
    # ── Initialize state ──
    state = LoopState(
        messages=list(initial_messages),
        tool_use_context={
            "session_id": context.session_id,
            "agent_id": context.agent_id,
            "permission_mode": context.permission_mode,
        },
    )

    pipeline = await _get_compaction_pipeline()

    # ── Pre-loop: fire SessionStart hooks ──
    if hook_manager:
        try:
            await hook_manager.fire(
                "session_start",
                {"session_id": context.session_id, "agent_id": context.agent_id},
            )
        except Exception:
            pass  # Hooks are best-effort

    # ═══════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════════════════
    while True:
        turn_start_time = time.time()

        # ── Safety cap ──
        if state.turn_count >= context.max_turns:
            yield sse_event("error", {
                "message": f"已达到最大轮次上限 ({context.max_turns})，任务可能过于复杂，请拆分后重试。",
            })
            yield sse_event("done", {"exit_reason": ExitReason.MAX_TURNS.value})
            return

        # ── Callback: turn start ──
        if on_turn_start:
            try:
                await on_turn_start(state.turn_count, state)
            except Exception:
                pass

        # ═══════════════════════════════════════════════════
        #  Phase 1: Compression Pipeline (before every API call)
        # ═══════════════════════════════════════════════════
        pre_compaction_tokens = estimate_messages_tokens(state.messages)

        try:
            state = await pipeline.prepare(
                state,
                llm_client=llm_client,
                context=context,
            )
        except Exception as e:
            logger.warning("Compaction pipeline error (non-fatal): %s", e)
            # Continue with uncompressed messages — better than failing

        post_compaction_tokens = estimate_messages_tokens(state.messages)

        if pre_compaction_tokens > post_compaction_tokens:
            savings = round((1 - post_compaction_tokens / max(pre_compaction_tokens, 1)) * 100, 1)
            yield sse_event("compaction", {
                "layer": pipeline.last_layer_used,
                "original_tokens": pre_compaction_tokens,
                "compressed_tokens": post_compaction_tokens,
                "savings_percent": savings,
                "message": f"上下文已压缩 (Layer {pipeline.last_layer_used})，{savings}% token 节省",
            })

        # ═══════════════════════════════════════════════════
        #  Phase 2: Streaming API Call + Parallel Tool Execution
        # ═══════════════════════════════════════════════════

        # Build the full system prompt
        full_system = system_prompt_static
        if system_prompt_dynamic:
            full_system += "\n\n" + system_prompt_dynamic

        # Prepare messages for the API
        api_messages = [{"role": "system", "content": full_system}]
        for msg in state.messages:
            api_messages.append(_message_to_api_dict(msg))

        # Tool schemas for the API
        tool_schemas = _build_tool_schemas(tools)

        # Permission pre-filtering: strip denied tools
        if permission_engine and tool_schemas:
            tool_schemas = permission_engine.filter_tools_for_model(
                tool_schemas, context.permission_mode
            )

        # Streaming tool executor (runs tools DURING model streaming)
        streaming_executor = StreamingToolExecutor(
            max_concurrent=10,
            tool_executor=tool_executor,
            db_factory=db_factory,
        )

        assistant_content = ""
        assistant_tool_blocks: list[ToolBlock] = []
        error_withheld = False
        error_to_withhold: Optional[dict] = None

        model_start = time.time()

        try:
            # ── Stream the model ──
            if llm_client is None:
                # No LLM client — simulate a response for testing
                yield sse_event("content", {"delta": "[无 LLM 客户端连接]"})
            else:
                stream = await llm_client.chat.completions.create(
                    model=getattr(llm_client, "_model", "deepseek-chat"),
                    messages=api_messages,
                    tools=tool_schemas if tool_schemas else None,
                    temperature=0.7,
                    max_tokens=context.max_tokens_override or context.max_output_tokens,
                    stream=True,
                )

                current_tool_call: Optional[dict] = None
                current_tool_id: Optional[str] = None
                current_tool_name: Optional[str] = None
                current_tool_args: str = ""

                async for chunk in stream:
                    # Check for abort
                    if context.abort_signal and _is_aborted(context.abort_signal):
                        yield sse_event("error", {"message": "用户取消了操作"})
                        streaming_executor.cancel_all()
                        yield sse_event("done", {"exit_reason": ExitReason.ABORTED_STREAMING.value})
                        return

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    # ── Content delta ──
                    if delta.content:
                        assistant_content += delta.content
                        yield sse_event("content", {"delta": delta.content})

                    # ── Tool call delta ──
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            # New tool call
                            if idx is not None and (
                                current_tool_call is None or idx != current_tool_call.get("index")
                            ):
                                # Finalize previous tool call
                                if current_tool_id and current_tool_name:
                                    try:
                                        args = json.loads(current_tool_args) if current_tool_args.strip() else {}
                                    except json.JSONDecodeError:
                                        args = {}
                                    block = ToolBlock(
                                        id=current_tool_id,
                                        name=current_tool_name,
                                        arguments=args,
                                        raw_arguments=current_tool_args,
                                    )
                                    assistant_tool_blocks.append(block)

                                    # 🔑 KEY INNOVATION: dispatch immediately
                                    streaming_executor.add_tool(block)

                                    yield sse_event("tool_call", {
                                        "id": block.id,
                                        "name": block.name,
                                        "params": block.arguments,
                                    })

                                # Start new tool call
                                current_tool_call = {"index": idx}
                                current_tool_id = tc_delta.id or f"tool_{uuid.uuid4().hex[:8]}"
                                current_tool_name = ""
                                current_tool_args = ""

                            if tc_delta.id:
                                current_tool_id = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    current_tool_name = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    current_tool_args += tc_delta.function.arguments

                    # ── Yield completed tool results during streaming ──
                    for result in streaming_executor.get_completed_results():
                        yield sse_event("tool_result", {
                            "id": result.tool_id,
                            "name": result.tool_name,
                            "success": result.success,
                            "result": result.result_text[:3000],
                            "error": result.error,
                            "duration_ms": result.duration_ms,
                            "display_hint": result.display_hint,
                        })

                # ── Finalize last tool call after stream ends ──
                if current_tool_id and current_tool_name:
                    try:
                        args = json.loads(current_tool_args) if current_tool_args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                    block = ToolBlock(
                        id=current_tool_id,
                        name=current_tool_name,
                        arguments=args,
                        raw_arguments=current_tool_args,
                    )
                    assistant_tool_blocks.append(block)
                    streaming_executor.add_tool(block)
                    yield sse_event("tool_call", {
                        "id": block.id,
                        "name": block.name,
                        "params": block.arguments,
                    })

        except asyncio.CancelledError:
            streaming_executor.cancel_all()
            yield sse_event("done", {"exit_reason": ExitReason.CANCELLED.value})
            return

        except Exception as e:
            error_str = str(e)

            # ── Prompt-Too-Long recovery ──
            if "prompt_too_long" in error_str.lower() or "413" in error_str:
                if not state.has_attempted_collapse_drain:
                    state.has_attempted_collapse_drain = True
                    state.transition = ContinueReason.COLLAPSE_DRAIN_RETRY
                    yield sse_event("thinking", {
                        "text": "上下文超出限制，正在压缩历史消息...",
                        "append": True,
                    })
                    continue

                elif not state.has_attempted_reactive_compact:
                    state.has_attempted_reactive_compact = True
                    state.transition = ContinueReason.REACTIVE_COMPACT_RETRY
                    yield sse_event("thinking", {
                        "text": "正在深度压缩对话历史...",
                        "append": True,
                    })
                    continue

                else:
                    yield sse_event("error", {
                        "message": "对话内容过长，已超出系统处理能力。请开启新对话或减少附件。",
                    })
                    yield sse_event("done", {"exit_reason": ExitReason.PROMPT_TOO_LONG.value})
                    return

            # ── Max output tokens recovery ──
            elif "max_output_tokens" in error_str.lower() or "max_tokens" in error_str.lower():
                if state.max_tokens_recovery_count < context.max_recovery_attempts:
                    state.max_tokens_override = context.max_output_tokens_escalated
                    state.max_tokens_recovery_count += 1
                    state.transition = ContinueReason.MAX_TOKENS_RECOVERY
                    yield sse_event("thinking", {
                        "text": "输出达到上限，正在以更大上下文重试...",
                        "append": True,
                    })
                    continue
                else:
                    error_to_withhold = {"message": error_str}

            # ── Rate limit ──
            elif "rate_limit" in error_str.lower() or "429" in error_str:
                yield sse_event("error", {
                    "message": "API 请求频率过高，请稍后重试。",
                })
                yield sse_event("done", {"exit_reason": ExitReason.RATE_LIMITED.value})
                return

            # ── Auth error ──
            elif "auth" in error_str.lower() or "401" in error_str or "403" in error_str:
                yield sse_event("error", {
                    "message": "API 认证失败，请检查 API Key 配置。",
                })
                yield sse_event("done", {"exit_reason": ExitReason.AUTH_ERROR.value})
                return

            else:
                error_withheld = {"message": error_str}

        model_duration = (time.time() - model_start) * 1000

        # ── Collect remaining tool results (after stream) ──
        remaining_results = await streaming_executor.get_remaining_results()
        for result in remaining_results:
            yield sse_event("tool_result", {
                "id": result.tool_id,
                "name": result.tool_name,
                "success": result.success,
                "result": result.result_text[:3000],
                "error": result.error,
                "duration_ms": result.duration_ms,
                "display_hint": result.display_hint,
            })

        # Combine all tool results
        all_tool_results = streaming_executor.get_completed_results() + remaining_results

        # ═══════════════════════════════════════════════════
        #  Phase 3: Decision Point
        # ═══════════════════════════════════════════════════

        has_tool_calls = len(assistant_tool_blocks) > 0

        if not has_tool_calls:
            # ── No tool calls: model is done ──

            # Run stop hooks
            if hook_manager:
                try:
                    hook_results = await hook_manager.fire(
                        "stop",
                        {
                            "session_id": context.session_id,
                            "response": assistant_content,
                            "tool_blocks": assistant_tool_blocks,
                        },
                    )
                    for hr in hook_results:
                        if hr.get("preventContinuation"):
                            yield sse_event("done", {
                                "exit_reason": ExitReason.STOP_HOOK_PREVENTED.value,
                            })
                            return
                        if hr.get("blocking"):
                            # Inject blocking error message and restart loop
                            error_msg = hr.get("message", "系统检测到问题，正在重新检查...")
                            state.messages.append(_make_user_message(error_msg))
                            state.stop_hook_active = True
                            state.transition = ContinueReason.STOP_HOOK_BLOCKING
                            yield sse_event("thinking", {
                                "text": error_msg,
                                "append": True,
                            })
                            # Skip turn-end processing — go directly to next iteration
                            continue  # goes to while True top
                except Exception:
                    pass

            # Token budget continuation
            if context.token_budget is not None and context.token_budget > 0:
                current_usage = estimate_messages_tokens(state.messages)
                if current_usage < context.token_budget * context.token_budget_completion_threshold:
                    # Budget not exhausted — inject nudge
                    # Diminishing returns detection
                    recent_tokens = [
                        m.tokens_after_compaction
                        for m in state.metrics[-3:]
                        if m.tokens_after_compaction > 0
                    ]
                    if len(recent_tokens) >= 3:
                        deltas = [
                            recent_tokens[i] - recent_tokens[i - 1]
                            for i in range(1, len(recent_tokens))
                        ]
                        if all(d < 500 for d in deltas):
                            # Diminishing returns — stop
                            yield sse_event("done", {
                                "exit_reason": ExitReason.COMPLETED.value,
                            })
                            return

                    # Inject nudge
                    nudge = _make_user_message(
                        "请继续完成上述任务。如果已完成，请给出最终总结。"
                    )
                    state.messages.append(nudge)
                    state.transition = ContinueReason.TOKEN_BUDGET_CONTINUATION
                    yield sse_event("thinking", {
                        "text": "Token 预算未耗尽，继续执行...",
                        "append": True,
                    })
                    continue

            # ── Normal completion ──
            yield sse_event("done", {
                "exit_reason": ExitReason.COMPLETED.value,
                "final_content": assistant_content[:500],
            })
            return

        # ── Withheld error handling ──
        if error_withheld:
            # Only surface withheld errors if there are no tool calls
            # (no recovery path remaining)
            yield sse_event("error", error_withheld)
            yield sse_event("done", {"exit_reason": ExitReason.MODEL_ERROR.value})
            return

        # ═══════════════════════════════════════════════════
        #  Phase 4: Inject Tool Results into Messages
        # ═══════════════════════════════════════════════════

        # Add assistant message with tool calls
        assistant_msg = _make_assistant_message(
            content=assistant_content,
            tool_blocks=assistant_tool_blocks,
        )
        new_messages = list(state.messages) + [assistant_msg]

        # Add tool result messages
        for result in all_tool_results:
            tool_msg = _make_tool_result_message(result)
            new_messages.append(tool_msg)

        # ═══════════════════════════════════════════════════
        #  Phase 5: State Update (immutable assignment)
        # ═══════════════════════════════════════════════════

        # Record turn metrics
        metrics = TurnMetrics(
            turn_number=state.turn_count,
            messages_before=len(state.messages),
            tokens_before=pre_compaction_tokens,
            messages_after_compaction=len(new_messages),
            tokens_after_compaction=post_compaction_tokens,
            compaction_layer_used=pipeline.last_layer_used,
            model_call_duration_ms=model_duration,
            tool_calls_count=len(assistant_tool_blocks),
            tool_calls_streaming_completed=streaming_executor.completed_count,
            output_tokens=estimate_tokens(assistant_content),
            input_tokens=post_compaction_tokens,
            transition=ContinueReason.NEXT_TURN,
        )

        # Immutable state assignment
        state = LoopState(
            messages=new_messages,
            max_tokens_override=state.max_tokens_override,
            max_tokens_recovery_count=state.max_tokens_recovery_count,
            has_attempted_reactive_compact=state.has_attempted_reactive_compact,
            has_attempted_collapse_drain=state.has_attempted_collapse_drain,
            stop_hook_active=state.stop_hook_active,
            turn_count=state.turn_count + 1,
            transition=ContinueReason.NEXT_TURN,
            auto_compact_failures=state.auto_compact_failures,
            tool_use_context=state.tool_use_context,
            metrics=list(state.metrics) + [metrics],
        )

        # ── Callback: turn end ──
        if on_turn_end:
            try:
                await on_turn_end(state.turn_count, metrics)
            except Exception:
                pass

        # ── Fire PostToolUse hooks (for each completed tool) ──
        if hook_manager:
            for result in all_tool_results:
                try:
                    await hook_manager.fire(
                        "post_tool_use" if result.success else "post_tool_use_failure",
                        {
                            "session_id": context.session_id,
                            "tool_name": result.tool_name,
                            "result": result.result_text[:500],
                            "success": result.success,
                            "error": result.error,
                        },
                        tool_name=result.tool_name,
                    )
                except Exception:
                    pass

        # Back to while True top — NOT recursion!
        continue


# ═══════════════════════════════════════════════════════════════
# QueryEngine — session lifecycle wrapper (double-generator pattern)
# ═══════════════════════════════════════════════════════════════

class QueryEngine:
    """Manages session lifecycle around individual query_loop calls.

    Claude Code's QueryEngine (~1,295 lines) handles:
    - Multi-turn state accumulation
    - Transcript persistence (JSONL)
    - Usage/cost tracking
    - SDK protocol compatibility

    This is the "outer" generator — it wraps query_loop and adds
    session-level concerns.
    """

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    async def submit_message(
        self,
        message: str,
        session_id: str,
        agent_id: str,
        system_prompt_static: str,
        system_prompt_dynamic: str,
        tools: list[dict],
        context: QueryContext,
        *,
        llm_client=None,
        tool_executor=None,
        db_factory=None,
        hook_manager=None,
        permission_engine=None,
    ) -> AsyncGenerator[dict, None]:
        """Submit a user message and stream the agent's response.

        This is the main entry point for the chat endpoint.
        """
        # ── Transcript-first: persist user message BEFORE API call ──
        # (Claude Code pattern: even a process kill between send and
        #  response leaves a resumable session)
        user_msg_saved = False
        if db_factory:
            try:
                from app.models.agent_session import AgentMessage, AgentSession
                from sqlalchemy import select
                async with db_factory() as db:
                    # Ensure session exists
                    result = await db.execute(
                        select(AgentSession).where(AgentSession.id == session_id)
                    )
                    session = result.scalar_one_or_none()
                    if not session:
                        session = AgentSession(id=session_id, title="新对话", agent_id=agent_id)
                        db.add(session)
                        await db.flush()

                    user_msg = AgentMessage(
                        session_id=session_id,
                        role="user",
                        content=message,
                    )
                    db.add(user_msg)
                    await db.commit()
                    user_msg_saved = True
            except Exception:
                pass

        # ── Build initial messages ──
        from app.agent_os.loop.state import LoopState
        state = LoopState()

        # Load conversation history if available
        if db_factory and session_id:
            try:
                from app.models.agent_session import AgentMessage
                from sqlalchemy import select
                async with db_factory() as db:
                    result = await db.execute(
                        select(AgentMessage)
                        .where(AgentMessage.session_id == session_id)
                        .order_by(AgentMessage.created_at)
                        .limit(30)
                    )
                    history = result.scalars().all()
                    for h in history:
                        if h.role == "user":
                            state.messages.append(_make_user_message(h.content or ""))
                        elif h.role == "assistant":
                            state.messages.append(_make_assistant_message(
                                content=h.content or "",
                                tool_blocks=h.tool_blocks or [],
                            ))
            except Exception:
                pass

        # Add current user message if not already in history
        if not user_msg_saved:
            state.messages.append(_make_user_message(message))

        # ── Run the agent loop ──
        exit_reason = ExitReason.COMPLETED
        assistant_full_response = ""
        assistant_tool_blocks = []

        async for event in query_loop(
            initial_messages=state.messages,
            system_prompt_static=system_prompt_static,
            system_prompt_dynamic=system_prompt_dynamic,
            tools=tools,
            context=context,
            llm_client=llm_client,
            tool_executor=tool_executor,
            db_factory=db_factory,
            hook_manager=hook_manager,
            permission_engine=permission_engine,
        ):
            yield event

            # Track for persistence
            if event["type"] == "content":
                assistant_full_response += event["data"].get("delta", "")
            elif event["type"] == "done":
                exit_reason = ExitReason(event["data"].get("exit_reason", "completed"))

        # ── Persist assistant response ──
        if db_factory and assistant_full_response:
            try:
                async with db_factory() as db:
                    assistant_msg = AgentMessage(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_full_response,
                        tool_blocks=assistant_tool_blocks if assistant_tool_blocks else None,
                    )
                    db.add(assistant_msg)
                    await db.commit()
            except Exception:
                pass

        # Yield final result as a special event (can't return from async generator)
        final_result = LoopResult(
            exit_reason=exit_reason,
            final_response=assistant_full_response,
            tool_blocks=assistant_tool_blocks,
        )
        yield {"type": "loop_result", "data": {
            "exit_reason": final_result.exit_reason.value,
            "final_response": final_result.final_response[:500],
            "total_tool_blocks": len(final_result.tool_blocks),
        }}
        # NOTE: The caller can inspect assistant_full_response and exit_reason
        # from the yielded events directly.


# ═══════════════════════════════════════════════════════════════
# Convenience wrapper
# ═══════════════════════════════════════════════════════════════

async def query(
    messages: list,
    system_prompt: str,
    tools: list[dict],
    *,
    llm_client=None,
    tool_executor=None,
    db_factory=None,
    max_turns: int = 30,
    **kwargs,
) -> AsyncGenerator[dict, None]:
    """Convenience wrapper around query_loop for simple use cases.

    Usage::

        async for event in query(messages, system_prompt, tools, llm_client=client):
            if event["type"] == "content":
                print(event["data"]["delta"], end="")
    """
    context = QueryContext(max_turns=max_turns, **kwargs)
    async for event in query_loop(
        initial_messages=messages,
        system_prompt_static=system_prompt,
        system_prompt_dynamic="",
        tools=tools,
        context=context,
        llm_client=llm_client,
        tool_executor=tool_executor,
        db_factory=db_factory,
    ):
        yield event


# ═══════════════════════════════════════════════════════════════
# Message construction helpers
# ═══════════════════════════════════════════════════════════════

def _message_to_api_dict(msg) -> dict:
    """Convert an internal message object to an OpenAI API-compatible dict."""
    if isinstance(msg, dict):
        return msg

    role = getattr(msg, "role", "user")
    content = str(getattr(msg, "content", ""))

    result = {"role": role, "content": content}

    # Handle tool calls on assistant messages
    if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = msg.tool_calls

    # Handle tool_call_id on tool messages
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id

    return result


def _make_user_message(content: str):
    """Create a user message object."""
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=content)


def _make_assistant_message(content: str, tool_blocks: list = None):
    """Create an assistant message object with optional tool calls."""
    from langchain_core.messages import AIMessage

    if tool_blocks:
        tool_calls = [
            {
                "id": tb.id if hasattr(tb, "id") else tb.get("id", ""),
                "name": tb.name if hasattr(tb, "name") else tb.get("name", ""),
                "args": tb.arguments if hasattr(tb, "arguments") else tb.get("params", {}),
                "type": "tool_call",
            }
            for tb in tool_blocks
        ]
        return AIMessage(content=content, tool_calls=tool_calls)

    return AIMessage(content=content)


def _make_tool_result_message(result: ToolExecutionResult):
    """Create a tool result message object."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(
        content=result.result_text[:4000],
        tool_call_id=result.tool_id,
        name=result.tool_name,
    )


def _build_tool_schemas(tools: list[dict]) -> list[dict]:
    """Build OpenAI-compatible tool schemas from internal definitions."""
    schemas = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
        else:
            name = getattr(tool, "name", "")
            desc = getattr(tool, "description", "")
            params = getattr(tool, "args_schema", None)
            if params and hasattr(params, "schema"):
                params = params.schema()
            elif params:
                params = params if isinstance(params, dict) else {}

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc[:1024] if desc else "",
                "parameters": params if params else {"type": "object", "properties": {}},
            },
        })
    return schemas


def _is_aborted(abort_signal) -> bool:
    """Check if the abort signal is set."""
    if abort_signal is None:
        return False
    if hasattr(abort_signal, "is_set"):
        return abort_signal.is_set()
    if isinstance(abort_signal, asyncio.Event):
        return abort_signal.is_set()
    return False
