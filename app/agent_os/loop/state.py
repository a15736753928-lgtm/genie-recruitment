"""
LoopState, ContinueReason, ExitReason — the type system driving the agent loop.

Claude Code's agent loop uses typed enums for every transition and exit,
preventing infinite recovery loops and making error handling explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContinueReason(str, Enum):
    """Why the loop continued — each reason gates specific recovery logic.

    The ``transition`` field on LoopState records the reason for the
    PREVIOUS continuation.  Downstream iterations check this to avoid
    repeating failed recovery attempts.
    """
    NEXT_TURN = "next_turn"
    """Normal continuation: model returned tool_use blocks."""

    MAX_TOKENS_ESCALATE = "max_tokens_escalate"
    """First hit of output token cap; silently boost to 64K and retry."""

    MAX_TOKENS_RECOVERY = "max_tokens_recovery"
    """Model output was truncated; inject recovery nudge (up to 3×)."""

    REACTIVE_COMPACT_RETRY = "reactive_compact_retry"
    """Prompt-too-long → forced full compaction → retry."""

    COLLAPSE_DRAIN_RETRY = "collapse_drain_retry"
    """Prompt-too-long → drained collapse stages → retry."""

    STOP_HOOK_BLOCKING = "stop_hook_blocking"
    """Stop hook returned blocking error → inject user message → re-query."""

    TOKEN_BUDGET_CONTINUATION = "token_budget_continuation"
    """Token budget not yet consumed; inject nudge to continue work."""


class ExitReason(str, Enum):
    """Why the loop exited — typed terminal conditions.

    Every exit is explicit and named.  Callers can branch on the reason
    to decide whether to surface an error to the user or handle gracefully.
    """
    COMPLETED = "completed"
    """Normal completion: model finished without tool calls."""

    BLOCKING_LIMIT = "blocking_limit"
    """Token count exceeded hard limit with auto-compact disabled."""

    MODEL_ERROR = "model_error"
    """Unrecoverable model API error."""

    PROMPT_TOO_LONG = "prompt_too_long"
    """Prompt exceeded limit after all recovery attempts exhausted."""

    ABORTED_STREAMING = "aborted_streaming"
    """Client disconnected during streaming."""

    STOP_HOOK_PREVENTED = "stop_hook_prevented"
    """Stop hook returned preventContinuation: true."""

    ABORTED_TOOLS = "aborted_tools"
    """Client disconnected during tool execution."""

    HOOK_STOPPED = "hook_stopped"
    """Pre/post-tool hook returned shouldPreventContinuation."""

    MAX_TURNS = "max_turns"
    """Hard turn cap reached (safety limit)."""

    RATE_LIMITED = "rate_limited"
    """API rate limit hit with no retry budget remaining."""

    AUTH_ERROR = "auth_error"
    """Authentication failure (expired key, revoked token)."""

    CANCELLED = "cancelled"
    """Explicit cancellation by user or system."""


@dataclass
class QueryContext:
    """Immutable configuration for a single query execution.

    Passed through to the loop but never mutated by it.
    """
    session_id: str = ""
    agent_id: str = "genie"
    permission_mode: str = "default"       # "default" | "acceptEdits" | "plan" | "bypassPermissions" | "dontAsk"
    max_turns: int = 30                     # safety cap
    max_output_tokens: int = 8192           # default output limit
    max_output_tokens_escalated: int = 64000  # escalated limit
    max_recovery_attempts: int = 3          # max_tokens recovery cap
    token_budget: Optional[int] = None      # total token budget (None = unlimited)
    token_budget_completion_threshold: float = 0.90  # stop when 90% of budget used
    enable_streaming_tools: bool = True
    enable_auto_compact: bool = True
    auto_compact_circuit_breaker: int = 3
    abort_signal: Optional[any] = None      # asyncio.Event or similar


@dataclass
class TurnMetrics:
    """Per-turn telemetry for monitoring and debugging."""
    turn_number: int = 0
    messages_before: int = 0
    tokens_before: int = 0
    messages_after_compaction: int = 0
    tokens_after_compaction: int = 0
    compaction_layer_used: int = 0
    model_call_duration_ms: float = 0
    tool_calls_count: int = 0
    tool_calls_streaming_completed: int = 0  # tools that finished during streaming
    output_tokens: int = 0
    input_tokens: int = 0
    cache_hit: bool = False
    transition: Optional[ContinueReason] = None


@dataclass
class LoopState:
    """Immutable loop state — full assignment at every continue site.

    Design rule (from Claude Code): a single ``state = {...}`` assignment
    is preferred over multiple independent variable assignments because it
    ensures every continue site explicitly declares ALL state and nothing
    is accidentally omitted.

    NEVER mutate fields in place.  Always create a new LoopState.
    """
    messages: list = field(default_factory=list)
    """Current conversation messages (system + user + assistant + tool results)."""

    # ── Token / output management ──
    max_tokens_override: Optional[int] = None
    """If set, overrides the default max_output_tokens for this turn."""

    max_tokens_recovery_count: int = 0
    """How many times we've attempted max_tokens recovery (cap at 3)."""

    # ── Recovery guards (prevent infinite loops) ──
    has_attempted_reactive_compact: bool = False
    """True if we already tried reactive compact for a PTL error."""

    has_attempted_collapse_drain: bool = False
    """True if we already drained collapse stages for a PTL error."""

    stop_hook_active: bool = False
    """True if a stop hook is currently blocking (prevents re-entry)."""

    # ── Progress tracking ──
    turn_count: int = 0
    """Current turn number (incremented at each continue)."""

    transition: Optional[ContinueReason] = None
    """Why the PREVIOUS turn continued.  Checked by recovery logic."""

    # ── Compaction tracking ──
    auto_compact_failures: int = 0
    """Consecutive auto-compact failures (triggers circuit breaker)."""

    auto_compact_tracking: Optional[dict] = None
    """Opaque tracking data passed between compaction attempts."""

    # ── Tool execution context ──
    tool_use_context: dict = field(default_factory=dict)
    """Context passed to tool execution (permission state, session info)."""

    # ── Pending async work ──
    pending_tool_use_summary: Optional[any] = None
    """Future: Haiku-generated tool summary promise (not yet awaited)."""

    # ── Metrics (accumulated across turns) ──
    metrics: list[TurnMetrics] = field(default_factory=list)


@dataclass
class LoopResult:
    """Final result returned when the loop exits."""
    exit_reason: ExitReason
    final_response: str = ""
    tool_blocks: list[dict] = field(default_factory=list)
    total_turns: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    metrics: list[TurnMetrics] = field(default_factory=list)
    error: Optional[str] = None
