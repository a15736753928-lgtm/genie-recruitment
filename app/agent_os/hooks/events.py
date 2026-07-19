"""
Hook events — lifecycle events that can trigger automated actions.

Inspired by Claude Code's hook system (27+ events).  Each event fires
at a specific point in the agent's lifecycle, allowing plugins and
built-in handlers to react.
"""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    """Lifecycle events for the Agent OS hook system."""

    # ── Session lifecycle ──
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # ── Tool execution ──
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"

    # ── Message processing ──
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    AGENT_RESPONSE = "agent_response"

    # ── Sub-agent (Phase 4) ──
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"

    # ── Context ──
    PRE_COMPACT = "pre_compact"

    # ── Quality ──
    VERIFICATION_FAILED = "verification_failed"

    # ── Business events ──
    CANDIDATE_STATUS_CHANGED = "candidate_status_changed"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER_APPROVED = "offer_approved"


# Human-readable labels for each event
EVENT_LABELS: dict[HookEvent, str] = {
    HookEvent.SESSION_START: "会话开始",
    HookEvent.SESSION_END: "会话结束",
    HookEvent.PRE_TOOL_USE: "工具执行前",
    HookEvent.POST_TOOL_USE: "工具执行后",
    HookEvent.USER_PROMPT_SUBMIT: "用户消息提交",
    HookEvent.AGENT_RESPONSE: "Agent 回复完成",
    HookEvent.SUBAGENT_START: "子 Agent 启动",
    HookEvent.SUBAGENT_STOP: "子 Agent 完成",
    HookEvent.PRE_COMPACT: "上下文压缩前",
    HookEvent.VERIFICATION_FAILED: "验证失败",
    HookEvent.CANDIDATE_STATUS_CHANGED: "候选人状态变更",
    HookEvent.INTERVIEW_SCHEDULED: "面试已安排",
    HookEvent.OFFER_APPROVED: "Offer 已批准",
}
