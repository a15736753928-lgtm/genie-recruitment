"""
Hook events — 27 lifecycle events (upgraded from 12).

Inspired by Claude Code's hook system with 27+ events across 5 responsibility groups:
    1. Tool & Permission (5 events)
    2. Session Lifecycle (7 events)
    3. Context & Workspace (7 events)
    4. Sub-Agent & Task (6 events)
    5. Elicitation (2 events)

Each event fires at a specific point in the agent's lifecycle.
"""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    """27 lifecycle events for the Agent OS hook system."""

    # ═══════════════════════════════════════════════════════
    # Group 1: Tool & Permission (5 events)
    # ═══════════════════════════════════════════════════════
    PERMISSION_REQUEST = "permission_request"
    """Fires BEFORE showing the permission dialog.  Hooks can auto-allow/deny."""

    PRE_TOOL_USE = "pre_tool_use"
    """Fires BEFORE tool execution.  Hooks can block or modify input."""

    POST_TOOL_USE = "post_tool_use"
    """Fires AFTER successful tool execution."""

    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    """Fires AFTER failed tool execution.  Hooks can inject recovery hints."""

    PERMISSION_DENIED = "permission_denied"
    """Fires when auto-classifier or rule denies a tool.  Hook can return {retry: true}."""

    # ═══════════════════════════════════════════════════════
    # Group 2: Session Lifecycle (7 events)
    # ═══════════════════════════════════════════════════════
    SETUP = "setup"
    """Runtime preparation phase (--init / --maintenance)."""

    SESSION_START = "session_start"
    """Session begins or resumes.  Matchers: startup | resume | clear | compact."""

    USER_PROMPT_SUBMIT = "user_prompt_submit"
    """User submits a prompt, BEFORE model processing."""

    USER_PROMPT_EXPANSION = "user_prompt_expansion"
    """Command expanded into full prompt (e.g., /review → full review prompt)."""

    NOTIFICATION = "notification"
    """System notification triggered."""

    STOP = "stop"
    """Claude is about to stop.  Key interception point for continuation loops."""

    STOP_FAILURE = "stop_failure"
    """Stop flow failed (e.g., API error during final cleanup)."""

    SESSION_END = "session_end"
    """Session terminates.  Matchers: clear | resume | logout | timeout."""

    # ═══════════════════════════════════════════════════════
    # Group 3: Context & Workspace (7 events)
    # ═══════════════════════════════════════════════════════
    PRE_COMPACT = "pre_compact"
    """Context compression is about to run."""

    POST_COMPACT = "post_compact"
    """Context compression completed.  Hooks can re-inject memory/context."""

    INSTRUCTIONS_LOADED = "instructions_loaded"
    """CLAUDE.md or rules loaded.  Hooks can modify or supplement instructions."""

    CWD_CHANGED = "cwd_changed"
    """Working directory changed (e.g., cd command executed)."""

    FILE_CHANGED = "file_changed"
    """A watched file changed on disk.  Matcher specifies filename pattern."""

    WORKTREE_CREATE = "worktree_create"
    """A git worktree was created for sub-agent isolation."""

    WORKTREE_REMOVE = "worktree_remove"
    """A git worktree was removed after sub-agent completion."""

    # ═══════════════════════════════════════════════════════
    # Group 4: Sub-Agent & Task (6 events)
    # ═══════════════════════════════════════════════════════
    SUBAGENT_START = "subagent_start"
    """A sub-agent is about to start execution."""

    SUBAGENT_STOP = "subagent_stop"
    """A sub-agent completed execution."""

    TASK_CREATED = "task_created"
    """A new task was created (via TaskCreate tool)."""

    TASK_COMPLETED = "task_completed"
    """A task was marked complete."""

    TEAMMATE_IDLE = "teammate_idle"
    """A teammate agent is transitioning to idle state."""

    CONFIG_CHANGE = "config_change"
    """Configuration file changed (hot-reload)."""

    # ═══════════════════════════════════════════════════════
    # Group 5: Elicitation (2 events)
    # ═══════════════════════════════════════════════════════
    ELICITATION = "elicitation"
    """MCP server requests user input."""

    ELICITATION_RESULT = "elicitation_result"
    """User responded to an elicitation request."""

    # ═══════════════════════════════════════════════════════
    # Group 6: Quality & Business (Genie-specific)
    # ═══════════════════════════════════════════════════════
    AGENT_RESPONSE = "agent_response"
    """Agent produced a complete response."""

    VERIFICATION_FAILED = "verification_failed"
    """Post-execution quality check failed."""

    VERIFICATION_PASSED = "verification_passed"
    """Post-execution quality check passed."""

    MESSAGE_DISPLAY = "message_display"
    """A message is about to be displayed to the user."""

    # ── Business events (recruitment domain) ──
    CANDIDATE_STATUS_CHANGED = "candidate_status_changed"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER_APPROVED = "offer_approved"


# ── Human-readable labels ────────────────────────────────────

EVENT_LABELS: dict[HookEvent, str] = {
    # Tool & Permission
    HookEvent.PERMISSION_REQUEST: "权限请求",
    HookEvent.PRE_TOOL_USE: "工具执行前",
    HookEvent.POST_TOOL_USE: "工具执行后",
    HookEvent.POST_TOOL_USE_FAILURE: "工具执行失败",
    HookEvent.PERMISSION_DENIED: "权限被拒绝",

    # Session Lifecycle
    HookEvent.SETUP: "运行时准备",
    HookEvent.SESSION_START: "会话开始",
    HookEvent.SESSION_END: "会话结束",
    HookEvent.USER_PROMPT_SUBMIT: "用户提交Prompt",
    HookEvent.USER_PROMPT_EXPANSION: "命令展开",
    HookEvent.NOTIFICATION: "系统通知",
    HookEvent.STOP: "Agent停止",
    HookEvent.STOP_FAILURE: "停止失败",

    # Context & Workspace
    HookEvent.PRE_COMPACT: "上下文压缩前",
    HookEvent.POST_COMPACT: "上下文压缩后",
    HookEvent.INSTRUCTIONS_LOADED: "指令加载完成",
    HookEvent.CWD_CHANGED: "工作目录变更",
    HookEvent.FILE_CHANGED: "文件变更",
    HookEvent.WORKTREE_CREATE: "Worktree创建",
    HookEvent.WORKTREE_REMOVE: "Worktree移除",

    # Sub-Agent & Task
    HookEvent.SUBAGENT_START: "子Agent启动",
    HookEvent.SUBAGENT_STOP: "子Agent完成",
    HookEvent.TASK_CREATED: "任务创建",
    HookEvent.TASK_COMPLETED: "任务完成",
    HookEvent.TEAMMATE_IDLE: "Agent空闲",
    HookEvent.CONFIG_CHANGE: "配置变更",

    # Elicitation
    HookEvent.ELICITATION: "请求用户输入",
    HookEvent.ELICITATION_RESULT: "用户输入响应",

    # Quality & Business
    HookEvent.AGENT_RESPONSE: "Agent回复完成",
    HookEvent.VERIFICATION_FAILED: "验证失败",
    HookEvent.VERIFICATION_PASSED: "验证通过",
    HookEvent.MESSAGE_DISPLAY: "消息展示",

    HookEvent.CANDIDATE_STATUS_CHANGED: "候选人状态变更",
    HookEvent.INTERVIEW_SCHEDULED: "面试已安排",
    HookEvent.OFFER_APPROVED: "Offer已批准",
}

# ── Event groups for UI display ──────────────────────────────

EVENT_GROUPS: dict[str, list[HookEvent]] = {
    "工具与权限": [
        HookEvent.PERMISSION_REQUEST,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.POST_TOOL_USE_FAILURE,
        HookEvent.PERMISSION_DENIED,
    ],
    "会话生命周期": [
        HookEvent.SETUP,
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.USER_PROMPT_EXPANSION,
        HookEvent.NOTIFICATION,
        HookEvent.STOP,
        HookEvent.STOP_FAILURE,
    ],
    "上下文与工作区": [
        HookEvent.PRE_COMPACT,
        HookEvent.POST_COMPACT,
        HookEvent.INSTRUCTIONS_LOADED,
        HookEvent.CWD_CHANGED,
        HookEvent.FILE_CHANGED,
        HookEvent.WORKTREE_CREATE,
        HookEvent.WORKTREE_REMOVE,
    ],
    "子Agent与任务": [
        HookEvent.SUBAGENT_START,
        HookEvent.SUBAGENT_STOP,
        HookEvent.TASK_CREATED,
        HookEvent.TASK_COMPLETED,
        HookEvent.TEAMMATE_IDLE,
        HookEvent.CONFIG_CHANGE,
    ],
    "交互询问": [
        HookEvent.ELICITATION,
        HookEvent.ELICITATION_RESULT,
    ],
    "质量验证": [
        HookEvent.VERIFICATION_FAILED,
        HookEvent.VERIFICATION_PASSED,
        HookEvent.AGENT_RESPONSE,
        HookEvent.MESSAGE_DISPLAY,
    ],
    "业务事件": [
        HookEvent.CANDIDATE_STATUS_CHANGED,
        HookEvent.INTERVIEW_SCHEDULED,
        HookEvent.OFFER_APPROVED,
    ],
}
