"""
ExtendedHookManager — 6 execution types for 27 lifecycle events.

Upgraded from the original HookManager with:
- 6 hook execution types: command, prompt, agent, http, callback, function
- Exit code semantics: 0=success, 1=non-blocking error, 2=blocking error
- Matcher system: exact name, regex, wildcard
- Result aggregation: deny > ask > allow; multiple contexts concatenated

Inspired by Claude Code's hook runtime at ``src/services/hooks/``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional

from app.agent_os.hooks.events import HookEvent

logger = logging.getLogger("genie.hooks")


class HookType(str, Enum):
    """Six hook execution types."""
    COMMAND = "command"
    """Spawn shell subprocess.  stdin/stdout JSON communication.  Exit code semantics."""

    PROMPT = "prompt"
    """Single-turn LLM call.  Returns {ok: true/false}."""

    AGENT = "agent"
    """Multi-turn agent loop.  Can call tools to verify."""

    HTTP = "http"
    """POST request to external endpoint."""

    CALLBACK = "callback"
    """In-process async function.  SDK/plugin use only."""

    FUNCTION = "function"
    """Session-scoped function.  Isolated by sessionId."""


@dataclass
class HookResult:
    """Aggregated result from a hook execution."""
    hook_name: str = ""
    decision: str = "allow"         # "allow" | "deny" | "ask"
    blocking: bool = False          # If True, stop the triggering operation
    prevent_continuation: bool = False  # If True, end the session
    message: str = ""               # Human-readable message
    additional_context: str = ""    # Context to inject
    updated_input: Optional[dict] = None  # Modified tool input
    retry: bool = False             # For PermissionDenied: retry with modified params
    raw_output: str = ""            # Raw hook output for debugging


class ExitCode:
    """Hook exit code semantics."""
    SUCCESS = 0              # Normal continuation
    NON_BLOCKING_ERROR = 1   # Log but continue
    BLOCKING_ERROR = 2       # STOP execution (only "deny" signal)


class ExtendedHookManager:
    """Manages hook registration and firing across 6 execution types.

    Hooks fire in priority order (highest first).  A hook returning
    blocking=True or decision="deny" stops the chain.

    Result aggregation rules:
    - Any "deny" → overall deny
    - Any "blocking" → stop
    - Multiple "additionalContext" → concatenate
    - Multiple "updatedInput" → last one wins
    - Any "preventContinuation" → session stops
    """

    def __init__(self):
        self._hooks: list[_HookRegistration] = []
        self._session_hooks: dict[str, list[Callable]] = {}  # sessionId → callbacks
        self.register_builtins()

    # ── Registration ────────────────────────────────────────

    def register(
        self,
        name: str,
        event: HookEvent,
        hook_type: HookType = HookType.CALLBACK,
        *,
        matcher: Optional[str] = None,
        priority: int = 0,
        enabled: bool = True,
        # Type-specific configs
        command: Optional[str] = None,
        prompt_template: Optional[str] = None,
        agent_config: Optional[dict] = None,
        url: Optional[str] = None,
        callback: Optional[Callable[..., Awaitable[dict | None]]] = None,
        timeout_ms: int = 30000,
    ):
        """Register a hook.

        Args:
            name: Unique hook identifier.
            event: Which lifecycle event triggers this hook.
            hook_type: Execution type (command/prompt/agent/http/callback/function).
            matcher: Filter pattern (tool name, event source, notification type).
                     Supports: exact match, fnmatch wildcards, regex (if contains special chars).
            priority: Higher = executed first.
            command: Shell command string (for HookType.COMMAND).
            prompt_template: Prompt text (for HookType.PROMPT).
            agent_config: Agent configuration (for HookType.AGENT).
            url: HTTP endpoint URL (for HookType.HTTP).
            callback: Async callable (for HookType.CALLBACK/FUNCTION).
            timeout_ms: Execution timeout.
        """
        # Replace existing hook with same name
        self._hooks = [h for h in self._hooks if h.name != name]
        self._hooks.append(_HookRegistration(
            name=name,
            event=event,
            hook_type=hook_type,
            matcher=matcher,
            priority=priority,
            enabled=enabled,
            command=command,
            prompt_template=prompt_template,
            agent_config=agent_config,
            url=url,
            callback=callback,
            timeout_ms=timeout_ms,
        ))
        self._hooks.sort(key=lambda h: -h.priority)

    def unregister(self, name: str):
        """Remove a hook by name."""
        self._hooks = [h for h in self._hooks if h.name != name]

    def register_session_hook(
        self,
        session_id: str,
        callback: Callable[..., Awaitable[dict | None]],
    ):
        """Register a session-scoped hook (HookType.FUNCTION equivalent)."""
        if session_id not in self._session_hooks:
            self._session_hooks[session_id] = []
        self._session_hooks[session_id].append(callback)

    def unregister_session_hooks(self, session_id: str):
        """Remove all session-scoped hooks."""
        self._session_hooks.pop(session_id, None)

    # ── Firing ──────────────────────────────────────────────

    async def fire(
        self,
        event: HookEvent,
        context: dict | None = None,
        tool_name: str = "",
    ) -> list[HookResult]:
        """Fire all hooks registered for an event.

        Args:
            event: The lifecycle event (HookEvent or string value).
            context: Event context dict (session_id, tool params, response, etc.).
            tool_name: Name of the tool (for tool events, enables matcher filtering).

        Returns:
            List of HookResult.  Caller should aggregate:
            - Any result with decision="deny" or blocking=True → block operation
            - Any result with prevent_continuation=True → end session
        """
        # Normalize event
        if isinstance(event, str):
            try:
                event = HookEvent(event)
            except ValueError:
                logger.warning("Unknown hook event: %s", event)
                return []

        ctx = context or {}
        results = []

        # Get matching hooks
        matching = self._get_matching(event, tool_name)

        for hook in matching:
            try:
                result = await self._execute_hook(hook, ctx, tool_name)
                if result:
                    results.append(result)

                    # Deny signal → stop chain
                    if result.decision == "deny" or result.blocking:
                        break

                    # Prevent continuation → stop chain
                    if result.prevent_continuation:
                        break

            except Exception as e:
                logger.warning("Hook '%s' failed: %s", hook.name, e)
                results.append(HookResult(
                    hook_name=hook.name,
                    decision="allow",
                    message=f"Hook执行异常: {e}",
                ))

        # Fire session-scoped hooks
        session_id = ctx.get("session_id", "")
        if session_id and session_id in self._session_hooks:
            for callback in self._session_hooks[session_id]:
                try:
                    raw = await callback(ctx, tool_name=tool_name)
                    if raw:
                        results.append(HookResult(
                            hook_name=f"session:{session_id}",
                            **raw,
                        ))
                except Exception as e:
                    logger.warning("Session hook failed: %s", e)

        return results

    # ── Hook execution ──────────────────────────────────────

    async def _execute_hook(
        self,
        hook: "_HookRegistration",
        context: dict,
        tool_name: str,
    ) -> Optional[HookResult]:
        """Execute a single hook based on its type."""
        if hook.hook_type == HookType.COMMAND:
            return await self._execute_command_hook(hook, context, tool_name)
        elif hook.hook_type == HookType.PROMPT:
            return await self._execute_prompt_hook(hook, context, tool_name)
        elif hook.hook_type == HookType.AGENT:
            return await self._execute_agent_hook(hook, context, tool_name)
        elif hook.hook_type == HookType.HTTP:
            return await self._execute_http_hook(hook, context, tool_name)
        elif hook.hook_type in (HookType.CALLBACK, HookType.FUNCTION):
            return await self._execute_callback_hook(hook, context, tool_name)
        else:
            logger.warning("Unknown hook type: %s", hook.hook_type)
            return None

    async def _execute_command_hook(
        self, hook, context: dict, tool_name: str
    ) -> Optional[HookResult]:
        """Execute command-type hook via subprocess.

        stdin receives JSON: {event, context, tool_name}
        stdout returns JSON: {decision, blocking, message, ...}
        Exit codes: 0=success, 1=non-blocking error, 2=blocking error
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *shlex.split(hook.command),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            input_json = json.dumps({
                "event": hook.event.value,
                "context": context,
                "tool_name": tool_name,
            }, ensure_ascii=False)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_json.encode()),
                    timeout=hook.timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                process.kill()
                return HookResult(
                    hook_name=hook.name,
                    decision="allow",
                    message=f"Hook超时 ({hook.timeout_ms}ms)",
                )

            # Exit code semantics
            if process.returncode == ExitCode.BLOCKING_ERROR:
                return HookResult(
                    hook_name=hook.name,
                    decision="deny",
                    blocking=True,
                    message=stderr.decode()[:500] if stderr else "Hook拒绝了操作",
                )

            if process.returncode == ExitCode.NON_BLOCKING_ERROR:
                logger.warning("Hook '%s' non-blocking error: %s", hook.name,
                              stderr.decode()[:200] if stderr else "")
                return None  # Continue but log

            # Success: parse stdout JSON
            try:
                output = json.loads(stdout.decode())
                return HookResult(hook_name=hook.name, **output)
            except json.JSONDecodeError:
                # Plain text stdout → inject as context
                return HookResult(
                    hook_name=hook.name,
                    additional_context=stdout.decode()[:2000],
                )

        except FileNotFoundError:
            logger.warning("Hook command not found: %s", hook.command)
            return None
        except Exception as e:
            logger.warning("Command hook '%s' failed: %s", hook.name, e)
            return None

    async def _execute_prompt_hook(
        self, hook, context: dict, tool_name: str
    ) -> Optional[HookResult]:
        """Execute prompt-type hook via single-turn LLM call.

        The prompt template is populated with context and sent to a
        lightweight model.  Returns {ok: true/false}.
        """
        # For now, prompt hooks are stubs (require LLM client injection)
        # Full implementation would:
        # 1. Populate the template with context
        # 2. Call a fast model (Haiku equivalent)
        # 3. Parse {ok: true/false} from response
        logger.debug("Prompt hook '%s' (stub): %s", hook.name, hook.prompt_template[:100] if hook.prompt_template else "")
        return None

    async def _execute_agent_hook(
        self, hook, context: dict, tool_name: str
    ) -> Optional[HookResult]:
        """Execute agent-type hook via multi-turn agent loop.

        The agent can call tools to verify claims made by the main agent.
        """
        # Stub: full agent hook implementation requires embedding
        # a sub-agent loop within the hook execution
        logger.debug("Agent hook '%s' (stub)", hook.name)
        return None

    async def _execute_http_hook(
        self, hook, context: dict, tool_name: str
    ) -> Optional[HookResult]:
        """Execute HTTP-type hook via POST request."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=hook.timeout_ms / 1000) as client:
                response = await client.post(
                    hook.url,
                    json={
                        "event": hook.event.value,
                        "context": context,
                        "tool_name": tool_name,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return HookResult(hook_name=hook.name, **data)
                else:
                    logger.warning("HTTP hook '%s' returned %d", hook.name, response.status_code)
                    return None
        except Exception as e:
            logger.warning("HTTP hook '%s' failed: %s", hook.name, e)
            return None

    async def _execute_callback_hook(
        self, hook, context: dict, tool_name: str
    ) -> Optional[HookResult]:
        """Execute callback/function-type hook."""
        if not hook.callback:
            return None
        try:
            raw = await hook.callback(context, tool_name=tool_name)
            if raw is None:
                return None
            if isinstance(raw, HookResult):
                return raw
            if isinstance(raw, dict):
                return HookResult(hook_name=hook.name, **raw)
            return None
        except Exception as e:
            logger.warning("Callback hook '%s' failed: %s", hook.name, e)
            return HookResult(
                hook_name=hook.name,
                message=f"Hook执行异常: {e}",
            )

    # ── Matching ────────────────────────────────────────────

    def _get_matching(
        self, event: HookEvent, tool_name: str = ""
    ) -> list["_HookRegistration"]:
        """Get all enabled hooks matching an event and optional tool name."""
        matching = []
        for hook in self._hooks:
            if not hook.enabled:
                continue
            if hook.event != event:
                continue
            if hook.matcher and tool_name:
                if not self._match_pattern(hook.matcher, tool_name):
                    continue
            matching.append(hook)
        return matching

    @staticmethod
    def _match_pattern(matcher: str, target: str) -> bool:
        """Match a tool name/event source against a matcher pattern.

        Rules:
        - "*" or empty → match all
        - Only letters/numbers/underscore/dash/comma/pipe → exact or list match
        - Contains other characters → regex match
        """
        if not matcher or matcher == "*":
            return True

        # Check if it's a simple name/list (no regex special chars)
        simple_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-,| ")
        if all(c in simple_chars for c in matcher):
            # Split by | and comma for list matching
            names = [n.strip() for n in re.split(r'[|,]', matcher)]
            return target in names

        # Regex match
        try:
            return bool(re.search(matcher, target))
        except re.error:
            return fnmatch.fnmatch(target, matcher)

    # ── Built-in hooks ──────────────────────────────────────

    def register_builtins(self):
        """Register the default set of built-in hooks."""

        # 1. Audit logging for destructive operations
        async def audit_write(ctx, tool_name=""):
            if tool_name.startswith("delete_") or tool_name.startswith("update_"):
                logger.info(
                    "AUDIT: %s called params=%s session=%s",
                    tool_name,
                    str(ctx.get("params", {}))[:200],
                    ctx.get("session_id", "?"),
                )
            return None

        self.register(
            name="audit_write_operations",
            event=HookEvent.PRE_TOOL_USE,
            hook_type=HookType.CALLBACK,
            matcher="delete_*|update_*",
            callback=audit_write,
            priority=100,
        )

        # 2. Candidate status change notification
        async def notify_status_change(ctx, tool_name=""):
            params = ctx.get("params", {})
            old_status = params.get("old_status", "")
            new_status = params.get("status", "") or (params.get("fields", {}) or {}).get("status", "")
            if new_status and old_status != new_status:
                logger.info(
                    "STATUS_CHANGE: candidate %s → %s (was: %s)",
                    params.get("id", "?"), new_status, old_status,
                )
            return None

        self.register(
            name="notify_status_change",
            event=HookEvent.POST_TOOL_USE,
            hook_type=HookType.CALLBACK,
            matcher="update_resume",
            callback=notify_status_change,
            priority=50,
        )

        # 3. PII leak check on agent response
        async def check_pii(ctx, tool_name=""):
            response = str(ctx.get("response", ""))
            id_pattern = r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
            matches = re.findall(id_pattern, response)
            if matches:
                return {
                    "decision": "ask",
                    "message": f"检测到 {len(matches)} 处可能的身份证号码，请确认是否应该展示",
                }
            phone_pattern = r"1[3-9]\d{9}"
            phone_matches = re.findall(phone_pattern, response)
            if len(phone_matches) > 3:
                return {
                    "decision": "ask",
                    "message": f"检测到 {len(phone_matches)} 个手机号码，请确认展示范围是否合理",
                }
            return None

        self.register(
            name="check_pii_leak",
            event=HookEvent.AGENT_RESPONSE,
            hook_type=HookType.CALLBACK,
            callback=check_pii,
            priority=200,
        )

        # 4. Post-compact memory re-injection
        async def re_inject_memory(ctx, tool_name=""):
            """After compaction, suggest re-injecting CLAUDE.md content."""
            # This is a no-op stub — full implementation would call MemoryManager
            return None

        self.register(
            name="re_inject_memory_on_compact",
            event=HookEvent.POST_COMPACT,
            hook_type=HookType.CALLBACK,
            callback=re_inject_memory,
            priority=75,
        )

    # ── Introspection ───────────────────────────────────────

    def list_hooks(self) -> list[dict]:
        """Return all registered hooks as dicts for UI display."""
        return [
            {
                "name": h.name,
                "event": h.event.value,
                "type": h.hook_type.value,
                "matcher": h.matcher,
                "priority": h.priority,
                "enabled": h.enabled,
            }
            for h in self._hooks
        ]

    def list_events(self) -> list[dict]:
        """List all available hook events with labels."""
        from app.agent_os.hooks.events import EVENT_LABELS, EVENT_GROUPS
        return [
            {
                "event": e.value,
                "label": EVENT_LABELS.get(e, e.value),
                "group": group,
            }
            for group, events in EVENT_GROUPS.items()
            for e in events
        ]


# ── Internal hook registration dataclass ─────────────────────

@dataclass
class _HookRegistration:
    """Internal hook registration object."""
    name: str
    event: HookEvent
    hook_type: HookType = HookType.CALLBACK
    matcher: Optional[str] = None
    priority: int = 0
    enabled: bool = True
    # Type-specific
    command: Optional[str] = None
    prompt_template: Optional[str] = None
    agent_config: Optional[dict] = None
    url: Optional[str] = None
    callback: Optional[Callable[..., Awaitable[dict | None]]] = None
    timeout_ms: int = 30000
