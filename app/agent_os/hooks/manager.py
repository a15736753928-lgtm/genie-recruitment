"""
HookManager — event-driven automation for the Agent OS.

Hooks are callbacks that fire at specific lifecycle events.  They can
modify outputs, block operations, inject context, or trigger side effects.

Built-in hooks include:
- ``audit_write_operations`` — log all write tool calls to DB
- ``check_pii_leak`` — scan agent responses for sensitive data patterns
- ``notify_status_change`` — fire business event on candidate status change
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

from app.agent_os.hooks.events import HookEvent

logger = logging.getLogger("genie.hooks")


@dataclass
class Hook:
    """A registered hook."""

    name: str
    event: HookEvent
    handler: Callable[..., Awaitable[dict | None]]
    matcher: Optional[str] = None     # tool name pattern (for tool events)
    priority: int = 0
    enabled: bool = True

    def matches(self, event: HookEvent, tool_name: str = "") -> bool:
        """Check if this hook should fire for the given event."""
        if not self.enabled or self.event != event:
            return False
        if self.matcher and tool_name:
            import fnmatch
            return fnmatch.fnmatch(tool_name, self.matcher)
        return True


class HookManager:
    """Manages hook registration and firing.

    Hooks fire in priority order (highest first).  A hook returning
    ``{"action": "block", "reason": "..."}`` stops the chain and blocks
    the triggering operation.
    """

    def __init__(self):
        self._hooks: list[Hook] = []
        self.register_builtins()

    # ── Registration ──────────────────────────────────────

    def register(self, hook: Hook):
        """Register a hook."""
        # Replace existing hook with same name
        self._hooks = [h for h in self._hooks if h.name != hook.name]
        self._hooks.append(hook)
        self._hooks.sort(key=lambda h: -h.priority)

    def unregister(self, name: str):
        """Remove a hook by name."""
        self._hooks = [h for h in self._hooks if h.name != name]

    def list_hooks(self) -> list[dict]:
        """Return all registered hooks as dicts."""
        return [
            {
                "name": h.name,
                "event": h.event.value,
                "matcher": h.matcher,
                "priority": h.priority,
                "enabled": h.enabled,
            }
            for h in self._hooks
        ]

    # ── Firing ────────────────────────────────────────────

    async def fire(
        self,
        event: HookEvent,
        context: dict | None = None,
        tool_name: str = "",
    ) -> list[dict]:
        """Fire all hooks registered for an event.

        Returns a list of hook results.  If any hook returns
        ``{"action": "block"}``, subsequent hooks are skipped.
        """
        ctx = context or {}
        results = []

        for hook in self._hooks:
            if not hook.matches(event, tool_name):
                continue
            try:
                result = await hook.handler(ctx, tool_name=tool_name)
                if result:
                    results.append({"hook": hook.name, **result})
                    if result.get("action") == "block":
                        break
            except Exception as e:
                logger.warning("Hook '%s' failed: %s", hook.name, e)

        return results

    # ── Built-in hooks ────────────────────────────────────

    def register_builtins(self):
        """Register the default set of hooks."""

        # 1. Audit logging for destructive operations
        async def audit_write(ctx, tool_name=""):
            if tool_name.startswith("delete_") or tool_name.startswith("update_"):
                logger.info(
                    "AUDIT: %s called with params=%s session=%s",
                    tool_name,
                    str(ctx.get("params", {}))[:200],
                    ctx.get("session_id", "?"),
                )
            return None  # never block

        self.register(Hook(
            name="audit_write_operations",
            event=HookEvent.PRE_TOOL_USE,
            matcher="delete_*",
            handler=audit_write,
            priority=100,
        ))
        self.register(Hook(
            name="audit_update_operations",
            event=HookEvent.PRE_TOOL_USE,
            matcher="update_*",
            handler=audit_write,
            priority=99,
        ))

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

        self.register(Hook(
            name="notify_status_change",
            event=HookEvent.POST_TOOL_USE,
            matcher="update_resume",
            handler=notify_status_change,
            priority=50,
        ))

        # 3. PII leak check on agent response
        async def check_pii(ctx, tool_name=""):
            response = str(ctx.get("response", ""))
            # Check for Chinese ID card numbers (18 digits)
            import re
            id_pattern = r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
            matches = re.findall(id_pattern, response)
            if matches:
                logger.warning("PII_CHECK: potential ID numbers in agent response")
                return {
                    "action": "warn",
                    "message": f"检测到 {len(matches)} 处可能的身份证号码，请确认是否应该展示",
                }
            # Check for phone numbers
            phone_pattern = r"1[3-9]\d{9}"
            phone_matches = re.findall(phone_pattern, response)
            if len(phone_matches) > 3:  # >3 phone numbers in one response is suspicious
                logger.warning("PII_CHECK: %d phone numbers in agent response", len(phone_matches))
                return {
                    "action": "warn",
                    "message": f"检测到 {len(phone_matches)} 个手机号码，请确认展示范围是否合理",
                }
            return None

        self.register(Hook(
            name="check_pii_leak",
            event=HookEvent.AGENT_RESPONSE,
            handler=check_pii,
            priority=200,
        ))
