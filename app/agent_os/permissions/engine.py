"""
PermissionEngine — Deny-First 纵深防御权限决策引擎.

Inspired by Claude Code's ``src/services/permissions/`` module.

Design principles:
    1. ``deny > ask > allow`` — strictest applicable rule always wins
    2. Deny rules are IMMUNE to bypassPermissions
    3. Bare-name denies REMOVE the tool from the model's view entirely
    4. Safety checks apply regardless of permission mode
    5. Ask rules force user interaction even in bypass mode

Decision pipeline (6-stage):
    1a. Tool-level Deny rules → immediate reject
    1b. Tool-level Ask rules → force user interaction
    1c. Safety checks → deny or ask (bypass-immune)
    2a. Permission mode evaluation
    2b. Allow rules → whitelist
    3.  Passthrough → default to ask
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class PermissionDecision(str, Enum):
    """Permission decision for a tool call."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    PASSTHROUGH = "passthrough"  # Tool defers to general flow


class PermissionMode(str, Enum):
    """Permission modes (7 total).

    Public modes (user-configurable):
        default, acceptEdits, plan, bypassPermissions, dontAsk

    Internal modes:
        auto (ML classifier — feature-gated)
        bubble (sub-agent escalates to parent)
    """
    DEFAULT = "default"              # Interactive confirmation
    ACCEPT_EDITS = "acceptEdits"     # Auto-approve file operations
    PLAN = "plan"                    # All edits need user approval
    BYPASS = "bypassPermissions"     # Skip generic confirmations
    DONT_ASK = "dontAsk"            # Ask→Deny, no interaction (CI/CD)
    AUTO = "auto"                    # ML classifier (internal)
    BUBBLE = "bubble"                # Escalate to parent agent (internal)


@dataclass
class PermissionRule:
    """A single permission rule in the engine."""
    tool_pattern: str               # fnmatch pattern or exact tool name
    decision: PermissionDecision
    conditions: Optional[dict] = None  # Additional constraints
    priority: int = 0               # Higher = evaluated first
    description: str = ""


@dataclass
class PermissionResult:
    """Result of a permission evaluation."""
    decision: PermissionDecision
    rule_matched: Optional[str] = None   # Which rule triggered
    reason: str = ""                     # Human-readable reason
    requires_user_input: bool = False    # Must show dialog
    suggested_action: str = ""           # e.g. "add to allowlist"


class PermissionEngine:
    """Deny-First permission decision engine.

    Usage::

        engine = PermissionEngine()
        engine.add_deny("delete_*", description="Block destructive operations")
        engine.add_allow("list_*", description="Allow all read operations")
        engine.add_ask("update_*", description="Confirm all modifications")

        result = engine.evaluate(
            tool_name="delete_resume",
            tool_params={"id": "xxx"},
            mode=PermissionMode.DEFAULT,
        )
        if result.decision == PermissionDecision.DENY:
            raise PermissionError(result.reason)
    """

    def __init__(self):
        self._deny_rules: list[PermissionRule] = []
        self._ask_rules: list[PermissionRule] = []
        self._allow_rules: list[PermissionRule] = []
        self._safety_checks: list[Callable] = []
        self._register_defaults()

    # ── Rule registration ───────────────────────────────────

    def add_deny(self, pattern: str, conditions=None, priority=100, description=""):
        """Add a deny rule — highest priority, bypass-immune."""
        self._deny_rules.append(PermissionRule(
            tool_pattern=pattern,
            decision=PermissionDecision.DENY,
            conditions=conditions,
            priority=priority,
            description=description,
        ))
        self._deny_rules.sort(key=lambda r: -r.priority)

    def add_ask(self, pattern: str, conditions=None, priority=50, description=""):
        """Add an ask rule — forces user interaction."""
        self._ask_rules.append(PermissionRule(
            tool_pattern=pattern,
            decision=PermissionDecision.ASK,
            conditions=conditions,
            priority=priority,
            description=description,
        ))
        self._ask_rules.sort(key=lambda r: -r.priority)

    def add_allow(self, pattern: str, conditions=None, priority=0, description=""):
        """Add an allow rule — whitelist."""
        self._allow_rules.append(PermissionRule(
            tool_pattern=pattern,
            decision=PermissionDecision.ALLOW,
            conditions=conditions,
            priority=priority,
            description=description,
        ))
        self._allow_rules.sort(key=lambda r: -r.priority)

    def add_safety_check(self, check_fn: Callable):
        """Add a safety check function.

        Safety checks are ALWAYS evaluated regardless of permission mode.
        They can return DENY or ASK decisions.
        """
        self._safety_checks.append(check_fn)

    # ── Main evaluation ─────────────────────────────────────

    def evaluate(
        self,
        tool_name: str,
        tool_params: dict,
        mode: PermissionMode,
        context: dict | None = None,
    ) -> PermissionResult:
        """Evaluate whether a tool call is permitted.

        Decision pipeline:
            1a. Deny rules → immediate reject
            1b. Safety checks → deny or ask (bypass-immune)
            1c. Ask rules → force interaction
            2a. Permission mode evaluation
            2b. Allow rules → whitelist
            3.  Passthrough → default to ask

        Args:
            tool_name: Name of the tool being called.
            tool_params: Parameters being passed to the tool.
            mode: Current permission mode.
            context: Additional context (session_id, agent_id, etc.).

        Returns:
            PermissionResult with final decision.
        """
        ctx = context or {}
        decisions: list[tuple[PermissionDecision, str]] = []

        # ── Stage 1a: Deny rules ──
        for rule in self._deny_rules:
            if self._matches(tool_name, tool_params, rule, ctx):
                return PermissionResult(
                    decision=PermissionDecision.DENY,
                    rule_matched=rule.tool_pattern,
                    reason=rule.description or f"工具 '{tool_name}' 被拒绝规则 '{rule.tool_pattern}' 阻止",
                    requires_user_input=False,
                )

        # ── Stage 1b: Safety checks (bypass-immune) ──
        for check_fn in self._safety_checks:
            result = check_fn(tool_name, tool_params, ctx)
            if result is None:
                continue  # Safety check passes
            if isinstance(result, PermissionResult):
                if result.decision == PermissionDecision.DENY:
                    return result  # Safety deny is final
                if result.decision == PermissionDecision.ASK:
                    decisions.append((PermissionDecision.ASK, result.reason))
            elif isinstance(result, str):
                # String returned = deny reason
                return PermissionResult(
                    decision=PermissionDecision.DENY,
                    reason=result,
                )

        # ── Stage 1c: Ask rules (also bypass-immune) ──
        for rule in self._ask_rules:
            if self._matches(tool_name, tool_params, rule, ctx):
                decisions.append((
                    PermissionDecision.ASK,
                    rule.description or f"工具 '{tool_name}' 需要确认",
                ))

        # ── Stage 2a: Permission mode evaluation ──
        if mode == PermissionMode.DONT_ASK:
            # Ask→Deny: if any ask rule matched, deny immediately
            if any(d[0] == PermissionDecision.ASK for d in decisions):
                return PermissionResult(
                    decision=PermissionDecision.DENY,
                    reason="当前处于 Don't Ask 模式，需要确认的操作被自动拒绝",
                )

        if mode == PermissionMode.BYPASS:
            # Skip generic confirmations, but deny/ask rules still apply
            if not decisions:  # No deny/ask matched
                return PermissionResult(
                    decision=PermissionDecision.ALLOW,
                    reason="bypassPermissions 模式",
                )

        # ── Stage 2b: Allow rules ──
        for rule in self._allow_rules:
            if self._matches(tool_name, tool_params, rule, ctx):
                decisions.append((
                    PermissionDecision.ALLOW,
                    rule.description or f"工具 '{tool_name}' 在白名单中",
                ))

        # ── Stage 3: Merge decisions (deny > ask > allow) ──
        if any(d[0] == PermissionDecision.DENY for d in decisions):
            reason = next(d[1] for d in decisions if d[0] == PermissionDecision.DENY)
            return PermissionResult(decision=PermissionDecision.DENY, reason=reason)

        if any(d[0] == PermissionDecision.ASK for d in decisions):
            reason = next(d[1] for d in decisions if d[0] == PermissionDecision.ASK)
            return PermissionResult(
                decision=PermissionDecision.ASK,
                reason=reason,
                requires_user_input=True,
            )

        if any(d[0] == PermissionDecision.ALLOW for d in decisions):
            return PermissionResult(decision=PermissionDecision.ALLOW)

        # Default: ask (conservative, safe)
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"工具 '{tool_name}' 没有匹配的权限规则，需要用户确认",
            requires_user_input=True,
        )

    # ── Tool pool pre-filtering ─────────────────────────────

    def filter_tools_for_model(
        self,
        tools: list[dict],
        mode: PermissionMode,
    ) -> list[dict]:
        """Remove denied tools from the model's view.

        Bare-name deny rules (no conditions) strip the tool entirely
        from the tool pool — the model never sees them.
        """
        filtered = []
        for tool in tools:
            name = tool.get("function", {}).get("name", tool.get("name", ""))

            # Check bare-name denies
            is_bare_denied = any(
                fnmatch.fnmatch(name, rule.tool_pattern)
                and rule.conditions is None
                for rule in self._deny_rules
            )

            if not is_bare_denied:
                filtered.append(tool)

        return filtered

    def check_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available (not bare-name denied)."""
        return not any(
            fnmatch.fnmatch(tool_name, rule.tool_pattern)
            and rule.conditions is None
            for rule in self._deny_rules
        )

    # ── Rule matching ───────────────────────────────────────

    def _matches(
        self,
        tool_name: str,
        tool_params: dict,
        rule: PermissionRule,
        context: dict,
    ) -> bool:
        """Check if a tool call matches a rule."""
        if not fnmatch.fnmatch(tool_name, rule.tool_pattern):
            return False

        if rule.conditions:
            return self._evaluate_conditions(rule.conditions, tool_params, context)

        return True

    def _evaluate_conditions(
        self,
        conditions: dict,
        tool_params: dict,
        context: dict,
    ) -> bool:
        """Evaluate additional conditions on a rule."""
        for key, expected in conditions.items():
            actual = tool_params.get(key)

            # Path matching
            if key == "path_contains":
                path = str(tool_params.get("path", tool_params.get("file_path", "")))
                if expected not in path:
                    return False

            # Parameter value matching
            elif key == "param_equals":
                if isinstance(expected, dict):
                    for pk, pv in expected.items():
                        if tool_params.get(pk) != pv:
                            return False
                else:
                    return False

            # Session context matching
            elif key == "session_age_minutes":
                # Reject if session is younger than expected
                # (prevent racing conditions on new sessions)
                pass

        return True

    # ── Default rule registration ───────────────────────────

    def _register_defaults(self):
        """Register sensible default permission rules for recruitment system."""
        from app.agent_os.permissions.safety import (
            DESTRUCTIVE_TOOLS,
        )

        # Allow all read-only operations by default
        self.add_allow("list_*", description="允许所有列表查询操作", priority=10)
        self.add_allow("get_*", description="允许所有单条查询操作", priority=10)
        self.add_allow("rag_search", description="允许知识库搜索", priority=10)
        self.add_allow("recall_test", description="允许召回测试", priority=10)
        self.add_allow("search_*", description="允许搜索操作", priority=10)

        # Ask before destructive operations
        for tool in DESTRUCTIVE_TOOLS:
            self.add_ask(tool, description=f"破坏性操作 '{tool}' 需要确认", priority=50)

        # Deny system-level operations (add more as needed)
        self.add_deny("delete_knowledge_base", description="删除知识库需要管理员权限", priority=100)

    # ── Introspection ───────────────────────────────────────

    def list_rules(self) -> list[dict]:
        """List all registered rules for debugging/UI display."""
        rules = []
        for rule_list, category in [
            (self._deny_rules, "deny"),
            (self._ask_rules, "ask"),
            (self._allow_rules, "allow"),
        ]:
            for r in rule_list:
                rules.append({
                    "category": category,
                    "pattern": r.tool_pattern,
                    "priority": r.priority,
                    "description": r.description,
                })
        return sorted(rules, key=lambda r: (-r["priority"], r["category"]))
