"""
Deny-First Permission System — Claude Code architecture.

Core principle: ``deny > ask > allow``.  The strictest applicable rule always wins.
Deny rules are immune to bypassPermissions — bare-name denies even remove the
tool from the model's view entirely.

7 permission modes: default | acceptEdits | plan | bypassPermissions | dontAsk | auto | bubble
"""

from app.agent_os.permissions.engine import (
    PermissionEngine,
    PermissionDecision,
    PermissionMode,
    PermissionRule,
    PermissionResult,
)

from app.agent_os.permissions.safety import (
    PROTECTED_PATHS,
    DESTRUCTIVE_TOOLS,
    safety_check_protected_paths,
    safety_check_destructive_operations,
    safety_check_pii_leak,
    safety_check_batch_operations,
)

__all__ = [
    "PermissionEngine",
    "PermissionDecision",
    "PermissionMode",
    "PermissionRule",
    "PermissionResult",
    "PROTECTED_PATHS",
    "DESTRUCTIVE_TOOLS",
    "safety_check_protected_paths",
    "safety_check_destructive_operations",
    "safety_check_pii_leak",
    "safety_check_batch_operations",
]
