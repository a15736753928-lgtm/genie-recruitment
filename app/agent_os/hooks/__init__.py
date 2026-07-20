from app.agent_os.hooks.events import HookEvent, EVENT_LABELS, EVENT_GROUPS
from app.agent_os.hooks.manager import ExtendedHookManager, HookResult, HookType

# Backward-compatible aliases for existing code
HookManager = ExtendedHookManager
Hook = HookResult

__all__ = [
    "HookEvent", "EVENT_LABELS", "EVENT_GROUPS",
    "ExtendedHookManager", "HookManager",
    "HookResult", "Hook",
    "HookType",
]
