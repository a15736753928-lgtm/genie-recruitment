"""
Safety checks — bypass-immune permission constraints.

These checks are ALWAYS evaluated regardless of permission mode.
Even bypassPermissions cannot skip them.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Optional

from app.agent_os.permissions.engine import PermissionDecision, PermissionResult


# ── Protected paths ─────────────────────────────────────────

PROTECTED_PATHS = [
    ".git/",
    ".claude/",
    ".env",
    "*.env",
    "credentials*",
    "secrets*",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "authorized_keys",
    "known_hosts",
    "config.yaml",
    "config.yml",
    "settings.local.json",
    "*.secret",
]

DESTRUCTIVE_TOOLS = {
    "delete_resume", "delete_position", "delete_knowledge_item",
    "delete_document", "delete_knowledge_base",
    "batch_parse_resumes",  # can overwrite data
}

# ── PII patterns ────────────────────────────────────────────

PII_PATTERNS = {
    "chinese_id": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "phone": re.compile(r"1[3-9]\d{9}"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
}


# ── Safety check functions ──────────────────────────────────

def safety_check_protected_paths(
    tool_name: str,
    tool_params: dict,
    context: dict,
) -> Optional[PermissionResult]:
    """Check if a tool is accessing protected paths.

    Applied to: read_file, write_file, edit_file, bash (and equivalents).
    """
    PATH_TOOLS = {
        "upload_knowledge_file", "upload_document",
        "upload_resume", "batch_parse_resumes",
    }

    # Extract paths from tool params
    paths = _extract_paths(tool_name, tool_params)

    for path in paths:
        for protected in PROTECTED_PATHS:
            if fnmatch.fnmatch(path, protected) or protected.replace("*", "") in path:
                return PermissionResult(
                    decision=PermissionDecision.ASK,
                    reason=f"操作涉及受保护路径: {path}（匹配规则: {protected}）",
                    requires_user_input=True,
                    suggested_action="确认路径安全后手动批准",
                )

    return None  # Pass


def safety_check_destructive_operations(
    tool_name: str,
    tool_params: dict,
    context: dict,
) -> Optional[PermissionResult]:
    """Force user confirmation for destructive operations.

    Always returns ASK for tools in DESTRUCTIVE_TOOLS.
    """
    if tool_name in DESTRUCTIVE_TOOLS:
        entity_id = tool_params.get("id", tool_params.get("knowledge_id", "未知"))
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"即将执行破坏性操作: {tool_name} (目标ID: {entity_id})",
            requires_user_input=True,
            suggested_action="请确认您确实要删除此数据",
        )

    return None  # Pass


def safety_check_pii_leak(
    tool_name: str,
    tool_params: dict,
    context: dict,
) -> Optional[PermissionResult]:
    """Check for potential PII leaks in tool output.

    Applied to: tools that display or export candidate data.
    """
    PII_SENSITIVE_TOOLS = {
        "list_resumes", "get_resume", "export_*",
        "get_evaluation", "get_leaderboard", "get_rankings",
    }

    if tool_name not in PII_SENSITIVE_TOOLS:
        return None  # Not applicable

    # Check if we're exporting a large batch (potential bulk PII)
    limit = tool_params.get("limit", 0)
    if isinstance(limit, (int, float)) and limit > 50:
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"即将返回 {limit} 条候选人记录，可能包含大量个人信息",
            requires_user_input=True,
            suggested_action="考虑缩小查询范围或使用脱敏模式",
        )

    return None  # Pass


def safety_check_batch_operations(
    tool_name: str,
    tool_params: dict,
    context: dict,
) -> Optional[PermissionResult]:
    """Check for unusually large batch operations."""
    BATCH_TOOLS = {
        "batch_parse_resumes", "batch_*",
    }

    if tool_name not in BATCH_TOOLS and not tool_name.startswith("batch_"):
        return None

    count = tool_params.get("count", tool_params.get("batch_size", 0))
    if isinstance(count, (int, float)) and count > 100:
        return PermissionResult(
            decision=PermissionDecision.ASK,
            reason=f"批量操作超过100条 ({count}条)，可能需要较长时间并消耗大量资源",
            requires_user_input=True,
            suggested_action="考虑分批处理或减少批量大小",
        )

    return None


# ── Helpers ──────────────────────────────────────────────────

def _extract_paths(tool_name: str, tool_params: dict) -> list[str]:
    """Extract file system paths from tool parameters."""
    paths = []

    # Direct path fields
    for key in ("path", "file_path", "file", "source", "destination", "target"):
        if key in tool_params:
            paths.append(str(tool_params[key]))

    # Nested in 'fields'
    if "fields" in tool_params and isinstance(tool_params["fields"], dict):
        for key in ("path", "file_path", "file"):
            if key in tool_params["fields"]:
                paths.append(str(tool_params["fields"][key]))

    # Array of paths
    for key in ("paths", "files", "targets"):
        if key in tool_params and isinstance(tool_params[key], list):
            paths.extend(str(p) for p in tool_params[key])

    return paths


def mask_pii(text: str) -> str:
    """Mask PII in text for safe display."""
    for pattern_name, pattern in PII_PATTERNS.items():
        if pattern_name == "chinese_id":
            text = pattern.sub(lambda m: m.group()[:6] + "****" + m.group()[-4:], text)
        elif pattern_name == "phone":
            text = pattern.sub(lambda m: m.group()[:3] + "****" + m.group()[-4:], text)
        elif pattern_name == "email":
            text = pattern.sub(lambda m: m.group()[0] + "***@" + m.group().split("@")[1], text)
    return text
