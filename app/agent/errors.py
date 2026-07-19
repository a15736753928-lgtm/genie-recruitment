"""
Error classification and recovery strategies.

Mirrors Claude Code's pattern of distinguishing transient vs permanent
errors and applying appropriate recovery strategies automatically.

Error categories:
    NOT_FOUND     — Entity doesn't exist → suggest alternatives to LLM
    VALIDATION    — Invalid parameters → tell LLM what to fix
    PERMISSION    — Not allowed → explain why to user
    TRANSIENT     — Network timeout / API hiccup → auto-retry (max 2)
    SYSTEM        — DB down / config error → escalate to user
    BUSINESS      — Business rule violation → explain rule to user
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    PERMISSION = "permission"
    TRANSIENT = "transient"
    SYSTEM = "system"
    BUSINESS = "business"


# ── Error classifiers — each returns (category, user_msg, llm_hint) ──

def classify_db_error(error: Exception) -> tuple[ErrorCategory, str, str]:
    """Classify a database exception."""
    msg = str(error).lower()
    if "timeout" in msg or "connection" in msg or "deadlock" in msg:
        return (
            ErrorCategory.TRANSIENT,
            "数据库连接超时，正在重试…",
            "数据库暂时不可用，稍后重试。如果持续失败，检查数据库服务状态。",
        )
    if "unique" in msg or "duplicate" in msg:
        return (
            ErrorCategory.VALIDATION,
            "数据重复，请检查是否已存在相同记录。",
            "该记录已存在（唯一约束冲突）。请检查是否重复操作。",
        )
    if "foreign key" in msg or "constraint" in msg:
        return (
            ErrorCategory.VALIDATION,
            "数据关联错误，请检查引用的记录是否存在。",
            "外键约束失败。引用的记录可能已被删除。请检查关联数据。",
        )
    return (
        ErrorCategory.SYSTEM,
        f"数据库错误: {str(error)[:200]}",
        "数据库操作失败。请检查数据库连接和日志。",
    )


def classify_llm_error(error: Exception) -> tuple[ErrorCategory, str, str]:
    """Classify an LLM API exception."""
    msg = str(error).lower()
    if "timeout" in msg or "timed out" in msg:
        return (
            ErrorCategory.TRANSIENT,
            "AI 服务响应超时，正在重试…",
            "LLM API 超时。自动重试中。如果持续超时，检查 API 密钥和网络。",
        )
    if "rate" in msg or "429" in msg:
        return (
            ErrorCategory.TRANSIENT,
            "AI 服务繁忙，正在等待…",
            "API 速率限制。等待后自动重试。",
        )
    if "401" in msg or "403" in msg or "unauthorized" in msg:
        return (
            ErrorCategory.SYSTEM,
            "AI 服务认证失败，请联系管理员检查 API 密钥。",
            "LLM API 认证失败。检查 DEEPSEEK_API_KEY 配置。",
        )
    return (
        ErrorCategory.SYSTEM,
        f"AI 服务错误: {str(error)[:200]}",
        "LLM API 调用失败。如果持续发生，检查 API 配置。",
    )


def classify_not_found(entity_type: str, entity_id: str) -> tuple[ErrorCategory, str, str]:
    """Standard 'not found' classification."""
    return (
        ErrorCategory.NOT_FOUND,
        f"未找到{entity_type}（ID: {entity_id}）。",
        f"该{entity_type}不存在或已被删除。请使用 list_* 工具确认有效的 ID。",
    )


def classify_validation(field: str, reason: str) -> tuple[ErrorCategory, str, str]:
    """Standard validation error classification."""
    return (
        ErrorCategory.VALIDATION,
        f"参数错误: {field} — {reason}",
        f"参数 '{field}' 不合法: {reason}。请修正后重试。",
    )


def classify_business_rule(rule: str, explanation: str) -> tuple[ErrorCategory, str, str]:
    """Standard business rule violation classification."""
    return (
        ErrorCategory.BUSINESS,
        f"操作被拒绝: {rule}",
        f"违反业务规则「{rule}」: {explanation}。请向用户说明原因。",
    )


# ── Recovery strategies ─────────────────────────────────

MAX_RETRIES = 2

# Tools known to be idempotent (safe to retry)
IDEMPOTENT_TOOLS = {
    "list_resumes", "get_resume", "list_positions", "get_position",
    "get_questions", "get_evaluation", "get_leaderboard", "get_rankings",
    "rag_search", "list_knowledge", "get_knowledge_stats", "get_knowledge_categories",
    "get_operations_dashboard", "get_dashboard_overview", "get_settings",
    "list_probation", "get_probation_stats", "get_probation_employee",
    "list_performance", "get_performance_stats", "get_department_performance",
    "get_grade_distribution", "get_bonus_info", "get_quarter_trends",
    "list_knowledge_bases", "list_documents", "recall_test",
    "get_position_questions",
}


def is_retryable(tool_name: str, category: ErrorCategory) -> bool:
    """Determine if a failed tool call should be automatically retried."""
    if category != ErrorCategory.TRANSIENT:
        return False
    # Only retry read-only / idempotent tools
    return tool_name in IDEMPOTENT_TOOLS


def should_show_confirmation(tool_name: str) -> bool:
    """Determine if a tool requires user confirmation before execution."""
    destructive_tools = {
        "delete_resume", "delete_position", "delete_knowledge_item",
        "delete_knowledge_base", "delete_document",
        "submit_evaluation", "update_probation_status",
        "initiate_appraisal",
    }
    return tool_name in destructive_tools


def get_confirmation_message(tool_name: str, params: dict) -> str:
    """Generate a human-readable confirmation message for a destructive operation."""
    messages = {
        "delete_resume": f"将永久删除候选人「{params.get('id', '')}」及其简历文件。此操作不可撤销。",
        "delete_position": f"将永久删除岗位「{params.get('id', '')}」。如果该岗位下有候选人，操作将被拒绝。",
        "delete_knowledge_item": f"将永久删除知识库素材「{params.get('id', '')}」及其向量数据。",
        "delete_knowledge_base": f"将永久删除知识库「{params.get('id', '')}」及其所有文档和向量。",
        "delete_document": f"将永久删除文档「{params.get('id', '')}」及其向量分块。",
        "submit_evaluation": f"将提交候选人「{params.get('candidateId', '')}」的面试评定，提交后不可修改。",
        "update_probation_status": f"将更新试用期员工「{params.get('employeeId', '')}」的状态为「{params.get('status', '')}」。",
        "initiate_appraisal": f"将为 {params.get('quarter', '')} 季度发起绩效考核（{len(params.get('employeeIds', []))} 人参与）。",
    }
    return messages.get(tool_name, f"确认执行 {tool_name}？此操作可能无法撤销。")
