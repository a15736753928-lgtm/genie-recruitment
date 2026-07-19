"""
ToolResultAdapter — bridges legacy string tool results to structured ToolResult.

Wraps ``execute_tool_call`` so that existing tool handlers (which return
plain strings) are automatically promoted to structured ``ToolResult``
objects with proper display hints and metadata.

Over time, individual tool handlers in tool_executor.py can be migrated
to return ToolResult directly — at which point the adapter passes them
through unchanged.
"""

from __future__ import annotations

import re
import json
from typing import Callable, Awaitable

from app.agent_os.output.schemas import ToolResult, ToolError


class ToolResultAdapter:
    """Wraps a legacy tool executor to return structured ToolResult.

    Usage::

        adapter = ToolResultAdapter()
        async for tool_name, params in tool_calls:
            result = await adapter.execute(tool_name, params, legacy_executor, db)
            # result is always a ToolResult
    """

    # ── Display hint maps ─────────────────────────────────

    # Full mapping for all 60+ tools
    DISPLAY_HINTS: dict[str, str] = {
        # Resume / Candidate
        "list_resumes": "list",
        "get_resume": "card",
        "update_resume": "card",
        "upload_resume": "card",
        "batch_parse_resumes": "text",
        "reanalyze_resume": "card",
        "delete_resume": "text",
        # Position
        "list_positions": "list",
        "get_position": "card",
        "create_position": "card",
        "update_position": "card",
        "delete_position": "text",
        "get_position_questions": "questions",
        "save_position_questions": "questions",
        # Interview
        "get_questions": "questions",
        "generate_questions": "questions",
        "get_evaluation": "score",
        "ai_score_question": "score",
        "get_leaderboard": "table",
        "save_questions": "questions",
        "replace_question": "questions",
        "save_evaluation": "score",
        "submit_evaluation": "score",
        "get_rankings": "table",
        # Probation
        "list_probation": "list",
        "get_probation_stats": "stats",
        "get_probation_employee": "card",
        "create_probation_employee": "card",
        "save_week1_assessment": "score",
        "save_conversion": "score",
        "create_probation_task": "card",
        "update_probation_task": "card",
        "ai_evaluate_probation": "score",
        "update_probation_status": "text",
        "manual_review_probation": "score",
        # Performance
        "list_performance": "list",
        "get_performance_stats": "stats",
        "get_department_performance": "stats",
        "get_grade_distribution": "stats",
        "get_bonus_info": "stats",
        "get_quarter_trends": "stats",
        "initiate_appraisal": "text",
        "update_bonus": "text",
        # Knowledge / RAG
        "rag_search": "list",
        "list_knowledge": "list",
        "get_knowledge_stats": "stats",
        "get_knowledge_categories": "list",
        "upload_knowledge_file": "text",
        "create_knowledge_item": "card",
        "update_knowledge_item": "card",
        "delete_knowledge_item": "text",
        "recall_test": "list",
        "list_knowledge_bases": "list",
        "create_knowledge_base": "card",
        "update_knowledge_base": "card",
        "delete_knowledge_base": "text",
        "upload_document": "text",
        "list_documents": "list",
        "delete_document": "text",
        # Dashboard
        "get_operations_dashboard": "stats",
        "get_dashboard_overview": "stats",
        # Settings
        "get_settings": "text",
        "update_settings": "text",
    }

    # ── Tool result extractors (parse structured data from strings) ──

    @staticmethod
    def _extract_list_data(result_text: str) -> dict | None:
        """Try to extract list items from a text result."""
        items = []
        for line in result_text.split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                # Extract ID and name from "[uuid] Name | ..."
                match = re.match(r"\[([^\]]+)\]\s*(.+?)(?:\s*\|.*)?$", line)
                if match:
                    items.append({"id": match.group(1), "label": match.group(2).strip()[:80]})
        if items:
            return {"items": items, "count": len(items)}
        return None

    @staticmethod
    def _extract_stats_data(result_text: str) -> dict | None:
        """Try to extract key-value stats from a text result."""
        stats = {}
        for line in result_text.split("\n"):
            line = line.strip()
            # Pattern: "总 10，考核中 5，通过 3，未通过 2"
            for match in re.finditer(r"(\S+?)\s*[：:]\s*(\S+)", line):
                key = match.group(1)
                val = match.group(2)
                try:
                    stats[key] = int(val)
                except ValueError:
                    stats[key] = val
        if stats:
            return {"stats": stats}
        return None

    @staticmethod
    def _extract_score_data(result_text: str) -> dict | None:
        """Try to extract numeric scores from a text result."""
        scores = re.findall(r"(\d+(?:\.\d+)?)\s*分", result_text)
        if scores:
            return {"scores": [float(s) for s in scores], "count": len(scores)}
        return None

    @staticmethod
    def _extract_error(result_text: str) -> ToolError | None:
        """Detect if a string result represents an error."""
        error_patterns = [
            (r"(?:工具执行错误|查询失败|无权限)[：:]\s*(.+)", "external_error"),
            (r"(?:创建失败|更新失败|删除失败|上传失败)[：:]\s*(.+)", "external_error"),
            (r"(?:未找到|不存在|暂无|没有找到)", "not_found"),
        ]
        for pattern, category in error_patterns:
            match = re.search(pattern, result_text)
            if match:
                msg = match.group(1) if match.lastindex else match.group(0)
                return ToolError(
                    category=category,
                    message=msg[:200],
                    retryable=category != "not_found",
                    suggested_actions=_suggest_actions(category, result_text),
                )
        return None

    # ── Main adapter method ──────────────────────────────

    async def execute(
        self,
        tool_name: str,
        params: dict,
        legacy_executor: Callable[..., Awaitable[str]],
        db,
    ) -> ToolResult:
        """Execute a tool and return a structured ToolResult.

        If ``legacy_executor`` already returns a ToolResult, pass it through.
        Otherwise, wrap the string result with proper display hint and data
        extraction.
        """
        raw = await legacy_executor(tool_name, params, db)

        # Pass-through if already structured
        if isinstance(raw, ToolResult):
            return raw

        # Already a dict with 'success' key → coerce
        if isinstance(raw, dict) and "success" in raw:
            return ToolResult(**raw)

        text = str(raw) if raw else ""

        # Detect error
        error = self._extract_error(text)

        # Determine display hint
        hint = self.DISPLAY_HINTS.get(tool_name, "text")

        # Extract structured data based on hint
        data = None
        if not error:
            if hint == "list":
                data = self._extract_list_data(text)
            elif hint == "stats":
                data = self._extract_stats_data(text)
            elif hint == "score":
                data = self._extract_score_data(text)
            elif hint == "questions":
                data = self._extract_list_data(text)

        # Build summary (first meaningful line, max 150 chars)
        summary_line = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                summary_line = stripped[:150]
                break
        if not summary_line:
            summary_line = text[:150] if text else f"{tool_name} 执行完成"

        return ToolResult(
            success=error is None,
            tool_name=tool_name,
            summary=summary_line,
            display_hint=hint,
            data=data or {"raw_text": text},
            error=error,
        )


# ── Helpers ────────────────────────────────────────────────

def _suggest_actions(category: str, result_text: str) -> list[str]:
    """Suggest recovery actions based on error category."""
    suggestions: dict[str, list[str]] = {
        "not_found": ["检查 ID 是否正确", "尝试使用 list 类工具查找"],
        "permission_denied": ["联系管理员获取权限"],
        "validation_error": ["检查参数格式", "确保必填字段已填写"],
        "external_error": ["稍后重试", "检查服务状态"],
        "timeout": ["减少查询范围", "稍后重试"],
    }
    return suggestions.get(category, ["稍后重试"])

