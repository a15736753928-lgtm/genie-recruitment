"""Display-hint mapping — tells the frontend how to render each tool result.

历史上这里还有一个 ``ToolResultAdapter`` 类（execute() + 一堆正则提取器）和
配套的第二套 ToolResult/ToolError 定义（output/schemas.py），两者都没有任何
调用点：实际在用的结构化结果类型是 ``app/agent/tool_result.py``，
而唯一被消费的只有下面这张表（app/agent/graph.py::_display_hint_for）。
已整体删除，只留这张表。
"""

from __future__ import annotations


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
    # 知识库/RAG 工具已断开（2026-08-01）——适配项一并移除，可随工具恢复
    # Dashboard
    "get_operations_dashboard": "stats",
    # Settings
    "get_settings": "text",
    "update_settings": "text",
}
