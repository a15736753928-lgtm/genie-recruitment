"""
Intent pre-classifier + tool gating for the Genie recruitment agent.

Classifies user messages into intent categories using keyword matching
(fast, deterministic, no API call) and returns the appropriate tool subset
so the ReAct agent CANNOT call irrelevant tools — a **hard** constraint.

Principle
---------
Before:  User msg → ReAct Agent (ALL 60 tools) → LLM picks any tool
After:   User msg → Keyword Classifier → ReAct Agent (restricted tools)

The ``position_query`` intent is deliberately the most restrictive: it gets
*only* position tools + common tools.  Candidate tools (list_resumes /
get_resume) are NOT included, which directly fixes the reported bug where
"查看Agent工程师岗位JD" would trigger irrelevant candidate queries.
"""

from __future__ import annotations

import re
from app.agent.tools import TOOL_REGISTRY


# ═══════════════════════════════════════════════════════════
#  Tool Groups
# ═══════════════════════════════════════════════════════════

TOOL_GROUPS: dict[str, set[str]] = {
    "position": {
        "list_positions", "get_position", "create_position",
        "update_position", "delete_position",
        "get_position_questions", "save_position_questions",
    },
    "candidate": {
        "list_resumes", "get_resume", "update_resume", "upload_resume",
        "batch_parse_resumes", "reanalyze_resume", "delete_resume",
    },
    "interview": {
        "get_questions", "generate_questions", "save_questions",
        "replace_question", "get_evaluation", "save_evaluation",
        "submit_evaluation", "ai_score_question",
        "get_leaderboard", "get_rankings",
    },
    "probation": {
        "list_probation", "get_probation_stats", "get_probation_employee",
        "create_probation_employee", "save_week1_assessment", "save_conversion",
        "create_probation_task", "update_probation_task",
        "ai_evaluate_probation", "update_probation_status", "manual_review_probation",
    },
    "performance": {
        "list_performance", "get_performance_stats",
        "get_department_performance", "get_grade_distribution",
        "get_bonus_info", "get_quarter_trends",
        "initiate_appraisal", "update_bonus",
    },
    "knowledge": {
        "rag_search", "list_knowledge", "get_knowledge_stats",
        "get_knowledge_categories", "upload_knowledge_file",
        "create_knowledge_item", "update_knowledge_item", "delete_knowledge_item",
        "recall_test", "list_knowledge_bases", "create_knowledge_base",
        "update_knowledge_base", "delete_knowledge_base",
        "upload_document", "list_documents", "delete_document",
    },
    "dashboard": {
        "get_operations_dashboard", "get_dashboard_overview",
    },
    "settings": {
        "get_settings", "update_settings",
    },
}

# Tools always available regardless of intent
COMMON_TOOLS: set[str] = {"rag_search"}

# Intent → tool groups mapping.  ``None`` means "all tools" (general intent).
INTENT_TO_GROUPS: dict[str, list[str] | None] = {
    "position_query":    ["position"],
    "candidate_query":   ["candidate", "position"],
    "candidate_action":  ["candidate", "position"],
    "interview":         ["interview", "candidate", "position"],
    "probation":         ["probation"],
    "performance":       ["performance"],
    "knowledge":         ["knowledge"],
    "dashboard":         ["dashboard"],
    "settings":          ["settings"],
    "general":           None,   # → all tools
}


# ═══════════════════════════════════════════════════════════
#  Keyword-Based Intent Classification
# ═══════════════════════════════════════════════════════════

# Ordered by priority — first match wins.
# Patterns are case-insensitive regex, matched against the user message.
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Position queries (MUST come before candidate — "岗位" is the discriminator) ──
    ("position_query", [
        r"岗位.*JD", r"JD.*岗位",
        r"(查看|看看|查|打开|显示|展示).*岗位.*(JD|职责|要求|描述|详情|信息)",
        r"(查看|看看|查|打开|显示).*(JD|职位描述|岗位描述|任职要求|岗位职责)",
        r"(什么|哪些|有什么).*岗位",
        r"岗位.*(职责|要求|JD|详情|部门|信息|题库|题目)",
        r"(任职要求|岗位职责|岗位JD|职位描述|加分项|技术栈)",
        r"招聘岗位", r"岗位列表", r"岗位库",
        r"(创建|新增|添加|修改|更新|删除).*岗位",
    ]),
    # ── Candidate actions (write ops — check before candidate_query reads) ──
    ("candidate_action", [
        r"(把|将|更新|修改|设置).*(状态|为|到|成)",
        r"(上传|导入|添加|新增).*简历",
        r"(删除|移除|重新解析|批量解析).*(简历|候选人)",
        r"(标记|归档|淘汰|通过).*(候选人|简历)",
    ]),
    # ── Candidate queries ──
    ("candidate_query", [
        r"(查看|看看|查|打开|显示|展示).*(简历|候选人|人选|求职)",
        r"(搜索|筛选|查找|找|推荐).*(候选人|简历|人选)",
        r"(简历|候选人|人选).*(详情|信息|列表|筛选|搜索)",
        r"(谁|哪些人|有没有).*(合适|匹配|符合|应聘)",
        r"匹配.*(候选人|简历|人选|度)",
        r"(候选人|简历|人选|求职者)",
    ]),
    # ── Interview ──
    ("interview", [
        r"(生成|创建|出|换|重新).*(面试|题目|试题|考题)",
        r"(查看|看看|显示|展示).*(面试|题目|试题|考题|题单)",
        r"(面试|题目).*(评分|打分|评定|评估)",
        r"(给|为|对).*(评分|打分|评定)",
        r"(保存|提交).*(评分|评定|面试)",
        r"AI.*评分", r"面试.*(排行|排名)",
        r"(一面|二面|面试).*(开始|进行|安排|评定)",
    ]),
    # ── Probation ──
    ("probation", [
        r"试用期", r"转正", r"入职.*培养",
        r"(试用|实习).*(员工|考核|评估|任务)",
        r"(新增|添加|创建).*(试用|员工)",
        r"(周报|第一周|考核|评估).*(试用)",
    ]),
    # ── Performance ──
    ("performance", [
        r"绩效", r"考核.*(季度|发起|结果)",
        r"(季度|年度).*(考核|评估|绩效)",
        r"(奖金|工资|薪酬).*(分配|调整|设置)",
        r"(S|A|B\+?|C).*(等级|绩效)",
        r"(发起|开始).*(考核|评估)",
    ]),
    # ── Knowledge ──
    ("knowledge", [
        r"知识库", r"知识.*(检索|搜索|查询)",
        r"(检索|搜索|查|找).*(文档|资料|知识|制度|规范)",
        r"(上传|添加|创建|删除).*(文档|知识|素材)",
        r"RAG", r"向量.*(检索|搜索)",
        r"(召回|入库|向量化)",
    ]),
    # ── Dashboard ──
    ("dashboard", [
        r"(看板|仪表盘|概览|统计.*(数据|总览))",
        r"(运营|招聘).*(数据|统计|概览|漏斗)",
        r"(数据|统计).*(总览|概况|汇总)",
    ]),
    # ── Settings ──
    ("settings", [
        r"(系统|公司).*(设置|配置|信息)",
        r"(修改|更新|设置|配置).*(系统|公司|参数)",
        r"(AI|自动).*(开关|配置|设置)",
        r"(合格|分数|阈值|天数).*(设置|修改|调整)",
    ]),
]


def classify_intent(message: str) -> str:
    """Classify the user message into an intent category using keyword matching.

    Fast (sub-millisecond), deterministic, no API call.  Patterns are checked
    in priority order — the first match wins.

    Returns:
        An intent label (e.g. ``"position_query"``, ``"general"``).
    """
    if not message or not message.strip():
        return "general"

    normalized = message.strip()

    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return intent

    return "general"


# ═══════════════════════════════════════════════════════════
#  Tool Gating
# ═══════════════════════════════════════════════════════════

def get_tool_names_for_intent(intent: str) -> set[str]:
    """Resolve an intent label to the set of allowed tool names.

    ``"general"`` returns *all* registered tool names (no restriction).
    """
    group_names = INTENT_TO_GROUPS.get(intent)
    if group_names is None:
        # general → all tools
        return set(TOOL_REGISTRY.keys())

    names: set[str] = set(COMMON_TOOLS)
    for gn in group_names:
        names |= TOOL_GROUPS.get(gn, set())
    return names


def get_tool_defs_for_intent(intent: str) -> list[dict]:
    """Resolve an intent label to the list of full tool definition dicts.

    These can be passed directly to ``create_langchain_tools_from_defs``.
    """
    allowed = get_tool_names_for_intent(intent)
    return [TOOL_REGISTRY[name] for name in allowed if name in TOOL_REGISTRY]
