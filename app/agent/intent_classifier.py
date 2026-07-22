"""
Intent pre-classifier + tool gating for the Genie recruitment agent.

Classifies user messages into intent categories using keyword matching
(fast, deterministic, no API call) and returns the appropriate tool subset
so the ReAct agent CANNOT call irrelevant tools — a **hard** constraint.
"""

from __future__ import annotations

import re
from app.agent.tools import TOOL_REGISTRY, DB_TOOL_NAMES, GENERAL_TOOL_NAMES


# ═══════════════════════════════════════════════════════════
#  Tool Groups
# ═══════════════════════════════════════════════════════════

TOOL_GROUPS: dict[str, set[str]] = {
    "position_read": {
        "list_positions", "get_position",
    },
    "position_questions": {
        "get_position_questions", "save_position_questions",
    },
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
    "database": set(DB_TOOL_NAMES),
}

# Per-intent extras (e.g. rag_search only where it helps).
INTENT_EXTRA_TOOLS: dict[str, set[str]] = {
    "position_query": set(),
    "candidate_query": set(),
    "candidate_action": set(),
    "interview": {"rag_search"},
    "probation": set(),
    "performance": set(),
    "knowledge": set(),
    "dashboard": set(),
    "settings": set(),
    "database": set(),
    "general": {"rag_search"},
}

# Intent → tool groups. ``None`` → GENERAL_TOOL_NAMES (not all tools).
INTENT_TO_GROUPS: dict[str, list[str] | None] = {
    "position_query": ["position", "dashboard"],
    "candidate_query": ["candidate", "position_read", "dashboard"],
    "candidate_action": ["candidate"],
    "interview": ["interview", "candidate", "position_read", "position_questions"],
    "probation": ["probation"],
    "performance": ["performance"],
    "knowledge": ["knowledge"],
    "dashboard": ["dashboard", "candidate"],
    "settings": ["settings"],
    "database": ["database"],
    "general": None,
}


# ═══════════════════════════════════════════════════════════
#  Keyword-Based Intent Classification (first match wins)
# ═══════════════════════════════════════════════════════════

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Status writes (highest priority — never treat as interview/question gen) ──
    ("candidate_action", [
        r"\[状态变更\]",
        r"请调用\s*update_resume",
        r"status\s*=\s*(failed|passed|first_interview|second_interview|job_hunting)",
        r"(一面|二面).*(未通过|不通过|淘汰)",
        r"(未通过|不通过|淘汰).*(一面|二面|初筛|筛选)",
        r"(标记|设为|设置为).*(未通过|不通过|淘汰|通过|一面|二面|初筛)",
        r"(把|将|更新|修改|设置).*(状态|为|到|成)",
        r"(上传|导入|添加|新增).*简历",
        r"(删除|移除|重新解析|批量解析).*(简历|候选人)",
        r"(标记|归档|淘汰).*(候选人|简历)",
    ]),
    # ── Count / stats (before broad「候选人」matches) ──
    ("dashboard", [
        r"(多少|几个|有几|共有|总共|总数|数量|统计).*(候选人|简历|人选|岗位|面试)",
        r"(候选人|简历|人选).*(多少|几个|总数|数量|统计|概况|分布)",
        r"(看板|仪表盘|概览|漏斗)",
        r"(运营|招聘).*(数据|统计|概览|漏斗)",
        r"(数据|统计).*(总览|概况|汇总)",
    ]),
    # ── Position queries (JD / CRUD — not candidate list) ──
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
    # ── Candidate reads (before interview — avoid dragging in 出题/题单) ──
    ("candidate_query", [
        r"\[系统·已读库\]",
        r"\[系统·已写库\].*get_resume",
        r"\[查简历\]",
        r"(完整|详细).*(简历|画像|分析)",
        r"(查看|看看|获取).*(完整|详细).*(简历|画像|分析)",
        r"(其他|别的|其余|还有).*(候选人|人选|简历)",
        r"(看看|查看|查|有哪些|还有).*(岗位).*(候选人|人选|简历)",
        r"(岗位).*(有哪些|有什么|几个|多少).*(候选人|人选|简历)",
        r"(查看|看看|查|打开|显示|展示|获取).*(简历|候选人|人选).*(详情|信息)?",
        r"(搜索|筛选|查找|找|推荐).*(候选人|简历|人选)",
        r"(简历|候选人|人选).*(详情|信息|列表|筛选|搜索)",
        r"(谁|哪些人|有没有).*(合适|匹配|符合|应聘)",
        r"匹配.*(候选人|简历|人选|度)",
        r"get_resume", r"list_resumes",
    ]),
    # ── Interview (question gen / evaluation — not status change) ──
    ("interview", [
        r"(生成|创建|出|换|重新).*(面试题|题目|试题|考题)",
        r"(查看|看看|显示|展示).*(面试题|题目|试题|考题|题单)",
        r"(面试|题目).*(评分|打分|评定|评估)",
        r"(给|为|对).*(评分|打分|评定)",
        r"(保存|提交).*(评分|评定|面试)",
        r"AI.*评分", r"面试.*(排行|排名)",
        r"(开始|进行).*(面试评定|面试评估)",
        r"(一面|二面).*(开始|进行|安排).*(面试|评定)?",
    ]),
    ("probation", [
        r"试用期", r"转正", r"入职.*培养",
        r"(试用|实习).*(员工|考核|评估|任务)",
        r"(新增|添加|创建).*(试用|员工)",
        r"(周报|第一周|考核|评估).*(试用)",
    ]),
    ("performance", [
        r"绩效", r"考核.*(季度|发起|结果)",
        r"(季度|年度).*(考核|评估|绩效)",
        r"(奖金|工资|薪酬).*(分配|调整|设置)",
        r"(S|A|B\+?|C).*(等级|绩效)",
        r"(发起|开始).*(考核|评估)",
    ]),
    ("knowledge", [
        r"知识库", r"知识.*(检索|搜索|查询)",
        r"(检索|搜索|查|找).*(文档|资料|知识|制度|规范)",
        r"(上传|添加|创建|删除).*(文档|知识|素材)",
        r"RAG", r"向量.*(检索|搜索)",
        r"(召回|入库|向量化)",
    ]),
    ("database", [
        r"(数据库|数据表|SQL|sql)",
        r"(db_query|db_update|db_list_tables)",
        r"(直接)?(改|查|更新).*(数据库|数据表)",
    ]),
    ("settings", [
        r"(系统|公司).*(设置|配置|信息)",
        r"(修改|更新|设置|配置).*(系统|公司|参数)",
        r"(AI|自动).*(开关|配置|设置)",
        r"(合格|分数|阈值|天数).*(设置|修改|调整)",
    ]),
]


# Skills allowed per intent (empty set = inject none).
INTENT_SKILL_ALLOWLIST: dict[str, set[str] | None] = {
    "candidate_query": {"resume-screening"},
    "candidate_action": set(),
    "dashboard": set(),
    "position_query": set(),
    "interview": {"interview-question-gen", "interview-evaluation"},
    "probation": {"probation-assessment"},
    "performance": {"performance-analysis"},
    "knowledge": {"knowledge-search"},
    "settings": set(),
    "database": set(),
    "general": None,  # no filter
}


def filter_skills_for_intent(intent: str, skills: list) -> list:
    """Drop skills that don't match the classified intent."""
    allowed = INTENT_SKILL_ALLOWLIST.get(intent)
    if allowed is None:
        return skills
    return [s for s in skills if getattr(s, "name", None) in allowed]


def classify_intent(message: str) -> str:
    """Classify the user message into an intent category using keyword matching."""
    if not message or not message.strip():
        return "general"

    normalized = message.strip()

    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return intent

    return "general"


# Human-readable hints injected into system prompt per intent.
INTENT_GATE_HINTS: dict[str, str] = {
    "position_query": "仅查岗位/JD，禁止 list_resumes / get_resume / 出题。",
    "candidate_query": (
        "仅查候选人信息；统计总数用 get_operations_dashboard。"
        "完整简历/详情 → get_resume(id, view=full)；无 ID 时先 list_resumes(keyword=姓名)。"
        "**禁止** get_position_questions（岗位通用题库，参数是 positionId+round，不是简历）。"
        "也禁止 get_questions / generate_questions，除非用户明确要看题单。"
    ),
    "candidate_action": (
        "仅改候选人状态/上传/删除；有 ID 时直接 update_resume。"
        "禁止 generate_questions / get_questions / rag_search / get_evaluation。"
    ),
    "interview": "面试题/评定相关；改状态用 update_resume 而非本意图。",
    "dashboard": "仅统计/看板；用 get_operations_dashboard 回答总数/分布。",
    "knowledge": "仅知识库检索与管理。",
    "database": "仅数据库直查/直改工具。",
    "settings": "仅系统设置读写。",
}


def get_tool_names_for_intent(intent: str) -> set[str]:
    """Resolve an intent label to the set of allowed tool names."""
    group_names = INTENT_TO_GROUPS.get(intent)
    if group_names is None:
        return set(GENERAL_TOOL_NAMES)

    names: set[str] = set(INTENT_EXTRA_TOOLS.get(intent, set()))
    for gn in group_names:
        names |= TOOL_GROUPS.get(gn, set())
    return names


def get_tool_defs_for_intent(intent: str) -> list[dict]:
    """Resolve an intent label to full tool definition dicts."""
    allowed = get_tool_names_for_intent(intent)
    return [TOOL_REGISTRY[name] for name in allowed if name in TOOL_REGISTRY]
