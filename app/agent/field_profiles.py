"""
Intent-aware field projection for Agent tools.

Principle: return only the fields needed for the current turn —
not a fixed dump of everything, and not a one-line stub that hides
writable columns.

The model (or intent heuristics) picks a *view* or an explicit field
list; the executor projects the full API payload onto that set.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# ── Field catalogs (camelCase keys as returned by API layers) ──

POSITION_FIELDS: dict[str, str] = {
    "id": "岗位ID",
    "name": "名称",
    "department": "部门",
    "educationRequirement": "学历要求（独立字段，修改时用 educationRequirement）",
    "experienceRequirement": "经验要求（独立字段 experienceRequirement）",
    "ageRequirement": "年龄要求（独立字段 ageRequirement）",
    "salaryRange": "薪资范围（独立字段 salaryRange）",
    "jdResponsibilities": "岗位职责正文",
    "jdRequirements": "任职要求正文（自由文本，勿把学历等专项要求只写这里）",
    "jdPreferred": "加分项",
    "jdTechStack": "技术栈",
    "screeningCriteria": "筛选评分标准",
    "interviewCriteriaR1": "一面评分标准",
    "interviewCriteriaR2": "二面评分标准",
    "week1ProjectRequirement": "试用期第一周项目要求",
    "weeks24Plan": "试用期2-4周计划",
    "laterWeekScoring": "试用期后期评分",
    "conversionCriteria": "转正标准",
}

RESUME_FIELDS: dict[str, str] = {
    "id": "候选人ID",
    "name": "姓名",
    "position": "应聘岗位",
    "positionId": "岗位ID",
    "score": "匹配分",
    "status": "状态",
    "gender": "性别",
    "age": "年龄",
    "ethnicity": "民族",
    "nativePlace": "籍贯",
    "education": "学历",
    "experience": "经验",
    "phone": "电话",
    "email": "邮箱",
    "skills": "技能",
    "educationHistory": "教育经历",
    "workHistory": "工作经历",
    "projectHistory": "项目经历",
    "aiAnalysis": "AI分析",
    "screeningAiScore": "筛选AI分",
    "screeningManualConfirmed": "筛选是否人工确认",
    "screeningConfirmedBy": "筛选确认人",
    "interviewer": "面试官",
    "interviewRound": "面试轮次",
    "uploadTime": "上传时间",
    "resumeFile": "简历文件",
}

PROBATION_FIELDS: dict[str, str] = {
    "id": "员工ID",
    "name": "姓名",
    "positionName": "岗位",
    "positionId": "岗位ID",
    "department": "部门",
    "status": "状态",
    "gender": "性别",
    "age": "年龄",
    "joinDate": "入职日",
    "probationEnd": "试用期截止日",
    "mentorName": "导师",
    "taskProgress": "任务进度%",
    "totalTasks": "任务总数",
    "completedTasks": "已完成任务数",
    "week1Score": "第一周得分",
    "week1Passed": "第一周是否通过",
    "conversionScore": "转正得分",
    "conversionDecision": "转正结论",
    "aiScore": "AI综合分",
    "aiResult": "AI结论",
    "tasks": "任务列表",
    "week1Assessment": "第一周评估明细",
    "conversion": "转正评估明细",
}

# Settings are a flat dict of many keys — catalog = known defaults.
SETTINGS_FIELDS: dict[str, str] = {
    "systemName": "系统名称",
    "companyName": "公司名",
    "contactEmail": "联系邮箱",
    "defaultQuarter": "默认季度",
    "autoParseResume": "自动解析简历",
    "minMatchScore": "最低匹配分",
    "defaultPositionId": "默认岗位ID",
    "offerApprovalRequired": "Offer需审批",
    "defaultQuestionCount": "默认出题数",
    "defaultScoringMode": "默认评分模式",
    "passScoreThreshold": "合格分数线",
    "allowAudioUpload": "允许音频上传",
    "probationDays": "试用期天数",
    "defaultProbationTasks": "默认试用任务数",
    "aiResumeAnalysis": "AI简历分析开关",
    "aiQuestionGeneration": "AI出题开关",
    "aiInterviewScoring": "AI面试评分开关",
    "recallThreshold": "知识库召回阈值",
    "kbDefaultTopK": "知识库默认返回条数",
    "kbRerankEnabled": "知识库重排序开关",
    "kbOcrEnabled": "知识库OCR开关",
    "notifyNewResume": "新简历通知",
    "notifyInterviewReminder": "面试提醒通知",
    "notifyOfferPending": "Offer待审通知",
    "notifyProbationRisk": "试用期风险通知",
    "notifyPerformanceDue": "绩效到期通知",
    "dataRetentionDays": "数据保留天数",
    "exportFormat": "导出格式",
    "webhookEnabled": "Webhook开关",
    "webhookUrl": "Webhook地址",
}


# ── Named views (intent presets) ─────────────────────────────
# Always includes id + name-like identity keys so follow-up writes work.

POSITION_VIEWS: dict[str, list[str]] = {
    "summary": ["id", "name", "department", "educationRequirement", "experienceRequirement", "salaryRange"],
    "core": [
        "id", "name", "department",
        "educationRequirement", "experienceRequirement", "ageRequirement", "salaryRange",
        "jdResponsibilities", "jdRequirements", "jdPreferred", "jdTechStack",
    ],
    "jd": ["id", "name", "department", "jdResponsibilities", "jdRequirements", "jdPreferred", "jdTechStack"],
    "requirements": [
        "id", "name",
        "educationRequirement", "experienceRequirement", "ageRequirement", "salaryRange",
        "jdRequirements",
    ],
    "edit": [
        "id", "name", "department",
        "educationRequirement", "experienceRequirement", "ageRequirement", "salaryRange",
    ],
    "criteria": ["id", "name", "screeningCriteria", "interviewCriteriaR1", "interviewCriteriaR2"],
    "probation_plan": [
        "id", "name",
        "week1ProjectRequirement", "weeks24Plan", "laterWeekScoring", "conversionCriteria",
    ],
    "full": list(POSITION_FIELDS.keys()),
}

RESUME_VIEWS: dict[str, list[str]] = {
    "summary": ["id", "name", "position", "score", "status", "education", "experience", "skills"],
    "core": [
        "id", "name", "position", "positionId", "score", "status",
        "gender", "age", "education", "experience", "phone", "email", "skills",
    ],
    "detail": [
        "id", "name", "position", "positionId", "score", "status",
        "gender", "age", "ethnicity", "nativePlace", "education", "experience",
        "phone", "email", "skills",
        "educationHistory", "workHistory", "projectHistory", "aiAnalysis",
    ],
    "contact": ["id", "name", "phone", "email", "status"],
    "screening": [
        "id", "name", "score", "status",
        "screeningAiScore", "screeningManualConfirmed", "screeningConfirmedBy",
    ],
    "interview": ["id", "name", "status", "interviewer", "interviewRound", "score"],
    "full": list(RESUME_FIELDS.keys()),
}

PROBATION_VIEWS: dict[str, list[str]] = {
    "summary": ["id", "name", "positionName", "status", "taskProgress", "mentorName"],
    "core": [
        "id", "name", "positionName", "positionId", "department", "status",
        "joinDate", "probationEnd", "mentorName",
        "taskProgress", "week1Score", "week1Passed",
        "conversionScore", "conversionDecision", "aiScore", "aiResult",
    ],
    "tasks": ["id", "name", "status", "taskProgress", "totalTasks", "completedTasks", "tasks"],
    "week1": ["id", "name", "week1Score", "week1Passed", "week1Assessment"],
    "conversion": ["id", "name", "conversionScore", "conversionDecision", "conversion", "aiScore", "aiResult"],
    "full": list(PROBATION_FIELDS.keys()),
}

SETTINGS_VIEWS: dict[str, list[str]] = {
    "summary": ["companyName", "systemName", "aiResumeAnalysis", "minMatchScore", "passScoreThreshold"],
    "core": [
        "companyName", "systemName", "contactEmail", "defaultQuarter",
        "autoParseResume", "minMatchScore", "passScoreThreshold",
        "probationDays", "aiResumeAnalysis", "aiQuestionGeneration", "aiInterviewScoring",
    ],
    "scoring": ["minMatchScore", "passScoreThreshold", "defaultScoringMode", "defaultQuestionCount"],
    "ai": ["aiResumeAnalysis", "aiQuestionGeneration", "aiInterviewScoring", "autoParseResume"],
    "knowledge": [
        "recallThreshold", "kbDefaultTopK", "kbRerankEnabled", "kbOcrEnabled",
    ],
    "notify": [
        "notifyNewResume", "notifyInterviewReminder", "notifyOfferPending",
        "notifyProbationRisk", "notifyPerformanceDue",
    ],
    "full": list(SETTINGS_FIELDS.keys()),
}


# Keyword → preferred view (used when model omits view/fields)
_INTENT_VIEW_HINTS: list[tuple[str, str, str]] = [
    # (entity, regex-ish keyword fragment, view)
    ("position", "学历|经验|年龄|薪资|薪酬", "requirements"),
    ("position", "改|修改|更新|设置|调整", "edit"),
    ("position", "评分标准|筛选标准|面试标准", "criteria"),
    ("position", "试用期|转正|培养", "probation_plan"),
    ("position", "职责|JD|任职要求|加分|技术栈", "jd"),
    ("resume", "电话|邮箱|联系", "contact"),
    ("resume", "筛选|确认", "screening"),
    ("resume", "面试官|轮次|一面|二面", "interview"),
    ("resume", "详情|完整|简历|经历", "detail"),
    ("probation", "任务", "tasks"),
    ("probation", "第一周|周考", "week1"),
    ("probation", "转正", "conversion"),
    ("probation", "改|修改|更新|导师|入职", "core"),
    ("settings", "分数|合格|匹配|阈值", "scoring"),
    ("settings", "AI|解析|出题|评分", "ai"),
    ("settings", "通知|提醒", "notify"),
    ("settings", "改|修改|更新|设置", "core"),
]


def resolve_fields(
    entity: str,
    *,
    view: Optional[str] = None,
    fields: Optional[Iterable[str]] = None,
    purpose: str = "",
    default_view: str = "core",
) -> list[str]:
    """Resolve which fields to include for this tool call.

    Priority:
      1. Explicit ``fields`` list from the model
      2. Explicit ``view`` name
      3. Keyword heuristics on ``purpose`` (user message / free text)
      4. ``default_view``
    """
    catalogs = {
        "position": (POSITION_FIELDS, POSITION_VIEWS),
        "resume": (RESUME_FIELDS, RESUME_VIEWS),
        "probation": (PROBATION_FIELDS, PROBATION_VIEWS),
        "settings": (SETTINGS_FIELDS, SETTINGS_VIEWS),
    }
    if entity not in catalogs:
        return []
    catalog, views = catalogs[entity]

    # 1) explicit fields
    if fields:
        requested = [f for f in fields if f and f != "*"]
        if "*" in list(fields) or "all" in [str(f).lower() for f in fields]:
            return list(catalog.keys())
        # Keep only known fields; always ensure identity keys first
        known = [f for f in requested if f in catalog]
        if not known:
            known = list(views.get(default_view, views.get("core", [])))
        return _with_identity(entity, known)

    # 2) explicit view
    if view:
        v = view.strip().lower()
        if v in ("*", "all", "full"):
            return list(catalog.keys())
        if v in views:
            return list(views[v])
        # unknown view → default
        return list(views.get(default_view, []))

    # 3) purpose heuristics
    if purpose:
        for ent, pattern, hinted_view in _INTENT_VIEW_HINTS:
            if ent != entity:
                continue
            if re_search_any(pattern, purpose):
                return list(views.get(hinted_view, views[default_view]))

    # 4) default
    return list(views.get(default_view, []))


def re_search_any(pattern: str, text: str) -> bool:
    import re
    return bool(re.search(pattern, text or "", re.IGNORECASE))


def _with_identity(entity: str, fields: list[str]) -> list[str]:
    """Ensure id + primary display name are always present for follow-up writes."""
    identity = {
        "position": ["id", "name"],
        "resume": ["id", "name"],
        "probation": ["id", "name"],
        "settings": [],
    }.get(entity, ["id"])
    ordered: list[str] = []
    for f in identity + fields:
        if f not in ordered:
            ordered.append(f)
    return ordered


def format_projected(
    data: dict[str, Any],
    fields: list[str],
    labels: dict[str, str],
    *,
    title: str = "",
    max_text: int = 800,
    max_json: int = 600,
) -> str:
    """Render selected fields as compact Chinese lines for the LLM."""
    lines: list[str] = []
    if title:
        lines.append(title)

    for key in fields:
        if key not in data and key not in labels:
            continue
        val = data.get(key)
        if val is None or val == "" or val == [] or val == {}:
            # Still show explicit empty for edit views so model knows the key exists
            if key.endswith("Requirement") or key in (
                "salaryRange", "mentorName", "status", "education", "experience",
            ):
                lines.append(f"  {labels.get(key, key)}: （空）  [字段: {key}]")
            continue

        label = labels.get(key, key)

        if isinstance(val, (dict, list)):
            text = json_compact(val, max_json)
            lines.append(f"  {label} [字段: {key}]:\n{text}")
        else:
            text = str(val)
            if len(text) > max_text:
                text = text[:max_text] + "…"
            # Mark writable independent columns clearly
            if key in (
                "educationRequirement", "experienceRequirement",
                "ageRequirement", "salaryRange", "minMatchScore",
                "passScoreThreshold", "probationDays",
            ):
                lines.append(f"  {label}: {text}  [字段: {key}]")
            else:
                lines.append(f"  {label}: {text}")

    if len(lines) <= (1 if title else 0):
        lines.append("  （所选字段均无数据）")
    return "\n".join(lines)


def json_compact(val: Any, max_len: int) -> str:
    import json
    try:
        s = json.dumps(val, ensure_ascii=False, indent=2)
    except Exception:
        s = str(val)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def parse_fields_param(params: dict) -> tuple[Optional[str], Optional[list[str]], str]:
    """Extract view / fields / purpose from a tool-call params dict."""
    view = params.get("view") or params.get("detail") or None
    if isinstance(view, str):
        view = view.strip() or None
    else:
        view = None

    raw_fields = params.get("fields")
    fields: Optional[list[str]] = None
    if isinstance(raw_fields, list):
        fields = [str(f).strip() for f in raw_fields if f]
    elif isinstance(raw_fields, str) and raw_fields.strip():
        # allow "a,b,c" or single field name
        fields = [p.strip() for p in raw_fields.split(",") if p.strip()]

    purpose = str(params.get("purpose") or params.get("query") or params.get("reason") or "")
    return view, fields, purpose
