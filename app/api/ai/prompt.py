"""Agent system prompt — assembled from modular text files.

Prompt content lives in ``prompts/*.txt``.  Edit those files directly when
you need to tweak rules or update status definitions — no Python changes
required.

``prompts/system.txt``   — role, capabilities, status flow, sub-agents
``prompts/tone.txt``    — reply style (Feishu-like: concise, friendly)
``prompts/recommend.txt`` — recommendation business rules
``prompts/rules.txt``    — working rules  (most frequently edited)
"""

from __future__ import annotations

from app.prompts import load_prompt, render_prompt


# ── 角色专属提示词（按账号角色叠加）──────────────────
# app/prompts/agent_workspace/roles/<role_code>.md —— 每个角色一份：
# 身份定位 / 职责视角 / 话术风格 / 补充语境。只影响 LLM 的说话方式与关注点，
# 不改变工具可见性（仍由 visible_tool_names 同源驱动）与共享工作规则（rules.txt）。

def _load_role_sections(role_codes: list[str] | None) -> str:
    """按角色加载统一目录中的专属段，拼为一段（无角色文件返回空串）。

    与共享段分开缓存：任何相关角色文件的 mtime 变化都会触发重载（支持热更新）。
    多角色用户按传入顺序拼接多个角色段。
    """
    if not role_codes:
        return ""
    sections = []
    for code in role_codes:
        try:
            sections.append(load_prompt(f"agent_workspace/roles/{code}.md"))
        except ValueError as exc:
            if "文件不存在" not in str(exc):
                raise
    return "\n\n".join(sections)


# ── Agent Configurations ──────────────────────────────────
# 8 个角色各有独立 agent 入口，agent_id = role_code。
# 与 agent_workspace/roles/<role>.md 一一对应，工具集由权限过滤驱动。

AGENT_CONFIGS = {
    "hr": {
        "name": "HR 招聘助手",
        "description": "负责招聘全流程：岗位发布、简历筛选、面试安排、Offer 流程",
        "icon": "user-search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "ceo": {
        "name": "公司负责人助手",
        "description": "查看人才全局概况、审批核心岗位晋级与期权决策",
        "icon": "crown",
        "iconBg": "#fff3e0",
        "iconColor": "#ff9800",
    },
    "manager": {
        "name": "部门负责人助手",
        "description": "提交用人需求、确认岗位标准、审批录用与转正",
        "icon": "users",
        "iconBg": "#e8f5e9",
        "iconColor": "#4caf50",
    },
    "interviewer": {
        "name": "面试官助手",
        "description": "查看简历、生成面试题、结构化评分与录音上传",
        "icon": "mic",
        "iconBg": "#fce4ec",
        "iconColor": "#e91e63",
    },
    "mentor": {
        "name": "带教人助手",
        "description": "制定培训任务、评价新员工、提交带教记录",
        "icon": "graduation-cap",
        "iconBg": "#e8f5e9",
        "iconColor": "#66bb6a",
    },
    "project_lead": {
        "name": "项目负责人助手",
        "description": "发布工作任务、验收成果、确认积分与处理申诉",
        "icon": "list-todo",
        "iconBg": "#e0f2f1",
        "iconColor": "#26a69a",
    },
    "employee": {
        "name": "员工助手",
        "description": "查看试用任务、培训进度、积分明细与申诉",
        "icon": "user",
        "iconBg": "#f3e5f5",
        "iconColor": "#ab47bc",
    },
    "admin": {
        "name": "系统管理员助手",
        "description": "配置权限规则、业务阈值、AI 模型与审计策略",
        "icon": "settings",
        "iconBg": "#eceff1",
        "iconColor": "#607d8b",
    },
    "equity_committee": {
        "name": "期权委员会助手",
        "description": "期权审批联签、查看人才池与期权候选人",
        "icon": "gem",
        "iconBg": "#fff9c4",
        "iconColor": "#f9a825",
    },
    # 旧 genie/recruit/interview/training/performance 保留向后兼容，未知 id 兜底 employee。
    "genie": {
        "name": "Genie 全能助手",
        "description": "通过对话框，按你的账号权限完成招聘系统各环节操作",
        "icon": "search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "recruit": {
        "name": "招聘 Agent",
        "description": "负责简历解析、人才筛选、岗位匹配",
        "icon": "search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "interview": {
        "name": "面试 Agent",
        "description": "AI 出题、面试分析、录用建议",
        "icon": "interview",
        "iconBg": "#fce4ec",
        "iconColor": "#e91e63",
    },
    "training": {
        "name": "培训 Agent",
        "description": "入职培养、试用期跟踪",
        "icon": "training",
        "iconBg": "#e8f5e9",
        "iconColor": "#4caf50",
    },
    "performance": {
        "name": "绩效 Agent",
        "description": "绩效分析、员工成长建议",
        "icon": "performance",
        "iconBg": "#fff3e0",
        "iconColor": "#ff9800",
    },
}


def build_system_prompt(
    agent_id: str = "genie",
    visible_tool_names: set[str] | None = None,
    role_codes: list[str] | None = None,
) -> str:
    """Build the full system prompt for a given agent.

    从 ``app/prompts/`` 加载提示词，并用严格的 ``[[name]]`` 占位符渲染。

    ``visible_tool_names``：当前用户可见的工具名集合（来自 get_tools_for_agent 的
    fail-closed 过滤）。能力清单由该集合反向驱动（app/agent/tools.py 的
    build_capability_lines），保证 AI 宣称的每项能力都有对应工具支撑——与
    TOOL_PERMISSIONS 强制同源，不再按角色权限点宣称无工具支撑的功能。
    传 None（未接权限的内部调用方）时保持全量能力，作为兜底以免误裁。

    ``role_codes``：当前账号的角色编码列表（如 CurrentUser.roles）。存在
    agent_workspace/roles/<role>.md 存在时，把角色专属段（身份/职责视角/话术/补充语境）
    叠加到共享提示词之后，实现各角色不同的对话模式。角色段只改对话人格，
    不改变工具可见性与共享工作规则。传 None 保持纯共享提示词（向后兼容）。
    """
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["employee"])
    capabilities = _render_capabilities(visible_tool_names)
    base = "\n\n".join((
        render_prompt("agent_workspace/system.md", {
            "name": agent_info["name"],
            "description": agent_info["description"],
            "capabilities": capabilities,
        }),
        load_prompt("agent_workspace/tone.md"),
        load_prompt("agent_workspace/recommend.md"),
        load_prompt("agent_workspace/rules.md"),
    ))
    role_sections = _load_role_sections(role_codes)
    if role_sections:
        base = base + "\n\n" + role_sections
    return base


def _render_capabilities(visible_tool_names: set[str] | None) -> str:
    """渲染 AI 可见能力清单（由可见工具集驱动，与 TOOL_PERMISSIONS 同源）。"""
    from app.agent.tools import build_capability_lines

    lines = build_capability_lines(visible_tool_names)
    if not lines:
        # 极端情况：用户无任何可见工具 → 仍给一句基础定位，避免空白段
        return "- 查询系统信息、处理你能访问的招聘事项；如需更多权限请联系管理员"
    return "\n".join(lines)
