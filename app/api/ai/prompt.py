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

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load(filename: str) -> str:
    """Read a prompt file, stripping leading/trailing whitespace."""
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


_PROMPT_MTIMES: dict[str, float] = {}
_PROMPT_CACHE: str | None = None


def _load_all() -> str:
    """Load and join all prompt sections; reload when any .txt file changes."""
    global _PROMPT_CACHE
    filenames = ("system.txt", "tone.txt", "recommend.txt", "rules.txt")
    mtimes = {
        name: os.path.getmtime(os.path.join(_PROMPTS_DIR, name)) for name in filenames
    }
    if _PROMPT_CACHE is not None and mtimes == _PROMPT_MTIMES:
        return _PROMPT_CACHE
    _PROMPT_MTIMES.clear()
    _PROMPT_MTIMES.update(mtimes)
    sections = [_load(name) for name in filenames]
    _PROMPT_CACHE = "\n\n".join(sections)
    return _PROMPT_CACHE


# ── 角色专属提示词（按账号角色叠加）──────────────────
# prompts/roles/<role_code>.txt —— 每个角色一份，做「对话模式」差异化：
# 身份定位 / 职责视角 / 话术风格 / 补充语境。只影响 LLM 的说话方式与关注点，
# 不改变工具可见性（仍由 visible_tool_names 同源驱动）与共享工作规则（rules.txt）。

_ROLE_DIR = os.path.join(_PROMPTS_DIR, "roles")
_ROLE_CACHE_KEY: tuple | None = None
_ROLE_CACHE: str | None = None


def _role_file_map() -> dict[str, str]:
    """返回 {role_code: absolute_path}，仅含 roles/ 下已存在的 .txt 文件。"""
    result: dict[str, str] = {}
    if not os.path.isdir(_ROLE_DIR):
        return result
    for name in os.listdir(_ROLE_DIR):
        if name.endswith(".txt"):
            result[name[:-4]] = os.path.join(_ROLE_DIR, name)
    return result


def _load_role_sections(role_codes: list[str] | None) -> str:
    """按角色加载 prompts/roles/<role>.txt 专属段，拼为一段（无角色文件返回空串）。

    与共享段分开缓存：任何相关角色文件的 mtime 变化都会触发重载（支持热更新）。
    多角色用户按传入顺序拼接多个角色段。
    """
    global _ROLE_CACHE, _ROLE_CACHE_KEY
    if not role_codes:
        return ""
    role_files = _role_file_map()
    want = [c for c in role_codes if c in role_files]
    if not want:
        return ""
    sig = tuple((c, os.path.getmtime(role_files[c])) for c in want)
    if _ROLE_CACHE is not None and _ROLE_CACHE_KEY == sig:
        return _ROLE_CACHE
    sections = []
    for code in want:
        with open(role_files[code], "r", encoding="utf-8") as f:
            sections.append(f.read().strip())
    text = "\n\n".join(sections)
    _ROLE_CACHE = text
    _ROLE_CACHE_KEY = sig
    return text


# ── Agent Configurations ──────────────────────────────────
# 8 个角色各有独立 agent 入口，agent_id = role_code。
# 与 prompts/roles/<role>.txt 一一对应，工具集由 permissions fail-closed 驱动。

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

    Loads prompt text from the ``prompts/`` directory, then interpolates
    the agent's name and description via ``str.format()``.

    ``visible_tool_names``：当前用户可见的工具名集合（来自 get_tools_for_agent 的
    fail-closed 过滤）。能力清单由该集合反向驱动（app/agent/tools.py 的
    build_capability_lines），保证 AI 宣称的每项能力都有对应工具支撑——与
    TOOL_PERMISSIONS 强制同源，不再按角色权限点宣称无工具支撑的功能。
    传 None（未接权限的内部调用方）时保持全量能力，作为兜底以免误裁。

    ``role_codes``：当前账号的角色编码列表（如 CurrentUser.roles）。存在
    prompts/roles/<role>.txt 时，把角色专属段（身份/职责视角/话术/补充语境）
    叠加到共享提示词之后，实现各角色不同的对话模式。角色段只改对话人格，
    不改变工具可见性与共享工作规则。传 None 保持纯共享提示词（向后兼容）。
    """
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["employee"])
    template = _load_all()
    capabilities = _render_capabilities(visible_tool_names)
    base = template.format(
        name=agent_info["name"],
        description=agent_info["description"],
        capabilities=capabilities,
    )
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
