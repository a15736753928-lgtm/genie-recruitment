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


# ── Agent Configurations ──────────────────────────────────

AGENT_CONFIGS = {
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


def build_system_prompt(agent_id: str = "genie", visible_tool_names: set[str] | None = None) -> str:
    """Build the full system prompt for a given agent.

    Loads prompt text from the ``prompts/`` directory, then interpolates
    the agent's name and description via ``str.format()``.

    ``visible_tool_names``：当前用户可见的工具名集合（来自 get_tools_for_agent 的
    fail-closed 过滤）。能力清单由该集合反向驱动（app/agent/tools.py 的
    build_capability_lines），保证 AI 宣称的每项能力都有对应工具支撑——与
    TOOL_PERMISSIONS 强制同源，不再按角色权限点宣称无工具支撑的功能。
    传 None（未接权限的内部调用方）时保持全量能力，作为兜底以免误裁。
    """
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["genie"])
    template = _load_all()
    capabilities = _render_capabilities(visible_tool_names)
    return template.format(
        name=agent_info["name"],
        description=agent_info["description"],
        capabilities=capabilities,
    )


def _render_capabilities(visible_tool_names: set[str] | None) -> str:
    """渲染 AI 可见能力清单（由可见工具集驱动，与 TOOL_PERMISSIONS 同源）。"""
    from app.agent.tools import build_capability_lines

    lines = build_capability_lines(visible_tool_names)
    if not lines:
        # 极端情况：用户无任何可见工具 → 仍给一句基础定位，避免空白段
        return "- 查询系统信息、处理你能访问的招聘事项；如需更多权限请联系管理员"
    return "\n".join(lines)
