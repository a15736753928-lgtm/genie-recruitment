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
        "description": "通过对话框完成招聘系统所有操作：简历筛选、面试出题、面试评定、试用期考核、绩效管理、知识库管理、系统设置",
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


def build_system_prompt(agent_id: str = "genie") -> str:
    """Build the full system prompt for a given agent.

    Loads prompt text from the ``prompts/`` directory, then interpolates
    the agent's name and description via ``str.format()``.
    """
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["genie"])
    template = _load_all()
    return template.format(
        name=agent_info["name"],
        description=agent_info["description"],
    )
