"""项目经验维度子 Agent —— LLM 打分。

读取本目录下 prompt.md 作为评判标准，结合简历文本与目标岗位，
只输出一个 0-100 的整数分数。
"""
from __future__ import annotations

from typing import Optional

from app.services.resume_scoring.base import _llm_score, _load_prompt

_RUBRIC = _load_prompt("project_experience")


async def score(text: str, parsed: dict, position_name: str = "") -> Optional[int]:
    return await _llm_score(_RUBRIC, text, position_name)
