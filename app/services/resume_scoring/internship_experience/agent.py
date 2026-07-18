"""实习经历维度子 Agent —— LLM 打分。"""
from __future__ import annotations

from typing import Optional

from app.services.resume_scoring.base import _llm_score, _load_prompt

_RUBRIC = _load_prompt("internship_experience")


async def score(text: str, parsed: dict, position_name: str = "") -> Optional[int]:
    return await _llm_score(_RUBRIC, text, position_name)
