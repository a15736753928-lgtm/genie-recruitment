"""Shared helpers for resume dimension sub-agents.

Each sub-agent lives in its own folder under ``resume_agents/`` and exposes
``async def score(text, parsed, position_name) -> Optional[int]`` returning a
0-100 integer. The LLM-based agents share a single prompt template and a robust
integer-score extractor here.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 固定的 5 个维度及其展示顺序。
DIMENSIONS = ("教育背景", "项目经验", "工作经验", "专业技能", "实习经历")


def _load_prompt(folder: str) -> str:
    """Load the rubric prompt for a sub-agent from its ``prompt.md`` file."""
    path = Path(__file__).parent / folder / "prompt.md"
    return path.read_text(encoding="utf-8")


async def _llm_score(rubric: str, resume_text: str, position_name: str) -> Optional[int]:
    """Call the LLM with a rubric + resume and parse a single 0-100 integer."""
    if not resume_text or not resume_text.strip():
        return None
    position_line = f"\n目标应聘岗位：{position_name}" if position_name else ""
    prompt = (
        f"{rubric}\n"
        f"{position_line}\n\n"
        f"【简历文本】\n{resume_text[:8000]}\n\n"
        f"请只输出一个 0 到 100 的整数分数，不要任何其它文字、解释或标点。"
    )
    try:
        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=20,
        )
        content = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d{1,3}", content)
        if not m:
            logger.warning("dimension agent returned non-numeric: %r", content)
            return None
        score = max(0, min(100, int(m.group())))
        return score
    except Exception as exc:
        logger.warning("dimension agent LLM call failed: %s", exc)
        return None
