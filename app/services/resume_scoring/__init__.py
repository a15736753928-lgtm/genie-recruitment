"""Resume dimension sub-agents.

5 个维度，按固定顺序返回：教育背景 → 项目经验 → 工作经验 → 专业技能 → 实习经历

- 教育背景：规则化打分（学校层次 × 学历层次查表），不走 LLM。
- 其余 4 个维度：合并为 1 次 LLM 调用，一次返回 4 个分数。
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from .base import DIMENSIONS, _llm_score_batched
from .education_background.agent import score as score_education

__all__ = ["DIMENSIONS", "score_all"]


async def score_all(
    resume_text: str, parsed: dict, position_name: str = ""
) -> List[dict]:
    """评分入口：教育背景（规则）+ 四个维度（1 次 LLM）并行执行。

    任何维度失败或返回 None 时，该维度补中性分 70，保证 5 个维度齐全。
    """
    edu_task = asyncio.ensure_future(
        score_education(resume_text, parsed, position_name)
    )
    batched_task = asyncio.ensure_future(
        _llm_score_batched(resume_text, position_name)
    )

    edu = await edu_task
    batched = await batched_task

    def _fill(v: Optional[int]) -> int:
        return v if isinstance(v, int) else 70

    return [
        {"name": "教育背景", "score": _fill(edu)},
        {"name": "项目经验", "score": _fill(batched.get("项目经验"))},
        {"name": "工作经验", "score": _fill(batched.get("工作经验"))},
        {"name": "专业技能", "score": _fill(batched.get("专业技能"))},
        {"name": "实习经历", "score": _fill(batched.get("实习经历"))},
    ]
