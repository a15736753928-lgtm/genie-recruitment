"""Resume dimension sub-agents.

5 个独立子 Agent，每个负责一个维度的打分，按固定顺序返回：

    教育背景 → 项目经验 → 工作经验 → 专业技能 → 实习经历

- 教育背景：规则化打分（学校层次 × 学历层次查表），不走 LLM。
- 其余 4 个：各自独立 LLM Agent，读自己文件夹下的 prompt.md 作为评判标准，
  只输出一个 0-100 的整数分数。
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from .base import DIMENSIONS
from .education_background.agent import score as score_education
from .project_experience.agent import score as score_project
from .work_experience.agent import score as score_work
from .professional_skill.agent import score as score_skill
from .internship_experience.agent import score as score_internship

__all__ = ["DIMENSIONS", "score_all"]


async def score_all(
    resume_text: str, parsed: dict, position_name: str = ""
) -> List[dict]:
    """并发跑 5 个维度子 Agent，按固定顺序返回 ``[{name, score}, ...]``。

    任何子 Agent 失败或返回 None 时，该维度补中性分 70，保证 5 个维度齐全。
    """
    edu, proj, work, skill, intern = await asyncio.gather(
        score_education(resume_text, parsed, position_name),
        score_project(resume_text, parsed, position_name),
        score_work(resume_text, parsed, position_name),
        score_skill(resume_text, parsed, position_name),
        score_internship(resume_text, parsed, position_name),
    )

    def _fill(v: Optional[int]) -> int:
        return v if isinstance(v, int) else 70

    return [
        {"name": "教育背景", "score": _fill(edu)},
        {"name": "项目经验", "score": _fill(proj)},
        {"name": "工作经验", "score": _fill(work)},
        {"name": "专业技能", "score": _fill(skill)},
        {"name": "实习经历", "score": _fill(intern)},
    ]
