"""教育背景维度子 Agent —— 规则化打分（不走 LLM）。

按"最高学历所在学校的层次 × 学历层次"查表得到 0-100 分。
表内已固化所有规则：博士=100、985/清北=100、硕士≥85、本科≥60。
学校层次由 app.services.school_tier.classify_school_tier 判定。
"""
from __future__ import annotations

from typing import Optional

# 学校层次（行）× 学历层次（列），0-100。
EDU_SCORE_TABLE = {
    "清北":   {"本科": 100, "硕士": 100, "博士": 100},
    "985":    {"本科": 100, "硕士": 100, "博士": 100},
    "211":    {"本科": 95,  "硕士": 95,  "博士": 100},
    "一本":   {"本科": 90,  "硕士": 90,  "博士": 100},
    "二本":   {"本科": 80,  "硕士": 90,  "博士": 100},
    "民办本": {"本科": 70,  "硕士": 85,  "博士": 100},
    "专科":   {"专科": 60},
}


def classify_degree_level(degree: str) -> str:
    """从学位字符串判断学历层次：博士/硕士/本科/专科。"""
    d = (degree or "").strip()
    if "博士" in d:
        return "博士"
    if "硕士" in d:
        return "硕士"
    if "专科" in d or "大专" in d or "高职" in d:
        return "专科"
    if "本科" in d or "学士" in d:
        return "本科"
    return "本科"  # 大学条目默认按本科


def _pick_highest_education(parsed: dict) -> Optional[dict]:
    """返回 educationHistory 中学历层次最高的那条。"""
    rank = {"博士": 4, "硕士": 3, "本科": 2, "专科": 1}
    best, best_rank = None, 0
    for edu in (parsed.get("educationHistory") or []):
        if not isinstance(edu, dict):
            continue
        lvl = classify_degree_level(str(edu.get("degree") or ""))
        r = rank.get(lvl, 0)
        if r > best_rank:
            best_rank, best = r, edu
    return best


async def score(text: str, parsed: dict, position_name: str = "") -> Optional[int]:
    """按"最高学历的学校层次 × 学历层次"查表得到教育背景分数。"""
    edu = _pick_highest_education(parsed)
    if not edu:
        return None
    school = str(edu.get("school") or "").strip()
    if not school or school in ("未知", "null", "None"):
        return None
    degree_level = classify_degree_level(str(edu.get("degree") or ""))
    from app.services.school_tier import classify_school_tier
    tier = await classify_school_tier(school)
    row = EDU_SCORE_TABLE.get(tier)
    if not row:
        return None
    return row.get(degree_level)
