"""从教育经历解析学校层次标签（985/211/一本/二本/民办/专升本等）。"""
from __future__ import annotations

from typing import Optional

from app.services.recruitment.school_tier import (
    _211_ONLY_SCHOOLS,
    _985_SCHOOLS,
    _match_school,
    _quick_name_tier,
    classify_school_tier,
)
from app.services.resume_scoring.education_background.agent import (
    _pick_highest_education,
    classify_degree_level,
)

DISPLAY_TIER_MAP = {
    "清北": "985",
    "985": "985",
    "211": "211",
    "一本": "一本",
    "二本": "二本",
    "民办本": "民办",
    "专科": "专科",
}

SCHOOL_TIER_TAGS = frozenset(DISPLAY_TIER_MAP.values()) | {"专升本"}


def _education_blob(edu: dict) -> str:
    return " ".join(str(edu.get(key) or "") for key in ("school", "degree", "major", "period"))


def _has_zhuan_sheng_ben_marker(history: list[dict]) -> bool:
    has_zhuanke = False
    has_benke = False
    for item in history:
        if not isinstance(item, dict):
            continue
        blob = _education_blob(item)
        if "专升本" in blob:
            return True
        degree_level = classify_degree_level(str(item.get("degree") or ""))
        if degree_level == "专科":
            has_zhuanke = True
        if degree_level == "本科":
            has_benke = True
    return has_zhuanke and has_benke


def _sync_classify_school_tier(school: str) -> Optional[str]:
    name = (school or "").strip()
    if not name or name == "未知":
        return None

    quick = _quick_name_tier(name)
    if quick:
        return DISPLAY_TIER_MAP.get(quick, quick)
    if _match_school(name, _985_SCHOOLS):
        return "985"
    if _match_school(name, _211_ONLY_SCHOOLS):
        return "211"
    if any(token in name for token in ("民办", "独立学院", "科技学院")) and "学院" in name:
        return "民办"
    return None


def resolve_school_tier_tag_sync(parsed: dict) -> Optional[str]:
    """不调用 LLM，仅用规则判断学校层次标签。"""
    history = [item for item in (parsed.get("educationHistory") or []) if isinstance(item, dict)]
    edu = _pick_highest_education(parsed)
    if not edu:
        return None

    degree_level = classify_degree_level(str(edu.get("degree") or ""))

    if degree_level in ("硕士", "博士"):
        return _sync_classify_school_tier(str(edu.get("school") or ""))

    if degree_level == "本科":
        if _has_zhuan_sheng_ben_marker(history):
            return "专升本"
        return _sync_classify_school_tier(str(edu.get("school") or ""))

    if degree_level == "专科":
        return "专科"

    return None


async def resolve_school_tier_tag(parsed: dict) -> Optional[str]:
    """优先规则，其次 LLM 学校层次；硕士/博士仅参考最高学历院校。"""
    sync_tag = resolve_school_tier_tag_sync(parsed)
    if sync_tag:
        return sync_tag

    edu = _pick_highest_education(parsed)
    if not edu:
        return None

    degree_level = classify_degree_level(str(edu.get("degree") or ""))
    if degree_level == "本科" and _has_zhuan_sheng_ben_marker(
        [item for item in (parsed.get("educationHistory") or []) if isinstance(item, dict)]
    ):
        return "专升本"

    school = str(edu.get("school") or "").strip()
    if not school or school == "未知":
        return None

    tier = await classify_school_tier(school)
    return DISPLAY_TIER_MAP.get(tier)


def inject_school_tier_into_keywords(parsed: dict, tier_tag: str) -> None:
    analysis = parsed.setdefault("analysis", {})
    keywords = [str(item).strip() for item in (analysis.get("keywords") or []) if str(item).strip()]
    keywords = [item for item in keywords if item != tier_tag]
    keywords.insert(0, tier_tag)
    analysis["keywords"] = keywords[:5]
