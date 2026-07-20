"""校验 LLM 解析出的字段是否在简历原文中有依据，避免臆测。"""
from __future__ import annotations

import re
from typing import Any, Optional

UNKNOWN = "未知"

EDUCATION_LEVEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "大专": ("大专", "专科", "高职", "大学专科"),
    "本科": ("本科", "学士", "大学本科", "统招本科", "全日制本科", "bachelor"),
    "硕士": ("硕士", "研究生", "硕士研究生", "master"),
    "博士": ("博士", "博士研究生", "phd", "doctor"),
}

EDUCATION_LEVEL_ORDER = ("博士", "硕士", "本科", "大专", "其他")


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def _field_appears_in_text(value: Any, raw_text: str, *, min_len: int = 3) -> bool:
    text = _normalize_match_text(raw_text)
    field = _normalize_match_text(str(value or ""))
    if not field or len(field) < min_len:
        return False
    if field in text:
        return True
    if len(field) >= 6 and field[:6] in text:
        return True
    if len(field) >= 4 and field[:4] in text:
        return True
    return False


def _education_level_in_text(level: str, raw_text: str) -> bool:
    if not level or level in (UNKNOWN, "其他"):
        return False
    keywords = EDUCATION_LEVEL_KEYWORDS.get(level, (level,))
    lowered = raw_text.lower()
    for keyword in keywords:
        if keyword in raw_text or keyword.lower() in lowered:
            return True
    return False


def _map_degree_to_level(degree: str) -> Optional[str]:
    text = (degree or "").strip()
    if not text or text == UNKNOWN:
        return None
    normalized = _normalize_match_text(text)
    for level, keywords in EDUCATION_LEVEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text or keyword in normalized or keyword.lower() in text.lower():
                return level
    if "其他" in text:
        return "其他"
    return None


def _pick_highest_level(levels: list[str]) -> Optional[str]:
    picked: Optional[str] = None
    picked_rank = -1
    for level in levels:
        if level not in EDUCATION_LEVEL_ORDER:
            continue
        rank = EDUCATION_LEVEL_ORDER.index(level)
        if rank > picked_rank:
            picked = level
            picked_rank = rank
    return picked


def verify_education_history(parsed: dict, raw_text: str) -> list[dict]:
    """仅保留原文中能对应上的教育经历，并清理无依据的学位字段。"""
    verified: list[dict] = []
    for item in parsed.get("educationHistory") or []:
        if not isinstance(item, dict):
            continue
        school = item.get("school")
        degree = item.get("degree")
        major = item.get("major")
        period = item.get("period")

        evidence_fields = [school, degree, major, period]
        if not any(_field_appears_in_text(field, raw_text, min_len=2) for field in evidence_fields):
            continue

        cleaned = dict(item)
        if degree and not _field_appears_in_text(degree, raw_text, min_len=2):
            level = _map_degree_to_level(str(degree))
            if level and not _education_level_in_text(level, raw_text):
                cleaned["degree"] = UNKNOWN
        if major and not _field_appears_in_text(major, raw_text, min_len=2):
            cleaned["major"] = UNKNOWN

        verified.append(cleaned)
    return verified


def resolve_verified_education(parsed: dict, raw_text: str) -> str:
    """根据原文校验最高学历，无依据则返回「未知」。"""
    verified_history = verify_education_history(parsed, raw_text)
    parsed["educationHistory"] = verified_history

    derived_levels: list[str] = []
    for item in verified_history:
        level = _map_degree_to_level(str(item.get("degree") or ""))
        if level:
            derived_levels.append(level)

    derived = _pick_highest_level(derived_levels)
    if derived:
        return derived

    education = str(parsed.get("education") or "").strip()
    if not education or education == UNKNOWN:
        return UNKNOWN
    if education == "其他":
        return "其他" if _education_level_in_text("其他", raw_text) else UNKNOWN
    if _education_level_in_text(education, raw_text):
        return education
    return UNKNOWN
