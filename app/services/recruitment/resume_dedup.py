"""根据简历解析结果判断是否与库内候选人为同一人。"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.recruitment import Candidate, CandidateWorkExperience

UNKNOWN = "未知"


def normalize_person_name(value: Any) -> str:
    text = _clean_text(value)
    if not text or text == UNKNOWN:
        return ""
    return re.sub(r"\s+", "", text)


def normalize_education(value: Any) -> str:
    text = _clean_text(value)
    if not text or text == UNKNOWN:
        return ""
    text = re.sub(r"\s+", "", text)
    for suffix in ("学历", "学位"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
    aliases = {
        "本科": frozenset({"本科", "学士", "大学本科", "统招本科", "全日制本科"}),
        "硕士": frozenset({"硕士", "研究生", "硕士研究生", "全日制硕士"}),
        "博士": frozenset({"博士", "博士研究生"}),
        "大专": frozenset({"大专", "专科", "高职", "大学专科"}),
        "高中": frozenset({"高中", "中专", "职高", "中职"}),
    }
    for canonical, group in aliases.items():
        if text in group:
            return canonical
    return text


def normalize_org_name(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    for suffix in (
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "集团公司",
        "集团",
        "公司",
    ):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text.lower() if text.isascii() else text


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _work_pairs_from_parsed(work_history: Sequence[dict]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    for item in work_history or []:
        if not isinstance(item, dict):
            continue
        company = normalize_org_name(item.get("company"))
        role = normalize_org_name(item.get("role"))
        if company:
            pairs.add((company, role))
    return pairs


def _work_pairs_from_candidate(
    work_experiences: Iterable[CandidateWorkExperience],
) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    for item in work_experiences or []:
        company = normalize_org_name(item.company)
        role = normalize_org_name(item.role)
        if company:
            pairs.add((company, role))
    return pairs


def work_histories_overlap(parsed_work: Sequence[dict], candidate_work: Sequence[CandidateWorkExperience]) -> bool:
    left = _work_pairs_from_parsed(parsed_work)
    right = _work_pairs_from_candidate(candidate_work)
    if not left and not right:
        return True
    if not left or not right:
        return False
    for left_company, left_role in left:
        for right_company, right_role in right:
            if left_company != right_company:
                continue
            if not left_role or not right_role or left_role == right_role:
                return True
    return False


def is_same_person(parsed: dict, candidate: Candidate) -> bool:
    """姓名、年龄、性别、学历、工作经历均一致时视为同一人。"""
    parsed_name = normalize_person_name(parsed.get("name"))
    candidate_name = normalize_person_name(candidate.name)
    if not parsed_name or not candidate_name:
        return False
    if parsed_name != candidate_name:
        return False

    parsed_age = parsed.get("age")
    if not isinstance(parsed_age, int) or parsed_age <= 0:
        return False
    if not isinstance(candidate.age, int) or candidate.age <= 0:
        return False
    if parsed_age != candidate.age:
        return False

    parsed_gender = parsed.get("gender")
    candidate_gender = candidate.gender
    if parsed_gender not in ("男", "女") or candidate_gender not in ("男", "女"):
        return False
    if parsed_gender != candidate_gender:
        return False

    parsed_education = normalize_education(parsed.get("education"))
    candidate_education = normalize_education(candidate.education)
    if not parsed_education or not candidate_education:
        return False
    if parsed_education != candidate_education:
        return False

    if not work_histories_overlap(parsed.get("workHistory") or [], candidate.work_experiences or []):
        return False

    return True


async def find_duplicate_candidate(db: AsyncSession, parsed: dict) -> Optional[Candidate]:
    normalized_name = normalize_person_name(parsed.get("name"))
    if not normalized_name:
        return None

    result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.work_experiences))
        .where(func.replace(func.coalesce(Candidate.name, ""), " ", "") == normalized_name)
    )
    candidates = result.scalars().all()
    for candidate in candidates:
        if is_same_person(parsed, candidate):
            return candidate
    return None
