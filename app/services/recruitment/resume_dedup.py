"""根据简历解析结果判断是否与库内候选人为同一人（仅按姓名判断）。"""
from __future__ import annotations

import re
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.recruitment import Candidate

UNKNOWN = "未知"


def normalize_person_name(value: Any) -> str:
    text = _clean_text(value)
    if not text or text == UNKNOWN:
        return ""
    return re.sub(r"\s+", "", text)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def is_same_person(parsed: dict, candidate: Candidate) -> bool:
    """仅按姓名判断是否同一人。"""
    parsed_name = normalize_person_name(parsed.get("name"))
    candidate_name = normalize_person_name(candidate.name)
    if not parsed_name or not candidate_name:
        return False
    return parsed_name == candidate_name


async def find_duplicate_candidate(db: AsyncSession, parsed: dict) -> Optional[Candidate]:
    """按姓名查重：在库内找到同名候选人即视为重复。"""
    normalized_name = normalize_person_name(parsed.get("name"))
    if not normalized_name:
        return None

    result = await db.execute(
        select(Candidate)
        .where(func.replace(func.coalesce(Candidate.name, ""), " ", "") == normalized_name)
    )
    candidates = result.scalars().all()
    for candidate in candidates:
        if is_same_person(parsed, candidate):
            return candidate
    return None
