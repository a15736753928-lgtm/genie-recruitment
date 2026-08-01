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


async def find_duplicate_candidate(
    db: AsyncSession,
    parsed: dict,
    content_hash: Optional[str] = None,
) -> Optional[Candidate]:
    """按文件内容 SHA256 判重：只有字节完全相同的简历才视为重复。

    不再按姓名判重——同名不同内容的简历（同名不同人 / 同一人多版）会被
    误判跳过，漏掉候选人。历史数据 resume_file_hash 为空时不判重。
    """
    if not content_hash:
        return None

    result = await db.execute(
        select(Candidate)
        .where(Candidate.resume_file_hash == content_hash)
        .limit(1)
    )
    return result.scalar_one_or_none()
