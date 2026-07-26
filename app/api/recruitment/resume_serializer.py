"""Candidate serialization — convert ORM objects to frontend-compatible dicts.

Extracted from ``resumes.py`` so the route layer stays focused on HTTP concerns.
"""

from __future__ import annotations

import os
from typing import Optional

from app.models.recruitment import Candidate
from app.utils.clock import iso_utc

UNKNOWN = "未知"
IN_SCHOOL = "在校中"
DEFAULT_ETHNICITY = "汉族"
NO_WORK_EXPERIENCE_VALUES = {
    UNKNOWN, "应届", "无", "暂无", "无工作经验", "0", "0年",
}


def display_text(value, default: str = UNKNOWN) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    return str(value)


def normalize_ethnicity(value) -> str:
    text = display_text(value, default="")
    if not text or text == UNKNOWN:
        return DEFAULT_ETHNICITY
    return text


def normalize_experience(value, *, has_work_history: bool = False) -> str:
    text = display_text(value, default="")
    if has_work_history:
        return text if text else UNKNOWN
    if not text or text in NO_WORK_EXPERIENCE_VALUES:
        return IN_SCHOOL
    return text


def format_experience_display(experience, work_experiences) -> str:
    text = (experience or "").strip()
    if work_experiences:
        return display_text(text)
    if not text or text in NO_WORK_EXPERIENCE_VALUES:
        return IN_SCHOOL
    return display_text(text)


def is_candidate_parsed(candidate: Candidate) -> bool:
    return bool(
        candidate.phone
        or candidate.email
        or candidate.education
        or candidate.experience
        or candidate.score
    )


# 与 scoring.py compute_grade() 使用同一组默认阈值。candidate.grade 并非独立列，
# 而是按 candidate.score 派生——这里是同步函数（大量调用点无 db/await），
# 拿不到 system_settings 里可能被管理员改过的阈值，只能退回默认值展示。
_DEFAULT_GRADE_THRESHOLDS = {"A": 85, "B": 70, "C": 60}


def compute_grade_sync(score: int) -> str:
    if score >= _DEFAULT_GRADE_THRESHOLDS["A"]:
        return "A"
    if score >= _DEFAULT_GRADE_THRESHOLDS["B"]:
        return "B"
    if score >= _DEFAULT_GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def serialize_candidate(c: Candidate) -> dict:
    """Convert ORM Candidate + relationships to frontend-expected dict."""
    from app.services.recruitment.education_tier_tag import (
        resolve_school_tier_tag_sync,
        SCHOOL_TIER_TAGS,
    )

    resume_path = c.resume_file or ""
    resume_ext = os.path.splitext(resume_path)[1].lower() if resume_path else ""
    gender = c.gender if c.gender in ("男", "女") else UNKNOWN
    data = {
        "id": str(c.id),
        "name": display_text(c.name, UNKNOWN),
        "gender": gender,
        "age": c.age if c.age and c.age > 0 else None,
        "education": display_text(c.education),
        "experience": format_experience_display(c.experience, c.work_experiences),
        "position": c.position.name if c.position else UNKNOWN,
        "positionId": str(c.position_id) if c.position_id else "",
        "score": c.score or 0,
        "grade": compute_grade_sync(c.score or 0),
        "status": c.status or "new",
        "uploadTime": iso_utc(c.upload_time),
        "phone": display_text(c.phone),
        "email": display_text(c.email),
        "ethnicity": normalize_ethnicity(c.ethnicity),
        "nativePlace": display_text(c.native_place),
        "skills": [s.skill for s in (c.skills or [])],
        "workHistory": [
            {"company": w.company, "role": w.role, "period": w.period, "description": w.description or ""}
            for w in (c.work_experiences or [])
        ],
        "educationHistory": [
            {"school": e.school, "degree": e.degree, "major": e.major, "period": e.period}
            for e in (c.educations or [])
        ],
        "projectHistory": [
            {"name": p.name, "role": p.role, "period": p.period, "description": p.description or ""}
            for p in (c.project_experiences or [])
        ],
        "resumeFileUrl": f"/resumes/{c.id}/file" if resume_path else "",
        "resumeFileName": os.path.basename(resume_path) if resume_path else "",
        "resumeFileType": resume_ext.lstrip("."),
        "parseStatus": "parsed" if is_candidate_parsed(c) else "pending",
        "aiAnalysis": None,
        "schoolTierLabel": None,
    }
    tier_from_keywords = None
    if c.ai_analysis and c.ai_analysis.keywords:
        for keyword in c.ai_analysis.keywords:
            if keyword in SCHOOL_TIER_TAGS:
                tier_from_keywords = keyword
                break
    data["schoolTierLabel"] = tier_from_keywords or resolve_school_tier_tag_sync({
        "education": data["education"],
        "educationHistory": data["educationHistory"],
    })
    if c.ai_analysis:
        a = c.ai_analysis
        data["aiAnalysis"] = {
            "overallScore": a.overall_score or 0,
            "dimensions": a.dimensions or [],
            "keywords": a.keywords or [],
            "summary": a.summary or "",
            "positionMatch": a.position_match or "",
            "experienceInsight": a.experience_insight or "",
            "highlights": a.highlights or [],
            "risks": a.risks or [],
            "recommendation": a.recommendation or "",
        }
    return data
