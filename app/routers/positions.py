from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.database import get_db
from app.models.candidate import Position, Candidate, PositionQuestion
from app.models.user import User
from app.routers.auth import get_current_user

router = APIRouter(tags=["岗位"])


class CreatePositionRequest(BaseModel):
    name: str
    chapter_number: Optional[int] = None
    department: Optional[str] = None
    jd_responsibilities: Optional[str] = None
    jd_requirements: Optional[str] = None
    jd_preferred: Optional[str] = None
    jd_tech_stack: Optional[str] = None


class UpdatePositionRequest(BaseModel):
    name: Optional[str] = None
    chapter_number: Optional[int] = None
    department: Optional[str] = None
    jd_responsibilities: Optional[str] = None
    jd_requirements: Optional[str] = None
    jd_preferred: Optional[str] = None
    jd_tech_stack: Optional[str] = None
    screening_criteria: Optional[dict] = None
    interview_criteria_r1: Optional[dict] = None
    interview_criteria_r2: Optional[dict] = None
    week1_project_requirement: Optional[dict] = None
    weeks_2_4_plan: Optional[dict] = None
    later_week_scoring: Optional[dict] = None
    conversion_criteria: Optional[dict] = None


def serialize_position(p: Position) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "chapterNumber": p.chapter_number,
        "department": p.department,
        "jdContent": p.jd_content,
        "jdResponsibilities": p.jd_responsibilities,
        "jdRequirements": p.jd_requirements,
        "jdPreferred": p.jd_preferred,
        "jdTechStack": p.jd_tech_stack,
        "screeningCriteria": p.screening_criteria,
        "interviewCriteriaR1": p.interview_criteria_r1,
        "interviewCriteriaR2": p.interview_criteria_r2,
        "week1ProjectRequirement": p.week1_project_requirement,
        "weeks24Plan": p.weeks_2_4_plan,
        "laterWeekScoring": p.later_week_scoring,
        "conversionCriteria": p.conversion_criteria,
        "createdAt": p.created_at.isoformat() if p.created_at else "",
        "updatedAt": p.updated_at.isoformat() if p.updated_at else "",
    }


@router.get("/positions")
async def list_positions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Position).order_by(Position.chapter_number, Position.created_at))
    positions = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [serialize_position(p) for p in positions],
    }


@router.get("/positions/{position_id}")
async def get_position(position_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Position).where(Position.id == position_id))
    position = result.scalar_one_or_none()
    if not position:
        return {"code": 404, "message": "岗位不存在", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_position(position)}


@router.post("/positions")
async def create_position(
    req: CreatePositionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = await db.execute(select(Position).where(Position.name == req.name))
    if existing.scalar_one_or_none():
        return {"code": 409, "message": "岗位名已存在", "data": None}

    position = Position(
        name=req.name,
        chapter_number=req.chapter_number,
        department=req.department,
        jd_responsibilities=req.jd_responsibilities,
        jd_requirements=req.jd_requirements,
        jd_preferred=req.jd_preferred,
        jd_tech_stack=req.jd_tech_stack,
    )
    db.add(position)
    await db.flush()
    await db.refresh(position)
    return {"code": 0, "message": "ok", "data": serialize_position(position)}


@router.put("/positions/{position_id}")
async def update_position(
    position_id: str,
    req: UpdatePositionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Position).where(Position.id == position_id))
    position = result.scalar_one_or_none()
    if not position:
        return {"code": 404, "message": "岗位不存在", "data": None}

    update_data = req.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(position, field, value)

    await db.flush()
    await db.refresh(position)
    return {"code": 0, "message": "ok", "data": serialize_position(position)}


@router.delete("/positions/{position_id}")
async def delete_position(
    position_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Position).where(Position.id == position_id))
    position = result.scalar_one_or_none()
    if not position:
        return {"code": 404, "message": "岗位不存在", "data": None}

    candidate_result = await db.execute(
        select(Candidate).where(Candidate.position_id == position_id).limit(1)
    )
    if candidate_result.scalar_one_or_none():
        return {"code": 409, "message": "该岗位下有关联候选人，无法删除", "data": None}

    await db.delete(position)
    return {"code": 0, "message": "ok", "data": None}


# ── Position question bank ──

@router.get("/positions/{position_id}/questions")
async def get_position_questions(
    position_id: str,
    round: str = "first",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PositionQuestion)
        .where(PositionQuestion.position_id == position_id, PositionQuestion.round == round)
        .order_by(PositionQuestion.index_num)
    )
    questions = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(q.id),
                "positionId": str(q.position_id),
                "round": q.round,
                "index": q.index_num,
                "content": q.content,
                "category": q.category,
                "difficulty": q.difficulty,
            }
            for q in questions
        ],
    }


@router.put("/positions/{position_id}/questions")
async def save_position_questions(
    position_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    round = body.get("round", "first")
    questions = body.get("questions", [])

    # Delete existing questions for this round
    existing = await db.execute(
        select(PositionQuestion).where(
            PositionQuestion.position_id == position_id,
            PositionQuestion.round == round,
        )
    )
    for q in existing.scalars().all():
        await db.delete(q)

    # Add new questions
    for qd in questions:
        db.add(PositionQuestion(
            position_id=position_id,
            round=round,
            index_num=qd.get("index", 1),
            content=qd.get("content", ""),
            category=qd.get("category", ""),
            difficulty=qd.get("difficulty", "medium"),
        ))

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}
