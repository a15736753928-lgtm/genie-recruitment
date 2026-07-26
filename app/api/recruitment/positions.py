from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field
from app.database import get_db
from app.models.recruitment import Position, Candidate, PositionQuestion
from app.utils.clock import iso_utc
router = APIRouter(tags=["岗位"])


class CreatePositionRequest(BaseModel):
    model_config = {"populate_by_name": True}

    name: str
    department: Optional[str] = None
    jd_responsibilities: Optional[str] = Field(None, alias="jdResponsibilities")
    jd_requirements: Optional[str] = Field(None, alias="jdRequirements")
    jd_preferred: Optional[str] = Field(None, alias="jdPreferred")
    jd_tech_stack: Optional[str] = Field(None, alias="jdTechStack")
    education_requirement: Optional[str] = Field(None, alias="educationRequirement")
    experience_requirement: Optional[str] = Field(None, alias="experienceRequirement")
    age_requirement: Optional[str] = Field(None, alias="ageRequirement")
    salary_range: Optional[str] = Field(None, alias="salaryRange")


class UpdatePositionRequest(BaseModel):
    model_config = {"populate_by_name": True}

    name: Optional[str] = None
    department: Optional[str] = None
    jd_responsibilities: Optional[str] = Field(None, alias="jdResponsibilities")
    jd_requirements: Optional[str] = Field(None, alias="jdRequirements")
    jd_preferred: Optional[str] = Field(None, alias="jdPreferred")
    jd_tech_stack: Optional[str] = Field(None, alias="jdTechStack")
    education_requirement: Optional[str] = Field(None, alias="educationRequirement")
    experience_requirement: Optional[str] = Field(None, alias="experienceRequirement")
    age_requirement: Optional[str] = Field(None, alias="ageRequirement")
    salary_range: Optional[str] = Field(None, alias="salaryRange")
    screening_criteria: Optional[dict] = Field(None, alias="screeningCriteria")
    interview_criteria_r1: Optional[dict] = Field(None, alias="interviewCriteriaR1")
    interview_criteria_r2: Optional[dict] = Field(None, alias="interviewCriteriaR2")
    week1_project_requirement: Optional[dict] = Field(None, alias="week1ProjectRequirement")
    weeks_2_4_plan: Optional[dict] = Field(None, alias="weeks24Plan")
    later_week_scoring: Optional[dict] = Field(None, alias="laterWeekScoring")
    conversion_criteria: Optional[dict] = Field(None, alias="conversionCriteria")


def serialize_position(p: Position) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "department": p.department,
        "jdContent": p.jd_content,
        "jdResponsibilities": p.jd_responsibilities,
        "jdRequirements": p.jd_requirements,
        "jdPreferred": p.jd_preferred,
        "jdTechStack": p.jd_tech_stack,
        "educationRequirement": p.education_requirement,
        "experienceRequirement": p.experience_requirement,
        "ageRequirement": p.age_requirement,
        "salaryRange": p.salary_range,
        "screeningCriteria": p.screening_criteria,
        "interviewCriteriaR1": p.interview_criteria_r1,
        "interviewCriteriaR2": p.interview_criteria_r2,
        "week1ProjectRequirement": p.week1_project_requirement,
        "weeks24Plan": p.weeks_2_4_plan,
        "laterWeekScoring": p.later_week_scoring,
        "conversionCriteria": p.conversion_criteria,
        "createdAt": iso_utc(p.created_at),
        "updatedAt": iso_utc(p.updated_at),
    }


@router.get("/positions")
async def list_positions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Position).order_by(Position.created_at))
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
):
    existing = await db.execute(select(Position).where(Position.name == req.name))
    if existing.scalar_one_or_none():
        return {"code": 409, "message": "岗位名已存在", "data": None}

    position = Position(
        name=req.name,
        department=req.department,
        jd_responsibilities=req.jd_responsibilities,
        jd_requirements=req.jd_requirements,
        jd_preferred=req.jd_preferred,
        jd_tech_stack=req.jd_tech_stack,
        education_requirement=req.education_requirement,
        experience_requirement=req.experience_requirement,
        age_requirement=req.age_requirement,
        salary_range=req.salary_range,
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
):
    from sqlalchemy import text

    result = await db.execute(select(Position).where(Position.id == position_id))
    position = result.scalar_one_or_none()
    if not position:
        return {"code": 404, "message": "岗位不存在", "data": None}

    candidate_result = await db.execute(
        select(Candidate).where(Candidate.position_id == position_id).limit(1)
    )
    if candidate_result.scalar_one_or_none():
        return {"code": 409, "message": "该岗位下有关联候选人，请先删除或转移相关简历后再删除岗位", "data": None}

    # employees 外键是 NO ACTION，先解除引用再删岗位
    await db.execute(
        text("UPDATE employees SET position_id = NULL WHERE position_id = :pid"),
        {"pid": position_id},
    )

    await db.delete(position)
    await db.flush()
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
