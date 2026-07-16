from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.models.candidate import Position, Candidate
from app.models.user import User
from app.routers.auth import get_current_user

router = APIRouter(tags=["岗位"])


class CreatePositionRequest(BaseModel):
    name: str


@router.get("/positions")
async def list_positions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Position).order_by(Position.created_at))
    positions = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [{"id": str(p.id), "name": p.name} for p in positions],
    }


@router.post("/positions")
async def create_position(
    req: CreatePositionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Check uniqueness
    existing = await db.execute(select(Position).where(Position.name == req.name))
    if existing.scalar_one_or_none():
        return {"code": 409, "message": "岗位名已存在", "data": None}

    position = Position(name=req.name)
    db.add(position)
    await db.flush()
    await db.refresh(position)
    return {
        "code": 0,
        "message": "ok",
        "data": {"id": str(position.id), "name": position.name},
    }


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

    # Check if any candidates reference this position
    candidate_result = await db.execute(
        select(Candidate).where(Candidate.position_id == position_id).limit(1)
    )
    if candidate_result.scalar_one_or_none():
        return {"code": 409, "message": "该岗位下有关联候选人，无法删除", "data": None}

    await db.delete(position)
    return {"code": 0, "message": "ok", "data": None}
