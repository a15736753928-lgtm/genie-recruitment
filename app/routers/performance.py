from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.database import get_db
from app.models.performance import PerformanceRecord, PerformanceQuarter
from app.models.probation import Employee
from app.config import get_settings

router = APIRouter(tags=["绩效"])
settings = get_settings()


# ── Serializer ──────────────────────────────────────────

def serialize_record(r: PerformanceRecord) -> dict:
    return {
        "id": str(r.id),
        "rank": r.rank or 0,
        "name": r.employee.name if r.employee else "",
        "department": r.employee.department if r.employee else "",
        "tasksCompleted": r.tasks_completed or 0,
        "quality": r.quality or 0,
        "speed": r.speed or 0,
        "compliance": r.compliance or 0,
        "totalScore": r.total_score or 0,
        "grade": r.grade or "B",
        "bonus": float(r.bonus) if r.bonus else None,
    }


# ── Endpoints ───────────────────────────────────────────

@router.get("/performance/stats")
async def get_performance_stats(quarter: str = Query(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PerformanceRecord).where(PerformanceRecord.quarter == quarter)
    )
    records = result.scalars().all()

    participants = len(records)
    if participants == 0:
        return {
            "code": 0, "message": "ok",
            "data": {"participants": 0, "avgScore": 0, "excellentCount": 0, "needsImprovement": 0},
        }

    avg_score = round(sum(r.total_score or 0 for r in records) / participants, 1)
    excellent_count = sum(1 for r in records if (r.grade or "") in ("S", "A"))
    needs_improvement = sum(1 for r in records if (r.grade or "") == "C")

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "participants": participants,
            "avgScore": avg_score,
            "excellentCount": excellent_count,
            "needsImprovement": needs_improvement,
        },
    }


@router.get("/performance")
async def list_performance(
    quarter: str = Query(...),
    page: int = Query(1),
    pageSize: int = Query(10),
    db: AsyncSession = Depends(get_db),
):
    query = select(PerformanceRecord).options(
        selectinload(PerformanceRecord.employee)
    ).where(PerformanceRecord.quarter == quarter)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(PerformanceRecord.rank.asc().nullslast())
    query = query.offset((page - 1) * pageSize).limit(pageSize)

    result = await db.execute(query)
    records = result.unique().scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "list": [serialize_record(r) for r in records],
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.get("/performance/departments")
async def get_department_performance(quarter: str = Query(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PerformanceRecord).options(
            selectinload(PerformanceRecord.employee)
        ).where(PerformanceRecord.quarter == quarter)
    )
    records = result.scalars().all()

    dept_scores = {}
    dept_counts = {}
    for r in records:
        dept = r.employee.department if r.employee else "未知"
        dept_scores[dept] = dept_scores.get(dept, 0) + (r.total_score or 0)
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    data = [
        {"department": dept, "score": round(dept_scores[dept] / dept_counts[dept])}
        for dept in dept_scores
    ]
    data.sort(key=lambda x: x["score"], reverse=True)

    return {"code": 0, "message": "ok", "data": data}


@router.get("/performance/grades")
async def get_grade_distribution(quarter: str = Query(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PerformanceRecord).where(PerformanceRecord.quarter == quarter)
    )
    records = result.scalars().all()
    total = len(records)

    grade_counts = {}
    for r in records:
        g = r.grade or "B"
        grade_counts[g] = grade_counts.get(g, 0) + 1

    grade_order = ["S", "A", "B+", "B", "C"]
    data = [
        {
            "grade": g,
            "count": grade_counts.get(g, 0),
            "percentage": round(grade_counts.get(g, 0) / total * 100) if total > 0 else 0,
        }
        for g in grade_order
    ]

    return {"code": 0, "message": "ok", "data": data}


@router.get("/performance/bonus")
async def get_bonus_info(quarter: str = Query(...), db: AsyncSession = Depends(get_db)):
    pq_result = await db.execute(
        select(PerformanceQuarter).where(PerformanceQuarter.quarter == quarter)
    )
    pq = pq_result.scalar_one_or_none()

    if not pq:
        return {
            "code": 0, "message": "ok",
            "data": {"totalPool": 0, "distributed": 0, "pending": 0},
        }

    total_pool = float(pq.bonus_pool or 0)
    distributed = float(pq.distributed or 0)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "totalPool": total_pool,
            "distributed": distributed,
            "pending": total_pool - distributed,
        },
    }


@router.get("/performance/trends")
async def get_quarter_trends(db: AsyncSession = Depends(get_db)):
    # Get last 4 quarters
    result = await db.execute(
        select(PerformanceQuarter).order_by(PerformanceQuarter.quarter.desc()).limit(4)
    )
    quarters = result.scalars().all()

    data = []
    for pq in reversed(quarters):
        # Average score for this quarter
        avg_result = await db.execute(
            select(func.avg(PerformanceRecord.total_score))
            .where(PerformanceRecord.quarter == pq.quarter)
        )
        avg = avg_result.scalar()

        data.append({
            "quarter": pq.quarter,
            "score": round(float(avg), 1) if avg else 0,
            "isCurrent": False,
        })

    if data:
        data[-1]["isCurrent"] = True

    return {"code": 0, "message": "ok", "data": data}


@router.post("/performance/initiate")
async def initiate_appraisal(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    quarter = body.get("quarter")
    employee_ids = body.get("employeeIds", [])

    # Create or update quarter
    pq_result = await db.execute(
        select(PerformanceQuarter).where(PerformanceQuarter.quarter == quarter)
    )
    pq = pq_result.scalar_one_or_none()
    if not pq:
        pq = PerformanceQuarter(quarter=quarter, status="active", bonus_pool=500000)
        db.add(pq)

    # Create records for employees
    for eid in employee_ids:
        existing = await db.execute(
            select(PerformanceRecord).where(
                PerformanceRecord.employee_id == eid,
                PerformanceRecord.quarter == quarter,
            )
        )
        if not existing.scalar_one_or_none():
            emp_result = await db.execute(select(Employee).where(Employee.id == eid))
            emp = emp_result.scalar_one_or_none()
            if emp:
                db.add(PerformanceRecord(
                    employee_id=eid,
                    quarter=quarter,
                    tasks_completed=0,
                    quality=0,
                    speed=0,
                    compliance=0,
                    total_score=0,
                    grade="B",
                ))

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.put("/performance/{employee_id}/bonus")
async def update_bonus(
    employee_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    bonus = body.get("bonus", 0)
    # Find most recent performance record
    result = await db.execute(
        select(PerformanceRecord)
        .where(PerformanceRecord.employee_id == employee_id)
        .order_by(PerformanceRecord.quarter.desc())
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if not record:
        return {"code": 404, "message": "绩效记录不存在", "data": None}

    record.bonus = bonus
    await db.flush()
    return {"code": 0, "message": "ok", "data": None}
