import json
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import OpenAI
from app.database import get_db
from app.models.probation import Employee, ProbationTask
from app.models.user import User
from app.routers.auth import get_current_user
from app.config import get_settings

router = APIRouter(tags=["试用期"])
settings = get_settings()

llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)


# ── Serializer ──────────────────────────────────────────

def serialize_employee(emp: Employee) -> dict:
    tasks = list(emp.tasks or [])
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.status == "completed")
    return {
        "id": str(emp.id),
        "name": emp.name,
        "gender": emp.gender,
        "age": emp.age,
        "department": emp.department,
        "joinDate": emp.join_date.isoformat() if emp.join_date else "",
        "probationEnd": emp.probation_end.isoformat() if emp.probation_end else "",
        "taskProgress": round(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        "totalTasks": total_tasks,
        "completedTasks": completed_tasks,
        "status": emp.status or "assessing",
        "aiScore": emp.ai_score,
        "aiResult": emp.ai_result,
        "tasks": [
            {"id": str(t.id), "title": t.title, "status": t.status}
            for t in tasks
        ],
    }


# ── Endpoints ───────────────────────────────────────────

@router.get("/probation/stats")
async def get_probation_stats(db: AsyncSession = Depends(get_db)):
    total_query = select(func.count()).select_from(Employee)
    assessing_query = select(func.count()).select_from(Employee).where(Employee.status == "assessing")
    passed_query = select(func.count()).select_from(Employee).where(Employee.status == "passed")
    failed_query = select(func.count()).select_from(Employee).where(Employee.status == "failed")

    total = (await db.execute(total_query)).scalar() or 0
    assessing = (await db.execute(assessing_query)).scalar() or 0
    passed = (await db.execute(passed_query)).scalar() or 0
    failed = (await db.execute(failed_query)).scalar() or 0

    return {
        "code": 0,
        "message": "ok",
        "data": {"total": total, "assessing": assessing, "passed": passed, "failed": failed},
    }


@router.get("/probation")
async def list_probation(
    department: str = Query("all"),
    status: str = Query("all"),
    page: int = Query(1),
    pageSize: int = Query(10),
    db: AsyncSession = Depends(get_db),
):
    query = select(Employee).options(selectinload(Employee.tasks))

    if department and department != "all":
        query = query.where(Employee.department == department)
    if status and status != "all":
        query = query.where(Employee.status == status)

    # Count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Paginate
    query = query.offset((page - 1) * pageSize).limit(pageSize).order_by(Employee.created_at.desc())
    result = await db.execute(query)
    employees = result.unique().scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "list": [serialize_employee(e) for e in employees],
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.post("/probation/tasks")
async def create_probation_task(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    employee_id = body.get("employeeId")
    task_title = body.get("taskTitle")

    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    task = ProbationTask(employee_id=employee_id, title=task_title)
    db.add(task)
    await db.flush()
    await db.refresh(emp)

    return {"code": 0, "message": "ok", "data": serialize_employee(emp)}


@router.post("/probation/{employee_id}/ai-evaluate")
async def ai_evaluate_probation(employee_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    # Build evaluation context
    tasks = [t.title for t in (emp.tasks or [])]
    completed = [t.title for t in (emp.tasks or []) if t.status == "completed"]

    prompt = f"""作为HR试用期评估专家，请对以下员工进行试用期表现评估。

员工：{emp.name}
部门：{emp.department or '未指定'}
入职日期：{emp.join_date}
试用期截止：{emp.probation_end}

所有任务：{json.dumps(tasks, ensure_ascii=False)}
已完成任务：{json.dumps(completed, ensure_ascii=False)}

请返回JSON：{{"score": 0-100的整数, "result": "通过"或"不通过"}}
只返回JSON。"""

    try:
        response = llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        ai_result = json.loads(content.strip())

        emp.ai_score = ai_result.get("score")
        emp.ai_result = ai_result.get("result")
        await db.flush()

        return {"code": 0, "message": "ok", "data": ai_result}
    except Exception as e:
        return {"code": 500, "message": f"AI评估失败: {str(e)}", "data": None}


@router.put("/probation/{employee_id}/status")
async def update_probation_status(
    employee_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    emp.status = body.get("status", emp.status)
    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.put("/probation/{employee_id}/review")
async def manual_review(
    employee_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    emp.ai_score = body.get("aiScore", emp.ai_score)
    emp.ai_result = body.get("aiResult", emp.ai_result)
    await db.flush()
    await db.refresh(emp)

    return {"code": 0, "message": "ok", "data": serialize_employee(emp)}
