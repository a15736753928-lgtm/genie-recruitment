import json
from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.database import get_db
from app.models.probation import Employee, ProbationTask
from app.models.recruitment import Candidate, Position
from app.config import get_settings
from app.services.system.system_settings import get_system_setting
from app.services.ai import get_llm_client
from app.services.talent.probation_sync import ensure_employee_for_candidate, sync_onboarding_candidates

router = APIRouter(tags=["试用期"])
settings = get_settings()


# ── Serializers ─────────────────────────────────────────



def serialize_employee(emp: Employee) -> dict:
    tasks = list(emp.tasks or [])
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.status == "completed")
    tasks_by_week = {}
    for t in tasks:
        wk = t.week_number or 0
        if wk not in tasks_by_week:
            tasks_by_week[wk] = []
        tasks_by_week[wk].append({
            "id": str(t.id),
            "title": t.title,
            "description": t.description,
            "status": t.status,
            "deadline": t.deadline.isoformat() if t.deadline else "",
            "reviewNotes": t.review_notes,
        })

    position_name = emp.position.name if emp.position else ""

    return {
        "id": str(emp.id),
        "candidateId": str(emp.candidate_id) if emp.candidate_id else "",
        "positionId": str(emp.position_id) if emp.position_id else "",
        "positionName": position_name,
        "name": emp.name,
        "gender": emp.gender,
        "age": emp.age,
        "department": emp.department,
        "joinDate": emp.join_date.isoformat() if emp.join_date else "",
        "probationEnd": emp.probation_end.isoformat() if emp.probation_end else "",
        "status": emp.status or "assessing",
        "mentorName": emp.mentor,        "taskProgress": round(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        "totalTasks": total_tasks,
        "completedTasks": completed_tasks,
        "aiScore": emp.ai_score,
        "aiResult": emp.ai_result,
        "tasks": [
            {"id": str(t.id), "title": t.title, "weekNumber": t.week_number,
             "status": t.status, "description": t.description,
             "deadline": t.deadline.isoformat() if t.deadline else "",
             "reviewNotes": t.review_notes}
            for t in tasks
        ],
        "tasksByWeek": tasks_by_week,
    }


# ── Endpoints ───────────────────────────────────────────

@router.get("/probation/stats")
async def get_probation_stats(db: AsyncSession = Depends(get_db)):
    await sync_onboarding_candidates(db)

    total_query = select(func.count()).select_from(Employee)
    assessing_query = select(func.count()).select_from(Employee).where(Employee.status == "assessing")
    passed_query = select(func.count()).select_from(Employee).where(Employee.status == "passed")
    failed_query = select(func.count()).select_from(Employee).where(Employee.status == "failed")

    total = (await db.execute(total_query)).scalar() or 0
    assessing = (await db.execute(assessing_query)).scalar() or 0
    passed = (await db.execute(passed_query)).scalar() or 0
    failed = (await db.execute(failed_query)).scalar() or 0

    return {
        "code": 0, "message": "ok",
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
    await sync_onboarding_candidates(db)

    query = select(Employee).options(
        selectinload(Employee.tasks),
        selectinload(Employee.position),
    )

    if department and department not in ("all", "全部部门"):
        query = query.where(Employee.department == department)
    if status and status != "all":
        query = query.where(Employee.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(Employee.created_at.desc()).offset((page - 1) * pageSize).limit(pageSize)
    result = await db.execute(query)
    employees = result.unique().scalars().all()

    return {
        "code": 0, "message": "ok",
        "data": {
            "list": [serialize_employee(e) for e in employees],
            "total": total, "page": page, "pageSize": pageSize,
        },
    }


@router.get("/probation/{employee_id}")
async def get_probation_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Employee).options(
            selectinload(Employee.tasks),
            selectinload(Employee.position),
            
        ).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_employee(emp)}


# ── Employee management ──

class CreateEmployeeRequest(BaseModel):
    candidateId: Optional[str] = None
    positionId: Optional[str] = None
    name: str
    gender: Optional[str] = None
    age: Optional[int] = None
    department: Optional[str] = None
    joinDate: Optional[str] = None
    probationEnd: Optional[str] = None
    mentorName: Optional[str] = None
    mentorId: Optional[str] = None


@router.post("/probation/employees")
async def create_employee(
    req: CreateEmployeeRequest,
    db: AsyncSession = Depends(get_db),
):
    if req.candidateId:
        cand_result = await db.execute(
            select(Candidate)
            .options(selectinload(Candidate.position))
            .where(Candidate.id == req.candidateId)
        )
        candidate = cand_result.scalar_one_or_none()
        if candidate:
            join_date = date.fromisoformat(req.joinDate) if req.joinDate else date.today()
            emp = await ensure_employee_for_candidate(db, candidate, join_date=join_date)
            if emp:
                if req.department:
                    emp.department = req.department
                if req.mentorName:
                    emp.mentor = req.mentorName
                if req.mentorId:
                    emp.mentor_id = req.mentorId
                await db.flush()
                await db.refresh(emp, attribute_names=["tasks", "position"])
                return {"code": 0, "message": "ok", "data": serialize_employee(emp)}

    join_date = date.fromisoformat(req.joinDate) if req.joinDate else date.today()
    if req.probationEnd:
        probation_end = date.fromisoformat(req.probationEnd)
    else:
        probation_days = int(await get_system_setting(db, "probationDays", 90) or 90)
        probation_end = join_date + timedelta(days=probation_days)

    emp = Employee(
        candidate_id=req.candidateId,
        position_id=req.positionId,
        name=req.name,
        gender=req.gender,
        age=req.age,
        department=req.department,
        join_date=join_date,
        probation_end=probation_end,
        mentor_name=req.mentorName,
        mentor_id=req.mentorId,
        status="assessing",
    )
    db.add(emp)
    await db.flush()

    # 按 defaultProbationTasks 批量生成默认周任务
    task_count = int(await get_system_setting(db, "defaultProbationTasks", 5) or 5)
    task_count = max(1, min(task_count, 12))
    for i in range(1, task_count + 1):
        db.add(ProbationTask(
            employee_id=emp.id,
            title=f"第 {i} 周试用期考核任务",
            week_number=i,
            description=f"完成第 {i} 周工作目标与复盘",
            deadline=join_date + timedelta(days=7 * i),
            status="pending",
        ))

    await db.flush()
    await db.refresh(emp)
    return {"code": 0, "message": "ok", "data": serialize_employee(emp)}


# ── Tasks ──

@router.post("/probation/tasks")
async def create_probation_task(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    employee_id = body.get("employeeId")
    week_number = body.get("weekNumber")
    task_title = body.get("taskTitle", body.get("title", ""))
    task_description = body.get("description", "")
    deadline_str = body.get("deadline", "")

    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    task = ProbationTask(
        employee_id=employee_id,
        title=task_title,
        week_number=week_number,
        description=task_description,
        deadline=date.fromisoformat(deadline_str) if deadline_str else None,
    )
    db.add(task)
    await db.flush()
    await db.refresh(emp)
    return {"code": 0, "message": "ok", "data": serialize_employee(emp)}


@router.put("/probation/tasks/{task_id}")
async def update_probation_task(
    task_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ProbationTask).where(ProbationTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        return {"code": 404, "message": "任务不存在", "data": None}

    for field in ["title", "status", "description", "reviewNotes"]:
        if field in body:
            setattr(task, field, body[field])
    if "weekNumber" in body:
        task.week_number = body["weekNumber"]
    if "deadline" in body and body["deadline"]:
        task.deadline = date.fromisoformat(body["deadline"])

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


# ── AI Evaluate ──

@router.post("/probation/{employee_id}/ai-evaluate")
async def ai_evaluate_probation(employee_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Employee).options(selectinload(Employee.tasks)).where(Employee.id == employee_id)
    )
    emp = result.scalar_one_or_none()
    if not emp:
        return {"code": 404, "message": "员工不存在", "data": None}

    tasks = [{"title": t.title, "status": t.status, "week": t.week_number}
             for t in (emp.tasks or [])]

    prompt = f"""作为HR试用期评估专家,请对以下员工进行试用期表现评估。

员工:{emp.name}
部门:{emp.department or '未指定'}
入职日期:{emp.join_date}
试用期截止:{emp.probation_end}

任务数据:{json.dumps(tasks, ensure_ascii=False)}

请从以下维度评估并返回JSON:
1. 项目表现(60分满分)
2. 技术能力(20分满分)
3. 团队协作(20分满分)

返回格式:{{"projectPerformance": 分数, "techCapability": 分数, "collaboration": 分数, "score": 综合总分, "result": "converted/extended/rejected", "comment": "评估意见"}}
只返回JSON。"""

    try:
        response = await get_llm_client().chat.completions.create(
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
