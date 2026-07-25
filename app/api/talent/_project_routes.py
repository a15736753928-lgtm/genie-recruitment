"""项目 & 项目人员推荐路由 —— 挂到 phase4 的 router 上。

拆到单独文件仅为控制单文件长度；phase4.py 末尾 import 并调用 register_project_routes(router)。
"""
from __future__ import annotations
import uuid
from typing import Optional, List

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.phase4 import Project, ProjectAssignment, TalentProfile
from app.models.probation import Employee
from app.core.security import get_current_user, require_permission, CurrentUser
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit

_LEVEL_NUM = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


def _serialize_project(p: Project, assigned_count: int = 0) -> dict:
    return {
        "id": str(p.id), "name": p.name, "description": p.description,
        "skills": p.required_skills or [], "requiredSkills": p.required_skills or [],
        "level": p.required_level, "headcount": p.headcount,
        "department": p.department, "status": p.status,
        "assignedCount": assigned_count,
        "createdAt": p.created_at.isoformat() if p.created_at else None,
    }


def _serialize_assignment(a: ProjectAssignment) -> dict:
    return {
        "id": str(a.id), "projectId": str(a.project_id),
        "employeeId": str(a.employee_id), "employeeName": a.employee_name,
        "assignedBy": a.assigned_by,
        "assignedAt": a.assigned_at.isoformat() if a.assigned_at else None,
    }


class CreateAssignmentRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    employee_name: Optional[str] = Field(None, alias="employeeName")


def register_project_routes(router):
    """把项目相关路由注册到传入的 APIRouter(在 phase4.py 末尾调用)。"""

    @router.get("/projects")
    async def list_projects(
        current: CurrentUser = Depends(require_permission("talent:view")),
        db: AsyncSession = Depends(get_db),
    ):
        rows = (await db.execute(
            select(Project).options(selectinload(Project.assignments))
            .order_by(Project.created_at.desc())
        )).scalars().all()
        return ok([_serialize_project(p, len(p.assignments or [])) for p in rows])

    @router.get("/projects/{project_id}/recommendations")
    async def project_recommendations(
        project_id: str,
        current: CurrentUser = Depends(require_permission("talent:view")),
        db: AsyncSession = Depends(get_db),
    ):
        """按项目技能匹配人才,返回带 matchScore 的推荐列表(服务端计算)。"""
        try:
            pid = uuid.UUID(project_id)
        except ValueError:
            return not_found("项目不存在")
        proj = (await db.execute(select(Project).where(Project.id == pid))).scalar_one_or_none()
        if not proj:
            return not_found("项目不存在")

        need_skills: List[str] = proj.required_skills or []
        need_level = _LEVEL_NUM.get(proj.required_level or "L3", 3)

        assigned_rows = (await db.execute(
            select(ProjectAssignment.employee_id).where(ProjectAssignment.project_id == pid)
        )).scalars().all()
        assigned_ids = {str(e) for e in assigned_rows}

        profiles = (await db.execute(select(TalentProfile))).scalars().all()
        emp_ids = [p.employee_id for p in profiles]
        emp_map = {}
        if emp_ids:
            emps = (await db.execute(select(Employee).where(Employee.id.in_(emp_ids)))).scalars().all()
            emp_map = {e.id: e for e in emps}

        recs = []
        for p in profiles:
            skills = p.skills or []
            matched = [s for s in need_skills if s in skills]
            if not matched:
                continue
            lvl = _LEVEL_NUM.get(p.ability_level or "L1", 1)
            score = min(100, len(matched) / max(len(need_skills), 1) * 100 + (20 if lvl >= need_level else 0))
            emp = emp_map.get(p.employee_id)
            recs.append({
                "employeeId": str(p.employee_id),
                "name": emp.name if emp else "",
                "abilityLevel": p.ability_level,
                "skills": skills,
                "department": p.department,
                "matchedSkills": matched,
                "matchScore": round(score),
                "joined": str(p.employee_id) in assigned_ids,
            })
        recs.sort(key=lambda r: r["matchScore"], reverse=True)
        return ok(recs)

    @router.post("/projects/{project_id}/assignments")
    async def add_assignment(
        project_id: str,
        body: CreateAssignmentRequest,
        current: CurrentUser = Depends(require_permission("talent:view")),
        db: AsyncSession = Depends(get_db),
    ):
        try:
            pid = uuid.UUID(project_id)
            eid = uuid.UUID(body.employee_id)
        except ValueError:
            return not_found("项目或员工不存在")
        proj = (await db.execute(
            select(Project).options(selectinload(Project.assignments)).where(Project.id == pid)
        )).scalar_one_or_none()
        if not proj:
            return not_found("项目不存在")

        existing = list(proj.assignments or [])
        if any(a.employee_id == eid for a in existing):
            return fail(409, "该员工已加入本项目")
        if len(existing) >= proj.headcount:
            return fail(409, f"项目人数已满({proj.headcount})")

        a = ProjectAssignment(
            project_id=pid, employee_id=eid,
            employee_name=body.employee_name,
            assigned_by=current.username,
        )
        db.add(a)
        if len(existing) + 1 >= proj.headcount:
            proj.status = "staffed"
        await db.flush()
        await write_audit(db, actor=current.username,
                          action=f"项目组队: 加入 {body.employee_name or eid}", section="project")
        return ok(_serialize_assignment(a))

    @router.get("/projects/{project_id}/assignments")
    async def list_assignments(
        project_id: str,
        current: CurrentUser = Depends(require_permission("talent:view")),
        db: AsyncSession = Depends(get_db),
    ):
        try:
            pid = uuid.UUID(project_id)
        except ValueError:
            return not_found("项目不存在")
        rows = (await db.execute(
            select(ProjectAssignment).where(ProjectAssignment.project_id == pid)
            .order_by(ProjectAssignment.assigned_at.desc())
        )).scalars().all()
        return ok([_serialize_assignment(a) for a in rows])
