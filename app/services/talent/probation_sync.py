"""Sync onboarding candidates into probation Employee records."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.probation import Employee, ProbationTask
from app.models.recruitment import Candidate
from app.services.system.system_settings import get_system_setting

# Candidate statuses that should appear on the probation assessment page.
# 与状态机(app/core/state_machine.py TRANSITIONS["candidate"])对齐：
# 只有 hired（Offer 审批通过后）才应该有对应的 Employee 记录。
# 曾经这里包含 "passed"/"onboarded"/"probation"/"offer_pending"(拼写有误，
# 应为 pending_offer) —— 会在候选人尚未经过录用审批时就抢先建员工，
# 与 offer.py 的官方入职流程形成两条并行通道，现已收敛为仅 "hired" 一个值，
# 此函数只作为"hired 但漏建 Employee"的兜底补偿，不再作为入职触发器。
PROBATION_CANDIDATE_STATUSES = frozenset({"hired"})


async def _default_probation_tasks(db: AsyncSession, emp: Employee, join_date: date) -> None:
    task_count = int(await get_system_setting(db, "defaultProbationTasks", 5) or 5)
    task_count = max(1, min(task_count, 12))
    for i in range(1, task_count + 1):
        db.add(
            ProbationTask(
                employee_id=emp.id,
                title=f"第 {i} 周试用期考核任务",
                week_number=i,
                description=f"完成第 {i} 周工作目标与复盘",
                deadline=join_date + timedelta(days=7 * i),
                status="in_progress",
            )
        )


async def ensure_employee_for_candidate(
    db: AsyncSession,
    candidate: Candidate,
    *,
    join_date: date | None = None,
) -> Employee | None:
    """Create a probation Employee for an onboarding candidate if one does not exist."""
    if not candidate or (candidate.status or "") not in PROBATION_CANDIDATE_STATUSES:
        return None

    existing = await db.execute(
        select(Employee).where(Employee.candidate_id == candidate.id).limit(1)
    )
    emp = existing.scalar_one_or_none()
    if emp:
        return emp

    join = join_date or date.today()
    probation_days = int(await get_system_setting(db, "probationDays", 90) or 90)
    probation_end = join + timedelta(days=probation_days)

    department = None
    if candidate.position and candidate.position.department:
        department = candidate.position.department

    emp = Employee(
        candidate_id=candidate.id,
        position_id=candidate.position_id,
        name=candidate.name,
        gender=candidate.gender,
        age=candidate.age,
        department=department,
        onboard_date=join,
        probation_end_date=probation_end,
        status="pending_onboard",
    )
    db.add(emp)
    await db.flush()
    await _default_probation_tasks(db, emp, join)
    await db.flush()
    return emp


async def sync_onboarding_candidates(db: AsyncSession) -> int:
    """Ensure every onboarding-status candidate has a probation Employee row."""
    result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.position))
        .where(Candidate.status.in_(PROBATION_CANDIDATE_STATUSES))
    )
    candidates = result.scalars().all()

    created = 0
    for candidate in candidates:
        before = await db.execute(
            select(Employee.id).where(Employee.candidate_id == candidate.id).limit(1)
        )
        if before.scalar_one_or_none():
            continue
        if await ensure_employee_for_candidate(db, candidate):
            created += 1
    return created
