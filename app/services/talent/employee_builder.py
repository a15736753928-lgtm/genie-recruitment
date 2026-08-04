"""候选人 → 员工档案的统一构建器（2026-08-03）。

背景：候选人接受 Offer 后生成待入职员工档案。此前存在两条创建路径
（offer_public.public_accept_offer 与 probation_sync.ensure_employee_for_candidate），
拷贝字段不一致——offer_public 带 phone/email 缺 gender/age，
probation_sync 带 gender/age 缺 phone/email。统一收敛到本模块。

规则：
  - 身份信息（name/gender/age/phone/email）一律从候选人主档拷贝（单一事实源）
  - department 优先取 Offer 快照（防岗位改部门后员工档案漂移），否则取岗位部门
  - 不拷贝履历子表（技能/教育/工作/项目）——它们仍归属候选人主档，需要时联查
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recruitment import Candidate
from app.models.probation import Employee


def build_employee_from_candidate(
    candidate: Candidate,
    *,
    position_id=None,
    department: Optional[str] = None,
    onboard_date: Optional[date] = None,
    probation_months: Optional[int] = None,
) -> Employee:
    """按统一规则构造 Employee（不落库，由调用方 db.add + flush）。"""
    onboard = onboard_date or datetime.utcnow().date()
    probation_end = None
    if probation_months:
        probation_end = onboard + timedelta(days=probation_months * 30)

    emp = Employee(
        candidate_id=candidate.id,
        position_id=position_id or candidate.position_id,
        name=candidate.name,
        gender=candidate.gender,
        age=candidate.age,
        phone=candidate.phone,
        email=candidate.email,
        department=department,
        onboard_date=onboard,
        probation_end_date=probation_end,
        status="pending_onboard",
    )
    return emp


async def ensure_employee_uniform(
    db: AsyncSession,
    candidate: Candidate,
    *,
    position_id=None,
    department: Optional[str] = None,
    onboard_date: Optional[date] = None,
    probation_months: Optional[int] = None,
) -> Employee:
    """幂等创建：候选人已有员工档案则返回既有，否则用统一构建器创建。"""
    from sqlalchemy import select

    existing = await db.execute(
        select(Employee).where(Employee.candidate_id == candidate.id).limit(1)
    )
    emp = existing.scalar_one_or_none()
    if emp:
        return emp
    emp = build_employee_from_candidate(
        candidate,
        position_id=position_id,
        department=department,
        onboard_date=onboard_date,
        probation_months=probation_months,
    )
    db.add(emp)
    await db.flush()
    return emp
