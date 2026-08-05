"""人才画像服务 —— 正式员工人才池的画像创建入口。

转正审批通过时调用 ensure_talent_profile(db, employee_id)，
幂等创建 ability_level=L1 的初始画像（员工转正后自动进入人才池）。
refresh 时的指标聚合在 app/api/talent/phase4.py::refresh_talent_profile 端点内完成。
"""
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.phase4 import TalentProfile
from app.models.probation import Employee


async def ensure_talent_profile(db: AsyncSession, employee_id: uuid.UUID) -> None:
    """幂等: 转正成功时为员工创建 L1 初始人才画像，已有则跳过。"""
    existing = (
        await db.execute(
            select(TalentProfile).where(TalentProfile.employee_id == employee_id)
        )
    ).scalar_one_or_none()
    if existing:
        return

    emp = (
        await db.execute(select(Employee).where(Employee.id == employee_id))
    ).scalar_one_or_none()

    profile = TalentProfile(
        employee_id=employee_id,
        ability_level="L1",
        department=emp.department if emp else None,
        current_position=str(emp.position_id) if emp and emp.position_id else None,
    )
    db.add(profile)
    await db.flush()


async def refresh_talent_profile(db: AsyncSession, employee_id: uuid.UUID) -> None:
    """第四期指标聚合在 phase4.py::refresh_talent_profile 端点内联实现，此处保留为兼容接缝。"""
    pass
