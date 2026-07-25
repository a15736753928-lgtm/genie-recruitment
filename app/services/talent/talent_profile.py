"""人才画像服务 —— 第二期提供 idempotent 空壳, 第四期实现完整指标聚合。

P2→P4 接缝: 员工转正成功时调用 ensure_talent_profile(db, employee_id),
确保 talent_profiles 有一条 L1 初始记录。第四期 refresh 时填充具体指标。
"""
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_talent_profile(db: AsyncSession, employee_id: uuid.UUID) -> None:
    """幂等: 不存在则插入一条 ability_level=L1 的初始画像(为第四期预留)。"""
    # 第四期 hard-import 替换为 app.models.phase4.TalentProfile
    # 当前不创建表, 仅作为接缝占位
    pass


async def refresh_talent_profile(db: AsyncSession, employee_id: uuid.UUID) -> None:
    """第四期: 聚合 point_records / work_tasks / ability_tags 更新画像。"""
    pass
