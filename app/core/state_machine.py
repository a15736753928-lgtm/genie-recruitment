"""状态机 —— 合法迁移表 + 写流转日志的核心函数 (§2.3)。

所有状态变更必须经 transition();禁止在业务代码里直接 entity.status = x。
非法迁移 -> StateError -> 端点层返回 code:409。

支持的 entity_type:
  candidate / employee / task / offer / recruitment_request
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.system import ExceptionQueue


class StateError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)


# ── 合法迁移表 ──────────────────────────────────────────────
#   None 表示初始状态(insert)可直接到该值
#   terminal 集合内的状态不可再迁移(除非明确列出)

TRANSITIONS: dict[str, dict[str, list[str | None]]] = {
    "candidate": {
        # to_status: [合法 from_status, ...] (None=初始)
        "new":              [None],
        "parsed":           ["new"],
        # AI 评分是解析与初筛的合并动作，简历入库后可能直接从 new 打分进筛选池，
        # 不一定先落 parsed。只允许 parsed 会让候选人永远卡在 new，整条筛选链断掉。
        "pending_screen":   ["new", "parsed"],
        "pending_materials":["pending_screen", "pending_materials"],
        "invited":          ["pending_screen", "pending_materials"],
        "round1":           ["invited"],
        "round2":           ["round1"],
        "pending_offer":    ["round2", "round1"],
        "hired":            ["pending_offer"],
        "talent_pool":      [
            "pending_screen", "pending_materials", "invited",
            "round1", "round2", "pending_offer",   # 任何非终止可进人才池
        ],
        "rejected":         [
            "pending_screen", "pending_materials", "invited",
            "round1", "round2", "pending_offer",
        ],
    },
    "employee": {
        # 完整生命周期(第二期会用全部迁移)
        "pending_onboard":       [None, "talent_pool"],
        "training":              ["pending_onboard"],
        # pending_confirmation → probation 是「转正未通过、延长试用期」的正规回退路径。
        # 缺了它会让 phase2.approve_confirmation 的拒绝分支 100% 失败。
        "probation":             ["training", "pending_confirmation"],
        "pending_confirmation":  ["probation"],
        "formal":                ["pending_confirmation"],
        "transferred":           ["formal", "probation"],
        "resigned":              ["formal", "probation", "training", "pending_onboard"],
    },
    # 注意: 试用期任务与工作任务是两个不同实体, 状态词表也不同,
    # 必须分开登记 —— 曾因共用 "task" 键导致工作任务发布必定 409。
    "probation_task": {
        # ProbationTask (probation_tasks 表), 见 models/probation.py
        # 自动建任务时直接以 in_progress 起步, 故 in_progress 允许 None
        "draft":           [None],
        "pending_confirm": ["draft"],
        "in_progress":     [None, "pending_confirm", "rework"],
        "pending_review":  ["in_progress"],
        "passed":          ["pending_review"],
        "rework":          ["pending_review"],
        "closed":          ["passed", "rework"],
    },
    "work_task": {
        # WorkTask (work_tasks 表), 见 models/phase3.py
        "draft":           [None],
        "pending":         ["draft"],
        "in_progress":     ["pending", "rework"],
        "pending_accept":  ["in_progress"],
        "passed":          ["pending_accept"],
        "rework":          ["pending_accept"],
        "closed":          ["passed", "rework"],
    },
    "offer": {
        "pending":       [None],
        "approved":      ["pending"],
        "rejected":      ["pending"],
        "conditional":   ["pending"],
    },
    "recruitment_request": {
        "draft":           [None],
        "ai_generated":    ["draft"],
        "hr_confirmed":    ["ai_generated", "draft"],
        "dept_confirmed":  ["hr_confirmed"],
        "published":       ["dept_confirmed"],
        "closed":          ["published"],
    },
}


def _is_legal(entity_type: str, from_status: str | None, to_status: str) -> bool:
    table = TRANSITIONS.get(entity_type)
    if table is None:
        return True  # 未知类型不拦截(宽容)
    allowed_froms = table.get(to_status)
    if allowed_froms is None:
        return False  # to_status 不在表中
    return from_status in allowed_froms


async def _has_blocking_exception(db: AsyncSession, entity_type: str, entity_id: uuid.UUID) -> str | None:
    """返回首个 open+block 异常的 exception_type;无则返回 None。"""
    row = await db.execute(
        select(ExceptionQueue).where(
            ExceptionQueue.entity_type == entity_type,
            ExceptionQueue.entity_id == entity_id,
            ExceptionQueue.status == "open",
            ExceptionQueue.severity == "block",
        ).limit(1)
    )
    exc = row.scalar_one_or_none()
    return exc.exception_type if exc else None


async def transition(
    db: AsyncSession,
    entity_type: str,
    entity: Any,
    to_status: str,
    *,
    actor_id: uuid.UUID | None = None,
    actor_name: str = "系统",
    reason: str | None = None,
    evidence: list[dict] | None = None,
    skip_block_check: bool = False,   # 人工强制处理异常后可跳过
) -> None:
    """变更 entity.status 并写 StateTransition 流转日志。

    非法迁移 -> StateError(端点 -> code:409)。
    存在 block 级异常且未指定 skip -> StateError。
    """
    # 懒导入避免循环
    from app.models.system import StateTransition

    from_status = getattr(entity, "status", None)

    if not _is_legal(entity_type, from_status, to_status):
        raise StateError(
            f"{entity_type} 状态 {from_status!r} → {to_status!r} 不合法"
        )

    # block 级异常阻止自动推进
    if not skip_block_check:
        block_type = await _has_blocking_exception(db, entity_type, entity.id)
        if block_type:
            raise StateError(
                f"存在未处理的 block 级异常 [{block_type}]，请先在异常队列中处理"
            )

    entity.status = to_status
    db.add(StateTransition(
        entity_type=entity_type,
        entity_id=entity.id,
        from_status=from_status,
        to_status=to_status,
        reason=reason,
        evidence=evidence,
        actor_id=actor_id,
        actor_name=actor_name or "系统",
    ))
    await db.flush()
