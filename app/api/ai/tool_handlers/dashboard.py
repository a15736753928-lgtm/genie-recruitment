"""Dashboard tool handlers."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# 候选人状态词表（与 app/core/state_machine.py TRANSITIONS["candidate"] 一致）。
# 顺序即展示顺序。
_CANDIDATE_STATUS_LABELS: list[tuple[str, str]] = [
    ("new", "新建未解析"),
    ("parsed", "已解析待初筛"),
    ("pending_screen", "待筛选"),
    ("pending_materials", "待补充材料"),
    ("invited", "初筛通过待安排面试"),
    ("round1", "一面中"),
    ("round2", "二面中"),
    ("pending_offer", "待发Offer"),
    ("hired", "已录用"),
    ("talent_pool", "人才池"),
    ("rejected", "未通过/淘汰"),
]


async def _get_operations_dashboard(params: dict, db: AsyncSession) -> str:
    from app.api.system.dashboard import query_recruitment_summary
    from app.models.recruitment import Candidate

    summary = await query_recruitment_summary(db)

    # 逐状态分布（query_recruitment_summary 也返回 byStatus，这里重算是为了
    # 让「未知状态」也能单独兜底展示，不至于在词表再次变动时静默漏掉）。
    rows = (await db.execute(
        select(Candidate.status, func.count()).group_by(Candidate.status)
    )).all()
    counts = {(s or "new"): int(n) for s, n in rows}

    in_flight = sum(counts.get(k, 0) for k in ("invited", "round1", "round2"))
    parts = [
        f"{label} {counts.get(key, 0)}"
        for key, label in _CANDIDATE_STATUS_LABELS
        if counts.get(key, 0) > 0
    ]
    unknown = sorted(k for k in counts if k not in dict(_CANDIDATE_STATUS_LABELS))
    if unknown:
        parts += [f"{k}(未知状态) {counts[k]}" for k in unknown]

    return (
        f"招聘运营概览：当前系统共有 {summary['totalCandidates']} 位候选人（简历），"
        f"{summary['totalPositions']} 个岗位，{in_flight} 位在面试流程中"
        f"（invited/round1/round2），{summary['totalEmployees']} 名在职员工，"
        f"平均匹配度 {summary['avgScore']} 分。\n"
        f"候选人状态分布：{('，'.join(parts)) if parts else '暂无候选人'}。"
    )


def register_handlers(registry) -> None:
    registry.register("get_operations_dashboard", _get_operations_dashboard)
