"""Dashboard tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


async def _get_operations_dashboard(params: dict, db: AsyncSession) -> str:
    from app.api.system.dashboard import query_recruitment_summary

    summary = await query_recruitment_summary(db)
    total = summary["totalCandidates"]
    return (
        f"招聘运营概览：当前系统共有 {total} 位候选人（简历），"
        f"{summary['totalEmployees']} 名在职员工，平均匹配度 {summary['avgScore']} 分。"
        f"状态分布：求职中 {summary['jobHunting']}，初筛通过 {summary['passed']}，"
        f"一面中 {summary['firstInterview']}，二面中 {summary['secondInterview']}，"
        f"未通过 {summary['failed']}。"
    )


async def _get_dashboard_overview(params: dict, db: AsyncSession) -> str:
    from app.api.system.dashboard import query_recruitment_summary

    summary = await query_recruitment_summary(db)
    return (
        f"数据看板：简历总数 {summary['totalCandidates']}，"
        f"岗位数 {summary['totalPositions']}，"
        f"面试流程中 {summary['totalInterviews']}，"
        f"在职员工 {summary['totalEmployees']}"
    )


def register_handlers(registry) -> None:
    registry.register("get_operations_dashboard", _get_operations_dashboard)
    registry.register("get_dashboard_overview", _get_dashboard_overview)
