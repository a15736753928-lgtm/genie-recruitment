"""Dashboard tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


async def _get_operations_dashboard(params: dict, db: AsyncSession) -> str:
    from app.api.system.dashboard import get_operations as fn
    result = await fn(db=db)
    data = result.get("data", {})
    summary = data.get("summary", {})
    return f"运营概览：{summary.get('title', '')} - {summary.get('text', '')}"


async def _get_dashboard_overview(params: dict, db: AsyncSession) -> str:
    from app.api.system.dashboard import get_overview as fn
    result = await fn(db=db)
    d = result.get("data", {})
    stats = d.get("stats", {})
    return (
        f"数据看板：简历总数 {stats.get('totalResumes', 0)}，"
        f"岗位数 {stats.get('totalPositions', 0)}，"
        f"面试中 {stats.get('totalInterviews', 0)}，"
        f"在职员工 {stats.get('totalEmployees', 0)}"
    )


def register_handlers(registry) -> None:
    registry.register("get_operations_dashboard", _get_operations_dashboard)
    registry.register("get_dashboard_overview", _get_dashboard_overview)
