"""Performance tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


async def _list_performance(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import list_performance as fn
    result = await fn(quarter=params["quarter"], db=db)
    data = result.get("data", {})
    if not isinstance(data, dict):
        return "查询完成"
    total = data.get("total", 0)
    items = data.get("list") or data.get("records") or data.get("items") or []
    if not items:
        return f"绩效数据：共 {total} 条记录（无明细）"
    limit = int(params.get("limit", 15) or 15)
    lines = [f"绩效数据（{params.get('quarter','')}）：共 {total} 条，前 {min(limit, len(items))} 条："]
    for r in items[:limit]:
        lines.append(
            f"  [{r.get('id') or r.get('employeeId','')}] "
            f"{r.get('name') or r.get('employeeName','')} | "
            f"分:{r.get('totalScore') or r.get('score','—')} | "
            f"等级:{r.get('grade','—')} | 奖金:{r.get('bonus','—')}"
        )
    return "\n".join(lines)


async def _get_performance_stats(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import get_performance_stats as fn
    result = await fn(quarter=params["quarter"], db=db)
    d = result.get("data", {})
    return f"绩效统计：参与 {d.get('participants', 0)} 人，平均分 {d.get('avgScore', 0)}，优秀 {d.get('excellentCount', 0)}，待改进 {d.get('needsImprovement', 0)}"


async def _get_department_performance(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import get_department_performance as fn
    result = await fn(quarter=params["quarter"], db=db)
    data = result.get("data", [])
    return f"部门绩效共 {len(data)} 个部门"


async def _get_grade_distribution(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import get_grade_distribution as fn
    result = await fn(quarter=params["quarter"], db=db)
    data = result.get("data", [])
    parts = [f"{g['grade']}:{g['count']}" for g in data]
    return f"等级分布：{', '.join(parts)}"


async def _get_bonus_info(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import get_bonus_info as fn
    result = await fn(quarter=params["quarter"], db=db)
    d = result.get("data", {})
    return f"奖金池：总额 {d.get('totalPool', 0)}，已分配 {d.get('distributed', 0)}，待分配 {d.get('pending', 0)}"


async def _get_quarter_trends(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import get_quarter_trends as fn
    result = await fn(db=db)
    data = result.get("data", [])
    return f"近 {len(data)} 个季度趋势"


async def _initiate_appraisal(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import initiate_appraisal as fn
    result = await fn(body={"quarter": params["quarter"], "employeeIds": params["employeeIds"]}, db=db)
    return f"已为 {params['quarter']} 发起绩效考核，{len(params['employeeIds'])} 人参与" if result["code"] == 0 else f"发起失败：{result.get('message', '')}"


async def _update_bonus(params: dict, db: AsyncSession) -> str:
    from app.api.talent.performance import update_bonus as fn
    result = await fn(employee_id=params["employeeId"], body={"bonus": params["bonus"]}, db=db)
    return f"已更新员工 {params['employeeId']} 奖金为 {params['bonus']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("list_performance", _list_performance)
    registry.register("get_performance_stats", _get_performance_stats)
    registry.register("get_department_performance", _get_department_performance)
    registry.register("get_grade_distribution", _get_grade_distribution)
    registry.register("get_bonus_info", _get_bonus_info)
    registry.register("get_quarter_trends", _get_quarter_trends)
    registry.register("initiate_appraisal", _initiate_appraisal)
    registry.register("update_bonus", _update_bonus)
