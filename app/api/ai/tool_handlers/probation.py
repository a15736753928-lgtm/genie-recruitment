"""Probation tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.field_profiles import (
    PROBATION_FIELDS,
    resolve_fields,
    format_projected,
    parse_fields_param,
)


async def _list_probation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import list_probation as fn
    try:
        page_size = int(params.get("limit", 10) or 10)
    except (TypeError, ValueError):
        page_size = 10
    result = await fn(
        department=params.get("department") or "all",
        status=params.get("status") or "all",
        page=1,
        pageSize=page_size,
        db=db,
    )
    if result.get("code") != 0:
        return f"❌ 试用期员工查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", {})
    if not isinstance(data, dict):
        return "查询完成"
    total = data.get("total", 0)
    items = data.get("list") or data.get("employees") or data.get("items") or []
    if not items:
        return f"试用期员工：共 {total} 人（无明细列表）"
    try:
        limit = int(params.get("limit", 15) or 15)
    except (TypeError, ValueError):
        limit = 15
    lines = [f"试用期员工：共 {total} 人，以下前 {min(limit, len(items))} 位："]
    for e in items[:limit]:
        lines.append(
            f"  [{e.get('id','')}] {e.get('name','')} | "
            f"{e.get('positionName') or e.get('position') or ''} | "
            f"状态:{e.get('status','')} | 导师:{e.get('mentorName') or '—'} | "
            f"进度:{e.get('taskProgress', '—')}%"
        )
    return "\n".join(lines)


async def _get_probation_stats(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import get_probation_stats as fn
    result = await fn(db=db)
    if result.get("code") != 0:
        return f"❌ 试用期统计查询失败：{result.get('message', '未知错误')}"
    d = result.get("data", {})
    # 路由返回的 key 仍是 total/assessing/passed/failed，但口径已是新词表：
    # assessing = training+probation+pending_confirmation，passed = formal，
    # failed = transferred+resigned。这里按新词表说明，避免模型沿用旧语义。
    return (
        f"试用期统计：员工总数 {d.get('total', 0)}，"
        f"在途 {d.get('assessing', 0)}（training/probation/pending_confirmation），"
        f"已转正 {d.get('passed', 0)}（formal），"
        f"已离场 {d.get('failed', 0)}（transferred/resigned）"
    )


async def _get_probation_employee(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import get_probation_employee as fn
    result = await fn(employee_id=params["id"], db=db)
    d = result.get("data") or {}
    if not d:
        return "试用期员工不存在"
    view, fields, purpose = parse_fields_param(params)
    selected = resolve_fields(
        "probation", view=view, fields=fields, purpose=purpose, default_view="core",
    )
    title = (
        f"试用期员工「{d.get('name', '')}」(ID: {d.get('id', '')})  "
        f"[视图字段: {', '.join(selected)}]"
    )
    return format_projected(d, selected, PROBATION_FIELDS, title=title, max_text=600, max_json=800)


async def _create_probation_employee(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import create_employee as fn, CreateEmployeeRequest
    from app.models.recruitment import Candidate
    from sqlalchemy import select

    # 守卫：候选人只有在 hired（录用审批通过）之后才允许生成员工档案。
    # 被调路由在候选人非 hired 时会落到 fallback 分支「凭空建员工」，
    # 形成一条绕过 Offer 审批的入职通道，这里提前拦掉。
    candidate_id = params.get("candidateId")
    if candidate_id:
        cand = (
            await db.execute(select(Candidate).where(Candidate.id == candidate_id))
        ).scalar_one_or_none()
        if cand is None:
            return f"新增试用期员工失败：候选人 {candidate_id} 不存在，请先用 list_resumes 确认 ID。"
        if (cand.status or "") != "hired":
            return (
                f"新增试用期员工失败：候选人「{cand.name}」当前状态为 {cand.status}，尚未录用。"
                f"员工档案只能由「录用审批」通过（候选人变为 hired）后自动生成，"
                f"请先走 Offer 审批流程，不要在此绕过。"
            )

    req = CreateEmployeeRequest(**params)
    result = await fn(req=req, db=db)
    if result["code"] == 0:
        return f"已新增试用期员工「{params['name']}」"
    return f"新增失败：{result.get('message', '')}"


async def _create_probation_task(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import create_probation_task as fn
    result = await fn(body=params, db=db)
    return f"已为员工 {params['employeeId']} 新增任务「{params.get('title', '')}」" if result["code"] == 0 else f"新增失败：{result.get('message', '')}"


async def _update_probation_task(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import update_probation_task as fn
    result = await fn(task_id=params["taskId"], body=params.get("fields", {}), db=db)
    return f"已更新任务 {params['taskId']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"


async def _ai_evaluate_probation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import ai_evaluate_probation as fn
    result = await fn(employee_id=params["employeeId"], db=db)
    if result["code"] == 0:
        d = result.get("data", {})
        return f"AI评估完成：综合分 {d.get('score', 0)}，结论 {d.get('result', '')}。{d.get('comment', '')}"
    return f"AI评估失败：{result.get('message', '')}"


async def _update_probation_status(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import update_probation_status as fn
    result = await fn(employee_id=params["employeeId"], body={"status": params["status"]}, db=db)
    return f"已更新员工 {params['employeeId']} 状态为 {params['status']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"


async def _manual_review_probation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.probation import manual_review as fn
    result = await fn(employee_id=params["employeeId"], body={"aiScore": params.get("aiScore"), "aiResult": params.get("aiResult")}, db=db)
    return f"已手动录入员工 {params['employeeId']} 的评估" if result["code"] == 0 else f"录入失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("list_probation", _list_probation)
    registry.register("get_probation_stats", _get_probation_stats)
    registry.register("get_probation_employee", _get_probation_employee)
    registry.register("create_probation_employee", _create_probation_employee)
    registry.register("create_probation_task", _create_probation_task)
    registry.register("update_probation_task", _update_probation_task)
    registry.register("ai_evaluate_probation", _ai_evaluate_probation)
    registry.register("update_probation_status", _update_probation_status)
    registry.register("manual_review_probation", _manual_review_probation)
