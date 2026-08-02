"""Position tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.field_profiles import (
    POSITION_FIELDS,
    resolve_fields,
    format_projected,
    parse_fields_param,
)


async def _list_positions(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import list_positions as fn
    result = await fn(db=db)
    if result.get("code") != 0:
        return f"❌ 岗位列表查询失败：{result.get('message', '未知错误')}"
    data = result["data"]
    lines = [f"共 {len(data)} 个岗位："]
    for p in data:
        edu = p.get("educationRequirement") or p.get("education_requirement") or ""
        exp = p.get("experienceRequirement") or p.get("experience_requirement") or ""
        sal = p.get("salaryRange") or p.get("salary_range") or ""
        extra = " | ".join(x for x in [
            f"学历:{edu}" if edu else "",
            f"经验:{exp}" if exp else "",
            f"薪资:{sal}" if sal else "",
        ] if x)
        base = f"  [{p['id']}] {p['name']} | 部门: {p.get('department', '未设置')}"
        lines.append(f"{base} | {extra}" if extra else base)
    return "\n".join(lines)


_COL_MAP = {
    "name": "name", "department": "department",
    "jdResponsibilities": "jd_responsibilities",
    "jdRequirements": "jd_requirements",
    "jdPreferred": "jd_preferred",
    "jdTechStack": "jd_tech_stack",
    "educationRequirement": "education_requirement",
    "experienceRequirement": "experience_requirement",
    "ageRequirement": "age_requirement",
    "salaryRange": "salary_range",
    "screeningCriteria": "screening_criteria",
    "interviewCriteriaR1": "interview_criteria_r1",
    "interviewCriteriaR2": "interview_criteria_r2",
    "week1ProjectRequirement": "week1_project_requirement",
    "weeks24Plan": "weeks_2_4_plan",
    "laterWeekScoring": "later_week_scoring",
    "conversionCriteria": "conversion_criteria",
}


async def _get_position(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import get_position as fn
    result = await fn(position_id=params["id"], db=db)
    d = result.get("data") or {}
    if not d:
        return "岗位不存在"
    view, fields, purpose = parse_fields_param(params)
    selected = resolve_fields(
        "position", view=view, fields=fields, purpose=purpose, default_view="core",
    )
    title = (
        f"岗位「{d.get('name', '')}」(ID: {d.get('id', '')})  "
        f"[视图字段: {', '.join(selected)}]"
    )
    body = format_projected(d, selected, POSITION_FIELDS, title=title)
    available = []
    if d.get("screeningCriteria") and "screeningCriteria" not in selected:
        available.append("criteria")
    if (d.get("week1ProjectRequirement") or d.get("conversionCriteria")) and "week1ProjectRequirement" not in selected:
        available.append("probation_plan")
    if available:
        body += (
            f"\n  （未展开的配置可用 view={{{','.join(available)},full}} 再查）"
        )
    return body


async def _create_position(params: dict, db: AsyncSession) -> str:
    """走正规路由函数：保留重名查重（conflict）与字段白名单校验。

    注意 CreatePositionRequest 只覆盖基础字段，评分标准/试用期计划等
    只在 UpdatePositionRequest 里；若模型一次性传了这些，创建成功后
    再补一次 update_position。
    """
    from app.api.recruitment.positions import (
        create_position as fn,
        update_position as update_fn,
        CreatePositionRequest,
        UpdatePositionRequest,
    )

    known = {k: v for k, v in params.items() if k in _COL_MAP and v is not None}
    if not known:
        return "创建岗位失败：没有提供有效字段"
    if not known.get("name"):
        return "创建岗位失败：缺少岗位名称 name"

    create_fields = set(CreatePositionRequest.model_fields) | {
        f.alias for f in CreatePositionRequest.model_fields.values() if f.alias
    }
    base = {k: v for k, v in known.items() if k in create_fields}
    extra = {k: v for k, v in known.items() if k not in create_fields}

    result = await fn(req=CreatePositionRequest(**base), db=db)
    if result.get("code") != 0:
        return f"创建岗位失败：{result.get('message', '')}"
    new_id = (result.get("data") or {}).get("id", "?")

    if extra:
        upd = await update_fn(
            position_id=new_id,
            req=UpdatePositionRequest(**extra),
            db=db,
        )
        if upd.get("code") != 0:
            return (
                f"已创建岗位「{known['name']}」(ID: {new_id})，"
                f"但补充字段写入失败：{upd.get('message', '')}"
            )
    return f"已创建岗位「{known['name']}」(ID: {new_id})"


async def _update_position(params: dict, db: AsyncSession) -> str:
    """走正规路由函数：岗位不存在会返回 404，不再出现「影响 0 行」被判成功。"""
    from app.api.recruitment.positions import (
        update_position as fn,
        UpdatePositionRequest,
    )

    fields = params.get("fields") or {}
    known = {k: v for k, v in fields.items() if k in _COL_MAP and v is not None}
    if not known:
        return "更新岗位失败：没有可更新的字段（提供的字段名可能不正确）"

    result = await fn(
        position_id=params["id"],
        req=UpdatePositionRequest(**known),
        db=db,
    )
    if result.get("code") != 0:
        return f"更新岗位失败：{result.get('message', '')}"

    data = result.get("data") or {}
    previews = []
    for key in known:
        val = data.get(key)
        previews.append(f"{key}={str(val)[:80]}")
    return f"已更新岗位 {params['id']}。回读验证: {'; '.join(previews)}"


async def _delete_position(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import delete_position as fn
    result = await fn(position_id=params["id"], db=db)
    return f"已删除岗位 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"


async def _get_position_questions(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import get_position_questions as fn
    from app.api.talent.interview import normalize_interview_round

    round_val = normalize_interview_round(params.get("round") or "first")
    result = await fn(position_id=params["positionId"], round=round_val, db=db)
    if result.get("code") != 0:
        return f"❌ 岗位题库查询失败：{result.get('message', '未知错误')}"
    qs = result.get("data", []) or []
    if not qs:
        return f"岗位题库（{round_val}）暂无题目"
    lines = [f"岗位题库（{round_val}）共 {len(qs)} 道题："]
    for q in qs:
        lines.append(f"  [{q.get('index','?')}] {q.get('content','')} | 分类:{q.get('category','')} | 难度:{q.get('difficulty','')}")
    return "\n".join(lines)


async def _save_position_questions(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import save_position_questions as fn
    from app.api.talent.interview import normalize_interview_round

    round_val = normalize_interview_round(params.get("round") or "first")
    result = await fn(
        position_id=params["positionId"],
        body={"round": round_val, "questions": params["questions"]},
        db=db,
    )
    return f"已保存岗位题库 {len(params['questions'])} 道题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("list_positions", _list_positions)
    registry.register("get_position", _get_position)
    registry.register("create_position", _create_position)
    registry.register("update_position", _update_position)
    registry.register("delete_position", _delete_position)
    registry.register("get_position_questions", _get_position_questions)
    registry.register("save_position_questions", _save_position_questions)
