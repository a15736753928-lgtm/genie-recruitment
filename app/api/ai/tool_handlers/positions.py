"""Position tool handlers."""

from __future__ import annotations

import json

from sqlalchemy import text as sa_text
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
    columns = []
    values = {}
    for camel_key, value in params.items():
        col = _COL_MAP.get(camel_key)
        if col and value is not None:
            columns.append(col)
            values[col] = value
    if not columns:
        return "创建失败：没有提供有效字段"
    placeholders = ", ".join(f":{c}" for c in columns)
    cols_str = ", ".join(columns)
    sql = (
        f"INSERT INTO positions (id, {cols_str}) "
        f"VALUES (gen_random_uuid(), {placeholders}) RETURNING id"
    )
    result = await db.execute(sa_text(sql), values)
    row = result.fetchone()
    new_id = str(row[0]) if row else "?"
    await db.flush()
    return f"已创建岗位「{params.get('name', '未命名')}」(ID: {new_id})"


async def _update_position(params: dict, db: AsyncSession) -> str:
    fields = params.get("fields", {})
    position_id = params["id"]
    set_clauses = []
    values = {"id": position_id}
    for camel_key, value in fields.items():
        col = _COL_MAP.get(camel_key)
        if col and value is not None:
            set_clauses.append(f"{col} = :{col}")
            if isinstance(value, dict):
                values[col] = json.dumps(value, ensure_ascii=False)
            else:
                values[col] = value
    if not set_clauses:
        return "没有可更新的字段（提供的字段名可能不正确）"
    sql = f"UPDATE positions SET {', '.join(set_clauses)} WHERE id = :id"
    result = await db.execute(sa_text(sql), values)
    await db.flush()
    rowcount = result.rowcount
    verify_cols = ", ".join(_COL_MAP[c] for c in fields if c in _COL_MAP)
    previews = []
    if verify_cols:
        verify = await db.execute(
            sa_text(f"SELECT {verify_cols} FROM positions WHERE id = :id"),
            {"id": position_id},
        )
        row = verify.fetchone()
        if row:
            for i, col_name in enumerate(verify_cols.split(", ")):
                val = row[i]
                preview = (str(val) or "")[:80]
                previews.append(f"{col_name}={preview}")
    preview_text = "; ".join(previews) if previews else "ok"
    return f"已更新岗位，影响 {rowcount} 行。回读验证: {preview_text}"


async def _delete_position(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import delete_position as fn
    result = await fn(position_id=params["id"], db=db)
    return f"已删除岗位 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"


async def _get_position_questions(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import get_position_questions as fn
    result = await fn(position_id=params["positionId"], round=params["round"], db=db)
    qs = result.get("data", [])
    if not qs:
        return f"岗位题库（{params['round']}）暂无题目"
    lines = [f"岗位题库（{params['round']}）共 {len(qs)} 道题："]
    for q in qs:
        lines.append(f"  [{q.get('index','?')}] {q.get('content','')} | 分类:{q.get('category','')} | 难度:{q.get('difficulty','')}")
    return "\n".join(lines)


async def _save_position_questions(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.positions import save_position_questions as fn
    result = await fn(position_id=params["positionId"], body={"round": params["round"], "questions": params["questions"]}, db=db)
    return f"已保存岗位题库 {len(params['questions'])} 道题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("list_positions", _list_positions)
    registry.register("get_position", _get_position)
    registry.register("create_position", _create_position)
    registry.register("update_position", _update_position)
    registry.register("delete_position", _delete_position)
    registry.register("get_position_questions", _get_position_questions)
    registry.register("save_position_questions", _save_position_questions)
