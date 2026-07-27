"""Recruitment/resume tool handlers."""

from __future__ import annotations

import json
import os
import io
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure import minio_storage
from app.agent.field_profiles import (
    RESUME_FIELDS,
    resolve_fields,
    format_projected,
    parse_fields_param,
)


async def _resolve_position_id(db: AsyncSession, position_id: str | None, position_name: str | None) -> str:
    """Resolve positionName to UUID; prefer explicit positionId."""
    if position_id and position_id != "all":
        return position_id
    if not position_name or not position_name.strip():
        return "all"
    from app.models.recruitment import Position

    name = position_name.strip()
    result = await db.execute(
        select(Position).where(Position.name.ilike(name)).limit(1)
    )
    pos = result.scalar_one_or_none()
    if pos:
        return str(pos.id)
    result = await db.execute(
        select(Position).where(Position.name.ilike(f"%{name}%")).limit(1)
    )
    pos = result.scalar_one_or_none()
    return str(pos.id) if pos else "all"


async def _list_resumes(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import query_candidate_list

    pos_id = await _resolve_position_id(
        db,
        params.get("positionId"),
        params.get("positionName"),
    )
    pos_name = (params.get("positionName") or "").strip()
    if pos_name and pos_id == "all":
        return f"未找到岗位「{pos_name}」，请先调用 list_positions 确认岗位名称。"
    statuses = params.get("statuses") or ""
    keyword = params.get("keyword") or ""
    limit = params.get("limit", 10)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    result = await query_candidate_list(
        db,
        position_id=pos_id,
        statuses=statuses,
        keyword=keyword,
        sort_by=params.get("sortBy") or "uploadTime",
        sort_order="desc",
        page=1,
        page_size=limit,
    )
    data = result["data"]
    if isinstance(data, dict) and "list" in data:
        candidates = data["list"]
        total = data["total"]
        if not candidates and not keyword and not statuses and (not pos_id or pos_id == "all"):
            return f"当前共有 {total} 位候选人。"
        scope = f"（岗位: {pos_name}）" if pos_name else ""
        lines = [f"共 {total} 位候选人{scope}，以下是前 {len(candidates)} 位："]
        for c in candidates:
            skills = ", ".join(c.get("skills", [])[:5]) or "无"
            lines.append(
                f"  [{c['id']}] {c['name']} | {c['position']} | "
                f"匹配度 {c['score']} 分 | 状态: {c['status']} | "
                f"技能: {skills}"
            )
        return "\n".join(lines)
    return f"查询结果：{json.dumps(data, ensure_ascii=False)[:500]}"


async def _get_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import get_resume as fn
    result = await fn(resume_id=params["id"], db=db)
    data = result.get("data")
    if not data:
        return "候选人不存在"
    view, fields, purpose = parse_fields_param(params)
    selected = resolve_fields(
        "resume", view=view, fields=fields, purpose=purpose, default_view="detail",
    )
    flat = dict(data)
    if "skills" in flat and isinstance(flat["skills"], list):
        flat["skills"] = ", ".join(flat["skills"]) or "无"
    title = f"候选人「{data.get('name', '')}」(ID: {data.get('id', '')})  [视图字段: {', '.join(selected)}]"
    return format_projected(flat, selected, RESUME_FIELDS, title=title, max_text=1200, max_json=900)


# 中文口语 → 状态机词表(state_machine.py TRANSITIONS["candidate"])的英文 key。
# 注意：不提供"已入职/入职"别名 —— hired 只能由 Offer 审批通过后自动生成
# （见 app/api/talent/offer.py），不允许对话侧直接把候选人改成 hired。
# 模块级导出：app/agent_os/quality/verifier.py 做写后回读值比对时要复用同一张表。
STATUS_ALIASES = {
    "淘汰": "rejected",
    "未通过": "rejected",
    "一面未通过": "rejected",
    "二面未通过": "rejected",
    "初筛不通过": "rejected",
    "求职中": "pending_screen",
    "待筛选": "pending_screen",
    "初筛通过": "invited",
    "一面中": "round1",
    "二面中": "round2",
    "待发offer": "pending_offer",
    "待发Offer": "pending_offer",
}


async def _update_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import update_resume as fn

    fields = dict(params.get("fields") or {})
    if "status" in params:
        raw_status = str(params["status"]).strip()
        fields["status"] = STATUS_ALIASES.get(raw_status, raw_status)
    if "status" in fields:
        raw_status = str(fields["status"]).strip()
        fields["status"] = STATUS_ALIASES.get(raw_status, raw_status)

    # hired 只能由「录用审批通过」自动生成(会同时创建 Employee 记录)，
    # 对话侧禁止直接把候选人改成 hired，否则会出现"候选人显示已录用但无员工档案"的脏数据。
    if fields.get("status") == "hired":
        return "无法直接将候选人标记为已录用：请通过「录用审批」流程操作，审批通过后会自动生成员工档案。"

    result = await fn(resume_id=params["id"], body=fields, db=db)
    if result["code"] == 0:
        data = result.get("data") or {}
        name = data.get("name") or params["id"]
        new_status = fields.get("status") or data.get("status") or ""
        if new_status:
            msg = f"已成功更新候选人「{name}」(ID: {params['id']})，状态 → {new_status}"
            if new_status == "pending_offer":
                msg += "（可发起录用审批）"
            return msg
        return f"已成功更新候选人 {params['id']} 的信息"
    return f"更新失败：{result['message']}"


async def _upload_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import upload_resume as fn
    from fastapi import UploadFile
    object_key = params["fileKey"]
    file_name = params["fileName"]
    position_id = params["positionId"]
    tmp_path = await asyncio.to_thread(minio_storage.download_to_temp, object_key)
    try:
        with open(tmp_path, "rb") as f:
            content = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    upload_file = UploadFile(file=io.BytesIO(content), filename=file_name)
    result = await fn(file=upload_file, positionId=position_id, db=db)
    if result["code"] == 0:
        d = result.get("data", {})
        return f"已上传简历「{d.get('name', file_name)}」，匹配度 {d.get('score', 0)} 分，状态 {d.get('status', '')}。{result.get('message', '')}"
    return f"上传失败：{result.get('message', '')}"


async def _batch_parse_resumes(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import batch_parse as fn
    result = await fn(body={"ids": params["ids"]}, db=db)
    return f"已批量解析 {len(params['ids'])} 位候选人" if result["code"] == 0 else f"批量解析失败：{result.get('message', '')}"


async def _reanalyze_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import reanalyze_resume as fn
    result = await fn(resume_id=params["id"], db=db)
    if result["code"] == 0:
        d = result.get("data", {})
        return f"已重新解析「{d.get('name', '')}」，新匹配度 {d.get('score', 0)} 分"
    return f"重新解析失败：{result.get('message', '')}"


async def _delete_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import delete_resume as fn
    result = await fn(resume_id=params["id"], db=db)
    return f"已删除候选人 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("list_resumes", _list_resumes)
    registry.register("get_resume", _get_resume)
    registry.register("update_resume", _update_resume)
    registry.register("upload_resume", _upload_resume)
    registry.register("batch_parse_resumes", _batch_parse_resumes)
    registry.register("reanalyze_resume", _reanalyze_resume)
    registry.register("delete_resume", _delete_resume)
