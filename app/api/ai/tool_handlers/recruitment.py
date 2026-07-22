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


async def _update_resume(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import update_resume as fn

    STATUS_ALIASES = {
        "rejected": "failed",
        "淘汰": "failed",
        "未通过": "failed",
        "一面未通过": "failed",
        "二面未通过": "failed",
        "初筛不通过": "failed",
        "求职中": "job_hunting",
        "初筛通过": "passed",
        "一面中": "first_interview",
        "二面中": "second_interview",
        "已入职": "passed",
        "已通过": "passed",
        "入职": "passed",
    }

    fields = dict(params.get("fields") or {})
    if "status" in params:
        raw_status = str(params["status"]).strip()
        fields["status"] = STATUS_ALIASES.get(raw_status, raw_status)
    if "status" in fields:
        raw_status = str(fields["status"]).strip()
        fields["status"] = STATUS_ALIASES.get(raw_status, raw_status)

    result = await fn(resume_id=params["id"], body=fields, db=db)
    if result["code"] == 0:
        data = result.get("data") or {}
        name = data.get("name") or params["id"]
        new_status = fields.get("status") or data.get("status") or ""
        if new_status:
            msg = f"已成功更新候选人「{name}」(ID: {params['id']})，状态 → {new_status}"
            if new_status == "passed":
                msg += "（已进入试用期考核列表）"
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
