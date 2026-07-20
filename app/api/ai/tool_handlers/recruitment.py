"""Recruitment/resume tool handlers."""

from __future__ import annotations

import json
import os
import io
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure import minio_storage
from app.agent.field_profiles import (
    RESUME_FIELDS,
    resolve_fields,
    format_projected,
    parse_fields_param,
)


async def _list_resumes(params: dict, db: AsyncSession) -> str:
    from app.api.recruitment.resumes import list_resumes as fn
    pos_id = params.get("positionId", "all")
    statuses = params.get("statuses", "")
    keyword = params.get("keyword", "")
    limit = params.get("limit", 10)
    result = await fn(
        positionId=pos_id, statuses=statuses, keyword=keyword,
        sortBy="uploadTime", sortOrder="desc",
        page=1, pageSize=limit, minScore=None, db=db,
    )
    data = result["data"]
    if isinstance(data, dict) and "list" in data:
        candidates = data["list"]
        total = data["total"]
        if not candidates:
            return f"未找到符合条件的候选人（共 {total} 位候选人）"
        lines = [f"共 {total} 位候选人，以下是前 {len(candidates)} 位："]
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
    fields = params.get("fields", {})
    if "status" in params:
        fields["status"] = params["status"]
    result = await fn(resume_id=params["id"], body=fields, db=db)
    if result["code"] == 0:
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
