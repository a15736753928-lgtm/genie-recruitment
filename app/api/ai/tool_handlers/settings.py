"""Settings tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.field_profiles import (
    SETTINGS_FIELDS,
    resolve_fields,
    format_projected,
    parse_fields_param,
)


async def _get_settings(params: dict, db: AsyncSession) -> str:
    from app.api.system.settings import get_settings as fn
    result = await fn(db=db)
    if result.get("code") != 0:
        return f"❌ 系统设置查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", {}) or {}
    view, fields, purpose = parse_fields_param(params)
    selected = resolve_fields(
        "settings", view=view, fields=fields, purpose=purpose, default_view="core",
    )
    if fields and ("*" in fields or "all" in [str(f).lower() for f in fields]):
        selected = list(data.keys())
    title = f"系统设置  [视图字段: {', '.join(selected)}]"
    labels = {**SETTINGS_FIELDS, **{k: k for k in data.keys() if k not in SETTINGS_FIELDS}}
    return format_projected(data, selected, labels, title=title, max_text=200)


async def _update_settings(params: dict, db: AsyncSession) -> str:
    from app.api.system.settings import update_settings as fn
    result = await fn(body=params.get("fields", {}), db=db)
    return f"已更新系统设置" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"


def register_handlers(registry) -> None:
    registry.register("get_settings", _get_settings)
    registry.register("update_settings", _update_settings)
