from datetime import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.core.security import require_permission, get_current_user, CurrentUser
from app.models.settings import SystemSetting, AuditLog
from app.services.system.system_settings import (
    DEFAULT_SETTINGS,
    get_system_settings,
    invalidate_cache,
)
from app.utils.clock import iso_utc
from app.utils.responses import ok

router = APIRouter(tags=["系统设置"])


@router.get("/settings")
async def get_settings(db: AsyncSession = Depends(get_db)):
    data = await get_system_settings(db)
    return ok(data)


@router.get("/settings/kb-engine")
async def get_kb_engine_info():
    from app.services.system.kb_settings import get_kb_engine_info

    return ok(get_kb_engine_info())


@router.put("/settings")
async def update_settings(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    setting = result.scalar_one_or_none()

    if setting:
        merged = {**(setting.value or {}), **body}
        setting.value = merged
    else:
        merged = {**DEFAULT_SETTINGS, **body}
        setting = SystemSetting(
            key="global",
            value=merged,
        )
        db.add(setting)

    await db.flush()
    invalidate_cache()
    return ok(merged)


@router.post("/settings/cleanup")
async def trigger_cleanup(db: AsyncSession = Depends(get_db)):
    """手动触发过期数据清理。"""
    from app.services.system.data_retention import cleanup_expired

    result = await cleanup_expired(db)
    return ok(result)


# ── Audit log ──────────────────────────────────────────

@router.get("/settings/audit-log")
async def list_audit_log(
    current: CurrentUser = Depends(require_permission("audit:view")),
    db: AsyncSession = Depends(get_db),
):
    """获取系统设置操作审计日志（最近 50 条，按时间倒序）。

    操作日志是安全审计资产，仅系统管理员(admin)可见；admin 因持 system:manage
    被通配放行，此依赖对 admin 不构成额外限制。
    """
    result = await db.execute(
        select(AuditLog).order_by(desc(AuditLog.created_at)).limit(50)
    )
    logs = result.scalars().all()
    return ok([
          {
              "id": log.id,
              "time": log.time,
              "actor": log.actor,
              "action": log.action,
              "section": log.section or "",
          }
          for log in logs
      ])


@router.post("/settings/audit-log")
async def append_audit_log(body: dict, db: AsyncSession = Depends(get_db)):
    """追加一条审计日志。body: { actor, action, section }"""
    log = AuditLog(
        id=f"log-{uuid.uuid4().hex[:12]}",
        time=body.get("time") or iso_utc(datetime.utcnow()),
        actor=body.get("actor", "系统"),
        action=body.get("action", ""),
        section=body.get("section"),
    )
    db.add(log)
    await db.flush()
    return ok({
          "id": log.id,
          "time": log.time,
          "actor": log.actor,
          "action": log.action,
          "section": log.section or "",
      })


# ── 公司信息 ────────────────────────────────────────────
# 复用 SystemSetting 表（key="company_info"），读=登录可见，写=仅系统管理员。

COMPANY_INFO_KEY = "company_info"

COMPANY_INFO_FIELDS = [
    "companyName", "shortName", "description",
    "creditCode", "legalPerson", "registeredCapital",
    "foundedAt", "industry", "headcount", "address",
    "phone", "email", "website", "logoUrl",
    "financingStage", "tags", "welfare",
    # 招聘扩展字段（2026-08-04）：雇主品牌与招聘联系
    "companyType", "workAddress", "recruitmentContact", "recruitmentEmail",
    "products", "culture", "honors",
    # 招聘版字段（2026-08-04）：经营状态 / 股权 / 财务概况 / 风险 / 资质
    "businessStatus", "controller", "insuredCount",
    "revenueScale", "profitable",
    "riskStatus", "riskNotes",
    "qualifications",
]


@router.get("/company-info")
async def get_company_info(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取公司信息（登录即可读）。未配置时返回空模板。"""
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == COMPANY_INFO_KEY)
    )
    setting = result.scalar_one_or_none()
    data = setting.value if setting and setting.value else {}
    return ok({k: data.get(k, "") for k in COMPANY_INFO_FIELDS})


@router.put("/company-info")
async def update_company_info(
    body: dict,
    current: CurrentUser = Depends(require_permission("system:manage")),
    db: AsyncSession = Depends(get_db),
):
    """更新公司信息（仅系统管理员；admin 持 system:manage 通配放行）。

    只接收白名单字段，忽略未知键；写操作落审计日志（section="company"）。
    """
    clean = {k: body.get(k) for k in COMPANY_INFO_FIELDS if k in body}
    if not clean:
        raise HTTPException(status_code=400, detail="没有可更新的公司信息字段")

    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == COMPANY_INFO_KEY)
    )
    setting = result.scalar_one_or_none()
    if setting:
        merged = {**(setting.value or {}), **clean}
        setting.value = merged
    else:
        setting = SystemSetting(key=COMPANY_INFO_KEY, value=clean)
        db.add(setting)

    # 审计日志（管理员本人操作，非"审计者≠被审计者"冲突场景）
    log = AuditLog(
        id=f"log-{uuid.uuid4().hex[:12]}",
        time=iso_utc(datetime.utcnow()),
        actor=current.username,
        action="更新公司信息",
        section="company",
    )
    db.add(log)

    await db.flush()
    return ok(clean)
