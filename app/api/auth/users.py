"""用户与角色管理 (sys_admin, system:manage) (§2.1.9)。"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.auth import User, Role, UserRole, RolePermission
from app.core.security import (
    hash_password, require_permission, get_current_user, CurrentUser,
)
from app.core.permissions import ROLES, ROLE_PERMISSIONS, PERMISSION_REGISTRY
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
import uuid

router = APIRouter(tags=["用户管理"], prefix="/users")


def _serialize_user(user: User, roles: list[str]) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "displayName": user.display_name,
        "email": user.email,
        "phone": user.phone,
        "department": user.department,
        "employeeId": str(user.employee_id) if user.employee_id else None,
        "status": user.status,
        "mustChangePassword": user.must_change_password,
        "roles": roles,
        "createdAt": user.created_at.isoformat() if user.created_at else None,
        "lastLoginAt": user.last_login_at.isoformat() if user.last_login_at else None,
    }


async def _get_user_roles(db: AsyncSession, user_id: uuid.UUID) -> list[str]:
    rows = await db.execute(select(UserRole.role_code).where(UserRole.user_id == user_id))
    return [r[0] for r in rows.all()]


@router.get("")
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    current: CurrentUser = Depends(require_permission("system:manage")),
    db: AsyncSession = Depends(get_db),
):
    q = select(User).where(User.is_deleted == False)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    users = (await db.execute(
        q.order_by(User.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()
    result = []
    for u in users:
        roles = await _get_user_roles(db, u.id)
        result.append(_serialize_user(u, roles))
    return ok({"list": result, "total": total, "page": page, "pageSize": page_size})


class CreateUserRequest(BaseModel):
    model_config = {"populate_by_name": True}
    username: str
    password: str
    display_name: str = Field(alias="displayName")
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    roles: list[str] = []


@router.post("")
async def create_user(
    body: CreateUserRequest,
    current: CurrentUser = Depends(require_permission("system:manage")),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none()
    if exists:
        return fail(409, f"用户名 {body.username!r} 已存在")
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        email=body.email,
        phone=body.phone,
        department=body.department,
        must_change_password=True,
    )
    db.add(user)
    await db.flush()
    for rc in body.roles:
        db.add(UserRole(user_id=user.id, role_code=rc))
    await write_audit(db, actor=current.username, action=f"创建用户 {body.username}", section="system")
    return ok(_serialize_user(user, body.roles))


class UpdateUserRequest(BaseModel):
    model_config = {"populate_by_name": True}
    display_name: Optional[str] = Field(None, alias="displayName")
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    status: Optional[str] = None        # active/disabled/left
    roles: Optional[list[str]] = None


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    current: CurrentUser = Depends(require_permission("system:manage")),
    db: AsyncSession = Depends(get_db),
):
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return not_found("用户不存在")
    row = await db.execute(select(User).where(User.id == uid, User.is_deleted == False))
    user = row.scalar_one_or_none()
    if user is None:
        return not_found("用户不存在")
    if body.display_name is not None:
        user.display_name = body.display_name
    if body.email is not None:
        user.email = body.email
    if body.phone is not None:
        user.phone = body.phone
    if body.department is not None:
        user.department = body.department
    if body.status is not None:
        user.status = body.status
    if body.roles is not None:
        # 重写角色
        await db.execute(
            UserRole.__table__.delete().where(UserRole.user_id == uid)  # type: ignore
        )
        for rc in body.roles:
            db.add(UserRole(user_id=uid, role_code=rc))
    await write_audit(db, actor=current.username, action=f"修改用户 {user.username}", section="system")
    roles = await _get_user_roles(db, uid)
    return ok(_serialize_user(user, roles))


class ResetPasswordRequest(BaseModel):
    model_config = {"populate_by_name": True}
    new_password: str = Field(alias="newPassword")


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: str,
    body: ResetPasswordRequest,
    current: CurrentUser = Depends(require_permission("system:manage")),
    db: AsyncSession = Depends(get_db),
):
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return not_found("用户不存在")
    row = await db.execute(select(User).where(User.id == uid, User.is_deleted == False))
    user = row.scalar_one_or_none()
    if user is None:
        return not_found("用户不存在")
    user.password_hash = hash_password(body.new_password)
    user.must_change_password = False
    await write_audit(db, actor=current.username, action=f"重置用户密码 {user.username}", section="system")
    return ok(None)


# ── /api/roles ──────────────────────────────────────────────

roles_router = APIRouter(tags=["角色"], prefix="/roles")


@roles_router.get("")
async def list_roles(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """返回全部角色及其权限点(供前端角色分配 UI 使用)。"""
    rows = (await db.execute(select(Role))).scalars().all()
    result = []
    for r in rows:
        perm_rows = (await db.execute(
            select(RolePermission.permission_key).where(RolePermission.role_code == r.code)
        )).scalars().all()
        result.append({
            "code": r.code,
            "name": r.name,
            "description": r.description,
            "permissions": list(perm_rows),
        })
    # 同时返回全部权限点注册表,供前端展示
    registry = [{"key": k, "label": v} for k, v in PERMISSION_REGISTRY.items()]
    return ok({"roles": result, "permissionRegistry": registry})
