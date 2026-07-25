"""鉴权接口 —— 登录/登出/当前用户/刷新令牌。

白名单(无需令牌): POST /api/auth/login, GET /api/health。
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.auth import User, UserRole
from app.models.settings import AuditLog
from app.core.security import (
    verify_password, create_access_token, create_refresh_token, decode_token,
    get_current_user, CurrentUser, AuthError,
)
from app.core.permissions import permissions_for_roles, ROLES
from app.utils.responses import ok, fail
from app.utils.audit import write_audit

router = APIRouter(tags=["鉴权"], prefix="/auth")


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    model_config = {"populate_by_name": True}
    refresh_token: str = Field(alias="refreshToken")


async def _build_user_payload(db: AsyncSession, user: User) -> dict:
    role_rows = await db.execute(select(UserRole.role_code).where(UserRole.user_id == user.id))
    roles = [r[0] for r in role_rows.all()]
    perms = sorted(permissions_for_roles(roles))
    return {
        "id": str(user.id),
        "username": user.username,
        "displayName": user.display_name,
        "department": user.department,
        "employeeId": str(user.employee_id) if user.employee_id else None,
        "roles": roles,
        "roleLabels": [ROLES.get(r, (r, ""))[0] for r in roles],
        "permissions": perms,
        "mustChangePassword": user.must_change_password,
    }


@router.post("/login")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    row = await db.execute(select(User).where(User.username == body.username))
    user = row.scalar_one_or_none()
    # 不区分"用户不存在/密码错",避免账号枚举
    if user is None or user.is_deleted or not verify_password(body.password, user.password_hash):
        return fail(400, "用户名或密码错误")
    if user.status != "active":
        return fail(403, "账号已停用")

    user.last_login_at = datetime.utcnow()
    await write_audit(db, actor=user.username, action="登录", section="auth")

    payload = await _build_user_payload(db, user)
    token = create_access_token(payload["id"], payload["roles"], payload["permissions"])
    refresh = create_refresh_token(payload["id"])
    return ok({"token": token, "refreshToken": refresh, "user": payload})


@router.post("/logout")
async def logout(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # 无状态 JWT:后端仅记审计,前端清 token 即可
    await write_audit(db, actor=current.username, action="登出", section="auth")
    return ok(None)


@router.get("/me")
async def me(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    row = await db.execute(select(User).where(User.id == current.id))
    user = row.scalar_one_or_none()
    if user is None:
        raise AuthError("用户不存在")
    return ok(await _build_user_payload(db, user))


@router.post("/refresh")
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(body.refresh_token)   # 无效 -> AuthError -> 401
    if payload.get("type") != "refresh":
        raise AuthError("刷新令牌类型错误")
    import uuid as _uuid
    try:
        user_id = _uuid.UUID(str(payload.get("sub")))
    except (ValueError, TypeError):
        raise AuthError("刷新令牌主体无效")
    row = await db.execute(select(User).where(User.id == user_id))
    user = row.scalar_one_or_none()
    if user is None or user.is_deleted or user.status != "active":
        raise AuthError("账号不可用")
    up = await _build_user_payload(db, user)
    return ok({"token": create_access_token(up["id"], up["roles"], up["permissions"])})
