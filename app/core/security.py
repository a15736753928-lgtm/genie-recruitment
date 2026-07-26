"""鉴权核心 —— JWT 签发/校验、密码哈希、get_current_user、require_permission。

- 访问令牌 8h,刷新令牌 7d(可由 system_settings.jwt_ttl 覆盖,第一期用默认)。
- 密码 passlib bcrypt,绝不存明文。
- 权限集在登录时算好塞进 JWT claims(改权限需重登),per-request 不再查库。
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

from fastapi import Depends, Header
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.auth import User, UserRole
from app.core.permissions import permissions_for_roles, WILDCARD_PERMISSION

# ── 配置 ──
_SECRET = (
    os.getenv("JWT_SECRET_KEY")
    or os.getenv("SETTINGS_SECRET_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or "genie-recruitment-dev-secret-change-me"
)
_ALGO = "HS256"
ACCESS_TTL = timedelta(hours=8)
REFRESH_TTL = timedelta(days=7)

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── 密码 ──
def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False


# ── 令牌 ──
def create_access_token(user_id: str, roles: list[str], permissions: list[str]) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "roles": roles,
        "perms": permissions,
        "type": "access",
        "iat": now,
        "exp": now + ACCESS_TTL,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def create_refresh_token(user_id: str) -> str:
    now = datetime.utcnow()
    payload = {"sub": str(user_id), "type": "refresh", "iat": now, "exp": now + REFRESH_TTL}
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def decode_token(token: str) -> dict:
    """解码并校验 JWT;失败抛 AuthError。"""
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGO])
    except JWTError as e:
        raise AuthError(f"令牌无效或已过期: {e}")


# ── 异常(全局 handler 捕获 -> HTTP 401 信封) ──
from starlette.exceptions import HTTPException as StarletteHTTPException

class AuthError(StarletteHTTPException):
    """鉴权失败异常。继承 Starlette HTTPException，FastAPI 框架层原生拦截，
    不会被 ExceptionGroup 包装后泄漏到 uvicorn ERROR 日志。"""
    def __init__(self, message: str = "未登录或登录已过期"):
        self.message = message
        super().__init__(status_code=401, detail=message)


class PermissionError_(Exception):
    """无权限 —— 端点层捕获返回 code:403(HTTP 200)。"""
    def __init__(self, message: str = "无权限执行此操作"):
        self.message = message
        super().__init__(message)


# ── 当前用户主体 ──
class CurrentUser:
    def __init__(self, user: User, roles: list[str], permissions: set[str]):
        self.id = user.id
        self.username = user.username
        self.display_name = user.display_name
        self.department = user.department
        self.employee_id = user.employee_id
        self.roles = roles
        self.permissions = permissions

    def has(self, key: str) -> bool:
        return WILDCARD_PERMISSION in self.permissions or key in self.permissions


# ── 默认访客用户（开发阶段跳过认证校验）──
def _default_user() -> CurrentUser:
    """返回一个拥有全部权限的虚拟访客用户，用于开发阶段绕过登录。"""
    guest = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        username="guest",
        password_hash="",  # 虚拟用户，无实际密码
        display_name="访客(开发模式)",
        email="guest@local",
        department="",
        status="active",
        is_deleted=False,
        must_change_password=False,
    )
    all_roles = ["admin"]
    # WILDCARD_PERMISSION (system:manage) 在 require_permission 中放行一切
    all_perms = {WILDCARD_PERMISSION}
    return CurrentUser(guest, all_roles, all_perms)


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """从 Authorization: Bearer <jwt> 解析当前用户。

    缺失/无效令牌、账号非 active -> AuthError(全局 handler -> 401)。
    满足 §21.7:离职/停用即时失效(每请求查库校验 status)。

    GENIE_DEV_MODE 环境变量为 true 时，无 token 返回默认访客（开发用）。
    """
    # 无 token → 仅 GENIE_DEV_MODE 下返回默认访客
    if not authorization or not authorization.lower().startswith("bearer "):
        if not os.getenv("GENIE_DEV_MODE"):
            raise AuthError("未登录")
        return _default_user()
    token = authorization.split(" ", 1)[1].strip()

    # demo token（前端演示用）→ 仅 GENIE_DEV_MODE 下放行
    if token.startswith("demo-token-"):
        if not os.getenv("GENIE_DEV_MODE"):
            raise AuthError("未登录")
        return _default_user()

    payload = decode_token(token)
    if payload.get("type") != "access":
        raise AuthError("令牌类型错误")

    sub = payload.get("sub")
    try:
        user_id = uuid.UUID(str(sub))
    except (ValueError, TypeError):
        raise AuthError("令牌主体无效")

    row = await db.execute(select(User).where(User.id == user_id))
    user = row.scalar_one_or_none()
    if user is None or user.is_deleted:
        raise AuthError("用户不存在")
    if user.status != "active":
        raise AuthError("账号已停用")

    # 权限以库为准(即时反映角色变更);令牌里的 perms 仅作降级备用。
    role_rows = await db.execute(select(UserRole.role_code).where(UserRole.user_id == user_id))
    roles = [r[0] for r in role_rows.all()]
    perms = permissions_for_roles(roles)
    return CurrentUser(user, roles, perms)


def require_permission(*keys: str):
    """写接口权限守卫。缺任一权限 -> PermissionError_(端点层 -> code:403)。

    持 WILDCARD_PERMISSION(system:manage,即 admin)放行一切。
    """
    async def _dep(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if WILDCARD_PERMISSION in current.permissions:
            return current
        missing = [k for k in keys if k not in current.permissions]
        if missing:
            raise PermissionError_(f"无权限: 缺少 {', '.join(missing)}")
        return current

    return _dep
