import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
from app.database import get_db
from app.models.user import User
from app.config import get_settings

router = APIRouter(tags=["认证"])

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Schemas ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


class UserInfo(BaseModel):
    id: str
    username: str
    display_name: str
    email: Optional[str] = None
    role: str
    department: Optional[str] = None


# ── JWT Helpers ──────────────────────────────────────────

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


async def get_current_user(
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency: return authenticated user, or fall back to admin (no login required)."""
    # Try to authenticate from token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization[7:]
            payload = verify_token(token)
            user_id = payload.get("sub")
            if user_id:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                if user and user.is_active:
                    return user
        except Exception:
            pass

    # No valid token — return default admin user
    result = await db.execute(select(User).where(User.is_active == True).limit(1))
    user = result.scalar_one_or_none()
    if user:
        return user
    raise HTTPException(status_code=500, detail="系统未初始化，请先启动服务")


async def get_optional_user(
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Optional auth — returns None if not authenticated"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization[7:]
        payload = verify_token(token)
        user_id = payload.get("sub")
        if user_id:
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
    except Exception:
        pass
    return None


# ── Endpoints ────────────────────────────────────────────

@router.post("/auth/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()
    if not user or not pwd_context.verify(req.password, user.password_hash):
        return {"code": 401, "message": "用户名或密码错误", "data": None}
    if not user.is_active:
        return {"code": 403, "message": "账号已禁用", "data": None}

    token = create_access_token({"sub": str(user.id), "role": user.role})
    user_data = {
        "id": str(user.id),
        "username": user.username,
        "display_name": user.display_name,
        "email": user.email,
        "role": user.role,
        "department": user.department,
    }
    return {"code": 0, "message": "ok", "data": {"token": token, "user": user_data}}


@router.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    # Stateless JWT — client discards token
    return {"code": 0, "message": "ok", "data": None}


@router.get("/auth/me")
async def me(current_user: User = Depends(get_current_user)):
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(current_user.id),
            "username": current_user.username,
            "display_name": current_user.display_name,
            "email": current_user.email,
            "role": current_user.role,
            "department": current_user.department,
        },
    }
