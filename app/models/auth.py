"""RBAC 鉴权模型 —— users / roles / user_roles / role_permissions。

第一期 P0 基础设施。角色编码采用原型口径:
ceo / hr / manager / interviewer / mentor / project_lead / employee / admin
(+ 预留 equity_committee 供第四期期权委员会)。
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, JSON, ForeignKey, PrimaryKeyConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    display_name = Column(String(64), nullable=False)
    email = Column(String(128), nullable=True)
    phone = Column(String(32), nullable=True)
    department = Column(String(64), nullable=True)
    # 绑定到 employees.id —— 第三期员工自助数据隔离用(current_user.employee_id)
    employee_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    status = Column(String(16), nullable=False, default="active")   # active/disabled/left
    must_change_password = Column(Boolean, nullable=False, default=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")


class Role(Base):
    __tablename__ = "roles"

    code = Column(String(32), primary_key=True)     # ceo/hr/manager/...
    name = Column(String(32), nullable=False)
    description = Column(String(128), nullable=True)

    permissions = relationship("RolePermission", back_populates="role", cascade="all, delete-orphan")


class UserRole(Base):
    __tablename__ = "user_roles"
    __table_args__ = (PrimaryKeyConstraint("user_id", "role_code"),)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role_code = Column(String(32), ForeignKey("roles.code", ondelete="CASCADE"), nullable=False)
    # 数据范围限定(部门/岗位),第一期可空
    scope = Column(JSON, nullable=True)

    user = relationship("User", back_populates="roles")


class RolePermission(Base):
    __tablename__ = "role_permissions"
    __table_args__ = (PrimaryKeyConstraint("role_code", "permission_key"),)

    role_code = Column(String(32), ForeignKey("roles.code", ondelete="CASCADE"), nullable=False)
    permission_key = Column(String(64), nullable=False)   # 形如 "recruitment_request:create"

    role = relationship("Role", back_populates="permissions")
