from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key = Column(String(64), primary_key=True)
    value = Column(JSONB, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """系统设置操作审计日志。"""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True)
    time = Column(String(32), nullable=False)
    actor = Column(String(64), nullable=False, default="系统")
    action = Column(String(255), nullable=False)
    section = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
