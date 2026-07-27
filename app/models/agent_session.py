import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, JSON, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class AgentProject(Base):
    """AI 对话工作台左栏的**会话文件夹**（用于把多个对话归到一组）。

    与 `app/models/phase4.py::Project` 完全无关——那个是「项目人员推荐」里的真实业务项目
    （有技能要求、能力等级、人员分配）。两者同名易混，凡涉及本类一律称「会话文件夹」，
    涉及 phase4.Project 一律称「业务项目」。
    """
    __tablename__ = "agent_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(64), nullable=False)
    # 会话文件夹归属用户——对话隔离用。admin 凭 system:manage 通配可见全部。
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("AgentSession", back_populates="project")


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(128))
    agent_id = Column(String(32))
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    # 会话归属用户——对话隔离的核心字段。子表(messages/materials/tasks)
    # 不直接挂 owner，经 session_id 间接归属，查询/删除前先校验 session 所有权。
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("AgentProject", back_populates="sessions")
    messages = relationship("AgentMessage", back_populates="session", cascade="all, delete-orphan")
    tasks = relationship("AgentTask", back_populates="session", cascade="all, delete-orphan")
    materials = relationship("AgentMaterial", back_populates="session", cascade="all, delete-orphan")


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'tool')"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("agent_sessions.id", ondelete="CASCADE"))
    role = Column(String(16), nullable=False)
    content = Column(Text)
    thinking = Column(Text)
    tool_blocks = Column(JSON)
    handoffs = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("AgentSession", back_populates="messages")


class AgentMaterial(Base):
    __tablename__ = "agent_materials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"))
    name = Column(String(256))
    type = Column(String(16))
    knowledge_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_items.id"))
    file_path = Column(String(512))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("AgentSession", back_populates="materials")


class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"))
    title = Column(String(256))
    description = Column(Text)
    progress = Column(Integer, default=0)
    status = Column(String(16))
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

    session = relationship("AgentSession", back_populates="tasks")
