import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, JSON, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String(128))
    agent_id = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
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
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
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
