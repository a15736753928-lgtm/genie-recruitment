"""
Agent Memory — database-backed persistence for the Agent OS memory system.

The primary store is the filesystem (``memory/*.md``).  This table acts
as a durability backup and enables SQL queries for admin/cleanup.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), unique=True, nullable=False)
    description = Column(String(256))
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def memory_type(self) -> str:
        return (self.metadata_ or {}).get("type", "reference")

    @property
    def scope(self) -> str:
        return (self.metadata_ or {}).get("scope", "general")
