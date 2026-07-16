import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class KnowledgeCategory(Base):
    __tablename__ = "knowledge_categories"

    key = Column(String(64), primary_key=True)
    title = Column(String(128), nullable=False)
    parent_key = Column(String(64), ForeignKey("knowledge_categories.key"))
    sort_order = Column(Integer, default=0)

    items = relationship("KnowledgeItem", back_populates="category")


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False)
    category_key = Column(String(64), ForeignKey("knowledge_categories.key"))
    category_path = Column(String(256))
    type = Column(String(16))
    file_path = Column(String(512))
    content = Column(Text)
    recall_count = Column(Integer, default=0)
    milvus_ids = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    category = relationship("KnowledgeCategory", back_populates="items")
