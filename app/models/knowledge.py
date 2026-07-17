import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, JSON, BigInteger, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# ── Legacy models (kept for backward compatibility) ─────

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


# ── New RAG Knowledge Base models ──────────────────────

def _now_ms() -> int:
    """Millisecond Unix timestamp (consistent with blueprint)."""
    return int(datetime.utcnow().timestamp() * 1000)


def _short_uuid(prefix: str) -> str:
    """Generate a short ID like kb_a1b2c3d4."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class KnowledgeBase(Base):
    """Knowledge base group — maps to frontend KbGroup."""
    __tablename__ = "knowledge_bases"

    id = Column(String(128), primary_key=True)  # kb_{uuid8}
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, default="")
    owner_id = Column(String(36), nullable=True, default=None)  # UUID of the creating user
    doc_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    created_at = Column(BigInteger, nullable=False, default=_now_ms)
    updated_at = Column(BigInteger, nullable=False, default=_now_ms)

    documents = relationship("KnowledgeDocument", back_populates="kb", cascade="all, delete-orphan")
    chunks = relationship("KnowledgeChunk", back_populates="kb", cascade="all, delete-orphan")


class KnowledgeDocument(Base):
    """Document within a knowledge base — maps to frontend KbDocument."""
    __tablename__ = "knowledge_documents"

    id = Column(String(128), primary_key=True)  # doc_{uuid8}
    kb_id = Column(String(128), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    file_type = Column(String(20), default="")   # pdf/docx/txt/md/png/jpg/jpeg
    chunk_count = Column(Integer, default=0)
    object_key = Column(String(512), default="")  # local file path
    file_hash = Column(String(64), default="")    # SHA-256 for dedup
    status = Column(String(20), default="pending")  # pending/parsing/encoding/indexing/completed/failed
    uploaded_at = Column(BigInteger, default=0)     # ms timestamp
    created_at = Column(BigInteger, nullable=False, default=_now_ms)

    kb = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_docs_kb_id", "kb_id"),
        Index("idx_docs_status", "status"),
    )


class KnowledgeChunk(Base):
    """Text chunk with vector linkage to Milvus."""
    __tablename__ = "knowledge_chunks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    doc_id = Column(String(128), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(String(128), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)    # 0-based order
    chunk_text = Column(Text, nullable=False)
    milvus_pk = Column(BigInteger, default=0)        # Milvus auto-increment ID
    created_at = Column(BigInteger, nullable=False, default=_now_ms)

    document = relationship("KnowledgeDocument", back_populates="chunks")
    kb = relationship("KnowledgeBase", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_doc_id", "doc_id"),
    )


class IngestionTask(Base):
    """Async ingestion task for progress tracking."""
    __tablename__ = "ingestion_tasks"

    id = Column(String(128), primary_key=True)  # task_{uuid8}
    doc_id = Column(String(128), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(String(128), nullable=False)
    file_name = Column(String(500), default="")
    file_size = Column(Integer, default=0)
    status = Column(String(20), default="waiting")  # waiting/parsing/encoding/indexing/completed/failed
    progress = Column(Integer, default=0)   # 0-100
    message = Column(String(500), default="")
    created_at = Column(BigInteger, nullable=False, default=_now_ms)
    updated_at = Column(BigInteger, nullable=False, default=_now_ms)


class GraphCommunity(Base):
    """Louvain community detection results — maps to blueprint Section 8.4.

    Each community groups related entities/chunks discovered by
    Louvain community detection on the Kuzu knowledge graph,
    with an LLM-generated summary (150-250 chars).
    """
    __tablename__ = "graph_communities"

    id = Column(String(128), primary_key=True)  # com_{uuid8}
    kb_id = Column(String(128), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), default="")
    summary = Column(Text, nullable=False, default="")  # LLM-generated summary
    entity_ids = Column(Text, default="[]")   # JSON array of entity names
    chunk_ids = Column(Text, default="[]")    # JSON array of milvus_pk values
    created_at = Column(BigInteger, nullable=False, default=_now_ms)

    __table_args__ = (
        Index("idx_communities_kb_id", "kb_id"),
    )
