"""招聘需求 AI 生成产物 —— 7 项独立存储，供面试/试用期/培训等下游模块复用。

每个产物独立一行，支持版本迭代（重新生成时 version+1）。
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, JSON, ForeignKey, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# artifact_type 枚举
ARTIFACT_TYPES = (
    "job_description",       # 岗位说明书
    "competency_model",      # 岗位能力模型
    "resume_scoring_rules",  # 简历评分规则
    "interview_r1",          # 第一轮面试维度
    "interview_r2",          # 第二轮面试维度
    "probation_framework",   # 试用期考核框架
    "training_plan",         # 培训内容建议
)


class PositionAIArtifact(Base):
    """岗位 AI 生成产物 —— 每种类型一条记录，content 为 JSONB。"""
    __tablename__ = "position_ai_artifacts"
    __table_args__ = (
        Index("ix_artifact_req_type", "recruitment_request_id", "artifact_type"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recruitment_request_id = Column(
        UUID(as_uuid=True),
        ForeignKey("recruitment_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position_id = Column(
        UUID(as_uuid=True),
        ForeignKey("positions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    artifact_type = Column(String(32), nullable=False)
    content = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
