"""状态机流转日志 + 异常暂停队列 —— 第一期 P0 基础设施。"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class StateTransition(Base):
    """每一次实体状态变更都在此留痕(操作人/时间/from/to/原因/证据)。

    绝不允许直接 entity.status = x —— 一律经 core.state_machine.transition() 写入。
    """
    __tablename__ = "state_transitions"
    __table_args__ = (
        Index("ix_state_transitions_entity", "entity_type", "entity_id", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(32), nullable=False)   # candidate/employee/task/offer/recruitment_request
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    from_status = Column(String(32), nullable=True)
    to_status = Column(String(32), nullable=False)
    reason = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)             # [{type, ref}]
    actor_id = Column(UUID(as_uuid=True), nullable=True)
    actor_name = Column(String(64), nullable=False, default="系统")
    created_at = Column(DateTime, default=datetime.utcnow)


class ExceptionQueue(Base):
    """异常暂停队列 —— severity=block 时阻止该实体自动推进,须人工处理后放行。

    exception_type 为自由字符串(不做闭合枚举),便于二/三/四期追加类型:
    low_confidence / score_gap / ai_vs_human_conflict / media_missing /
    resume_contradiction / no_acceptance_criteria / deduction_no_evidence /
    appeal_submitted / model_uncertain ...
    """
    __tablename__ = "exceptions_queue"
    __table_args__ = (
        Index("ix_exceptions_entity", "entity_type", "entity_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(32), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    exception_type = Column(String(48), nullable=False)
    detail = Column(Text, nullable=True)
    severity = Column(String(8), nullable=False, default="warn")   # warn/block
    status = Column(String(8), nullable=False, default="open")     # open/handled
    handler_id = Column(UUID(as_uuid=True), nullable=True)
    handler_name = Column(String(64), nullable=True)
    resolution = Column(Text, nullable=True)
    handled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
