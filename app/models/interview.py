import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, Boolean, UniqueConstraint, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"
    # 唯一约束包含 source：面试出题(pre_generated)与面试评定转写抽取(transcript)
    # 两类题目各自独立编号，互不冲突。
    __table_args__ = (
        UniqueConstraint("candidate_id", "round", "source", "index_num",
                         name="uq_interview_questions_cand_round_source_idx"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    index_num = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(64))
    difficulty = Column(String(8))
    # 题目来源：pre_generated=面试出题环节 AI 生成；transcript=从上传的面试转写文本中抽取
    source = Column(String(16), nullable=False, default="pre_generated")
    # 当 source='transcript' 时，关联到具体的转写记录（删除记录时级联删除其题目与评分）
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("interview_transcripts.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="questions")
    evaluations = relationship("InterviewEvaluation", back_populates="question", cascade="all, delete-orphan")
    transcript = relationship("InterviewTranscript", back_populates="questions")


class InterviewEvaluation(Base):
    __tablename__ = "interview_evaluations"
    __table_args__ = (UniqueConstraint("candidate_id", "round", "question_id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id", ondelete="CASCADE"))
    answer = Column(Text)
    ai_score = Column(Integer)
    ai_dimensions = Column(JSON)
    hr_score = Column(Integer)
    hr_dimensions = Column(JSON)
    status = Column(String(16), default="pending")
    transcript = Column(Text)
    audio_uploaded = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="evaluations")
    question = relationship("InterviewQuestion", back_populates="evaluations")


class InterviewTranscript(Base):
    __tablename__ = "interview_transcripts"
    # 支持同一候选人同一轮多次上传历史记录，故不再加 (candidate_id, round) 唯一约束。

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(16))
    # 原始文件名，用于历史记录展示
    filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="transcripts")
    questions = relationship("InterviewQuestion", back_populates="transcript", cascade="all, delete-orphan")
