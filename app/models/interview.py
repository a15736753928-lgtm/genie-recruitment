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
    __table_args__ = (UniqueConstraint("candidate_id", "round", "index_num"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    index_num = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(64))
    difficulty = Column(String(8))
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="questions")
    evaluations = relationship("InterviewEvaluation", back_populates="question", cascade="all, delete-orphan")


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
    __table_args__ = (UniqueConstraint("candidate_id", "round"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(16))
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="transcripts")
