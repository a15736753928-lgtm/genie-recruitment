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
    # 第一期新增: 题目类型 + 8要素规格
    question_type = Column(String(16), nullable=False, default="standard")
    # standard(固定标准题) / verify(简历验证题) / practical(场景实操题)
    spec = Column(JSON, nullable=True)
    # {purpose,followUp,scoringDimensions,maxScore,excellentAnswer,acceptableAnswer,failAnswer,riskSignals}
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
    # AI 全方位评定报告（JSON）
    assessment_report = Column(JSON, nullable=True)
    # ── 异步处理进度（转写→抽问答→评分→片段→报告，后台任务回写，前端轮询）──
    # pending / transcribing / extracting / scoring / segment / report / completed / failed
    process_status = Column(String(16), nullable=False, default="completed")
    process_progress = Column(Integer, nullable=False, default=100)  # 0-100
    process_stage = Column(String(64), nullable=True)                # 人类可读阶段名
    process_message = Column(Text, nullable=True)                    # 错误详情或补充说明
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="transcripts")
    questions = relationship("InterviewQuestion", back_populates="transcript", cascade="all, delete-orphan")
    # 自我介绍 / 反问环节等片段评分，关联到具体转写记录，删除记录时级联删除
    segment_evaluations = relationship(
        "InterviewSegmentEvaluation", back_populates="transcript", cascade="all, delete-orphan"
    )


class InterviewSegmentEvaluation(Base):
    """面试「自我介绍」「反问环节」等非问答片段的评分。

    与 InterviewEvaluation（按题目评分）平行：一个 transcript 下，每种 segment_type
    至多一条。AI 评分在上传转写时自动生成，HR 评分由前端保存。
    """
    __tablename__ = "interview_segment_evaluations"
    __table_args__ = (
        UniqueConstraint("candidate_id", "round", "transcript_id", "segment_type",
                         name="uq_interview_segment_evals_cand_round_tid_type"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    round = Column(String(8), nullable=False)
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("interview_transcripts.id", ondelete="CASCADE"), nullable=True)
    # 片段类型：self_intro=自我介绍；reverse_question=反问环节
    segment_type = Column(String(16), nullable=False)
    # 从转写文本中抽取出的片段原文，供前端展示
    content = Column(Text)
    ai_score = Column(Integer)
    ai_dimensions = Column(JSON)
    hr_score = Column(Integer)
    hr_dimensions = Column(JSON)
    status = Column(String(16), default="pending")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="segment_evaluations")
    transcript = relationship("InterviewTranscript", back_populates="segment_evaluations")
