import uuid
from datetime import datetime, date
from sqlalchemy import (
    Column, String, Integer, Text, Date, DateTime, ForeignKey, JSON, Boolean, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Position(Base):
    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), unique=True, nullable=False)
    department = Column(String(64), nullable=True)
    jd_content = Column(Text, nullable=True)
    jd_responsibilities = Column(Text, nullable=True)
    jd_requirements = Column(Text, nullable=True)
    jd_preferred = Column(Text, nullable=True)
    jd_tech_stack = Column(Text, nullable=True)
    # 任职要求结构化字段(下拉选择),与 jd_requirements 自由文本互补
    education_requirement = Column(String(32), nullable=True)
    experience_requirement = Column(String(32), nullable=True)
    age_requirement = Column(String(32), nullable=True)
    salary_range = Column(String(64), nullable=True)
    screening_criteria = Column(JSON, nullable=True)
    interview_criteria_r1 = Column(JSON, nullable=True)
    interview_criteria_r2 = Column(JSON, nullable=True)
    week1_project_requirement = Column(JSON, nullable=True)
    weeks_2_4_plan = Column(JSON, nullable=True)
    later_week_scoring = Column(JSON, nullable=True)
    conversion_criteria = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candidates = relationship("Candidate", back_populates="position")
    questions = relationship("PositionQuestion", back_populates="position", cascade="all, delete-orphan")
    employees = relationship("Employee", back_populates="position")


class PositionQuestion(Base):
    __tablename__ = "position_questions"
    __table_args__ = (UniqueConstraint("position_id", "round", "index_num"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id", ondelete="CASCADE"), nullable=False)
    round = Column(String(8), nullable=False)
    index_num = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(64), nullable=True)
    difficulty = Column(String(8), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    position = relationship("Position", back_populates="questions")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(64), nullable=False)
    gender = Column(String(4))
    age = Column(Integer)
    education = Column(String(32))
    experience = Column(String(32))
    ethnicity = Column(String(32))
    native_place = Column(String(64))
    phone = Column(String(32))
    email = Column(String(128))
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"))
    score = Column(Integer, default=0)
    status = Column(String(32), nullable=False, default="new")
    # 合法值与迁移见 app/core/state_machine.py TRANSITIONS["candidate"]:
    # new / parsed / pending_screen / pending_materials / invited /
    # round1 / round2 / pending_offer / hired / talent_pool / rejected
    # 一律通过 core.state_machine.transition() 变更，不允许直接赋值。
    resume_file = Column(String(512))
    resume_file_hash = Column(String(64), nullable=True, index=True)
    upload_time = Column(Date, default=date.today)
    # Screening dual-dimension fields
    screening_ai_score = Column(Integer, nullable=True)
    screening_manual_confirmed = Column(Boolean, default=False)
    screening_confirmed_by = Column(String(64), nullable=True)
    # Interview tracking
    interviewer = Column(String(64), nullable=True)
    interview_round = Column(String(8), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    position = relationship("Position", back_populates="candidates")
    skills = relationship("CandidateSkill", back_populates="candidate", cascade="all, delete-orphan")
    educations = relationship("CandidateEducation", back_populates="candidate", cascade="all, delete-orphan")
    work_experiences = relationship("CandidateWorkExperience", back_populates="candidate", cascade="all, delete-orphan")
    project_experiences = relationship("CandidateProjectExperience", back_populates="candidate", cascade="all, delete-orphan")
    ai_analysis = relationship("CandidateAIAnalysis", back_populates="candidate", uselist=False, cascade="all, delete-orphan")
    questions = relationship("InterviewQuestion", back_populates="candidate", cascade="all, delete-orphan")
    evaluations = relationship("InterviewEvaluation", back_populates="candidate", cascade="all, delete-orphan")
    transcripts = relationship("InterviewTranscript", back_populates="candidate", cascade="all, delete-orphan")
    segment_evaluations = relationship("InterviewSegmentEvaluation", back_populates="candidate", cascade="all, delete-orphan")


class CandidateSkill(Base):
    __tablename__ = "candidate_skills"
    __table_args__ = (UniqueConstraint("candidate_id", "skill"),)

    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"), primary_key=True)
    skill = Column(String(64), primary_key=True)

    candidate = relationship("Candidate", back_populates="skills")


class CandidateEducation(Base):
    __tablename__ = "candidate_educations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    school = Column(String(128))
    degree = Column(String(32))
    major = Column(String(128))
    period = Column(String(64))

    candidate = relationship("Candidate", back_populates="educations")


class CandidateWorkExperience(Base):
    __tablename__ = "candidate_work_experiences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    company = Column(String(128))
    role = Column(String(128))
    period = Column(String(64))
    description = Column(Text)

    candidate = relationship("Candidate", back_populates="work_experiences")


class CandidateProjectExperience(Base):
    __tablename__ = "candidate_project_experiences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"))
    name = Column(String(128))
    role = Column(String(128))
    period = Column(String(64))
    description = Column(Text)

    candidate = relationship("Candidate", back_populates="project_experiences")


class CandidateAIAnalysis(Base):
    __tablename__ = "candidate_ai_analyses"

    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"), primary_key=True)
    overall_score = Column(Integer)
    summary = Column(Text)
    position_match = Column(Text)
    experience_insight = Column(Text)
    recommendation = Column(Text)
    keywords = Column(JSON)
    highlights = Column(JSON)
    risks = Column(JSON)
    dimensions = Column(JSON)
    resume_extra = Column(JSON)  # 行业经验/管理经验/到岗时间/薪资/作品/证书
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="ai_analysis")


class Department(Base):
    """部门参考表 —— 用于招聘需求、岗位等统一引用。"""
    __tablename__ = "departments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

