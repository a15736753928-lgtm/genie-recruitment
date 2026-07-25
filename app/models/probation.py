"""试用期/员工相关模型 —— 已按全生命周期重建(第一至四期通用)。

Employee 状态机(第一期建表,第二期驱动迁移):
  pending_onboard → training → probation → pending_confirmation → formal → transferred/resigned
"""
import uuid
from datetime import datetime, date
from sqlalchemy import (
    Column, String, Integer, Date, DateTime, ForeignKey, Text, Boolean, Numeric, JSON, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 来源招聘候选人
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=True, index=True)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

    # 基本信息
    name = Column(String(64), nullable=False)
    gender = Column(String(4), nullable=True)
    age = Column(Integer, nullable=True)
    department = Column(String(64), nullable=True)
    phone = Column(String(32), nullable=True)
    email = Column(String(128), nullable=True)

    # 生命周期状态(由 core.state_machine.transition 驱动)
    status = Column(String(32), nullable=False, default="pending_onboard")
    # pending_onboard / training / probation / pending_confirmation / formal / transferred / resigned

    # 入职与试用期
    onboard_date = Column(Date, nullable=True)
    probation_end_date = Column(Date, nullable=True)

    # 员工类型与匹配级别(第二期带教周数)
    employee_type = Column(String(16), nullable=True)   # tech / non_tech
    match_level = Column(String(16), nullable=True)     # high(4w)/experienced(6w)/fresh(8w)/core(8w)
    current_week = Column(Integer, nullable=True, default=0)
    total_weeks = Column(Integer, nullable=True, default=4)

    # 管理关系(名称+用户ID)
    mentor = Column(String(64), nullable=True)
    mentor_id = Column(UUID(as_uuid=True), nullable=True)   # FK users(id) — 第二期补 FK
    manager = Column(String(64), nullable=True)
    manager_id = Column(UUID(as_uuid=True), nullable=True)

    # 试用期评估汇总
    overall_score = Column(Numeric(5, 2), nullable=True)
    risk_level = Column(String(8), nullable=True)       # low / medium / high

    # 旧字段保留兼容(ai_score/ai_result 旧评分,第二期用 overall_score)
    ai_score = Column(Integer, nullable=True)
    ai_result = Column(String(32), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    position = relationship("Position", back_populates="employees")
    tasks = relationship("ProbationTask", back_populates="employee", cascade="all, delete-orphan")
    performance_records = relationship("PerformanceRecord", back_populates="employee", cascade="all, delete-orphan")
    # 旧评估关系(桩,一期内兼容)
    week1_assessment = relationship("ProbationWeek1Assessment", back_populates="employee",
                                    uselist=False, cascade="all, delete-orphan")
    conversion = relationship("ProbationConversion", back_populates="employee",
                              uselist=False, cascade="all, delete-orphan")
    # 第二期新增关系
    plan = relationship("ProbationPlan", back_populates="employee", uselist=False,
                        cascade="all, delete-orphan")
    week_reviews = relationship("ProbationWeekReview", back_populates="employee",
                                cascade="all, delete-orphan")
    confirmation_review = relationship("ConfirmationReview", back_populates="employee",
                                       uselist=False, cascade="all, delete-orphan")
    training_progress = relationship("EmployeeTrainingProgress", back_populates="employee",
                                     uselist=False, cascade="all, delete-orphan")
    mentor_records = relationship("MentorRecord", back_populates="employee",
                                  cascade="all, delete-orphan")


class ProbationTask(Base):
    """试用期任务 —— 第二期扩充,第一期建表保持兼容。"""
    __tablename__ = "probation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(256), nullable=False)
    week_number = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(16), default="pending")
    deadline = Column(Date, nullable=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="tasks")


# ── 旧评估表(保留桩类,供现有 probation.py 路由 import —— 第二期全面重建) ──

class ProbationWeek1Assessment(Base):
    """第一周考核(旧结构桩,第二期用 probation_week_reviews 替代)。"""
    __tablename__ = "probation_week1_assessments"
    __table_args__ = ({"extend_existing": True},)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True)
    dimension_completion = Column(Integer, default=0)
    dimension_fidelity = Column(Integer, default=0)
    dimension_problem_solving = Column(Integer, default=0)
    dimension_standards = Column(Integer, default=0)
    total_score = Column(Integer, default=0)
    deduction_reasons = Column(JSON, nullable=True)
    assessor_signature = Column(String(64), nullable=True)
    dept_head_signature = Column(String(64), nullable=True)
    assessor_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="week1_assessment")


class ProbationConversion(Base):
    """转正考核(旧结构桩,第二期用 confirmation_reviews 替代)。"""
    __tablename__ = "probation_conversions"
    __table_args__ = ({"extend_existing": True},)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True)
    project_performance_score = Column(Integer, default=0)
    project_performance_weight = Column(Numeric(3, 2), default=0.60)
    tech_capability_score = Column(Integer, default=0)
    tech_capability_weight = Column(Numeric(3, 2), default=0.20)
    collaboration_score = Column(Integer, default=0)
    collaboration_weight = Column(Numeric(3, 2), default=0.20)
    total_score = Column(Numeric(5, 2), default=0)
    decision = Column(String(16), nullable=True)
    mentor_comments = Column(Text, nullable=True)
    mentor_signature = Column(String(64), nullable=True)
    mentor_date = Column(Date, nullable=True)
    dept_head_signature = Column(String(64), nullable=True)
    dept_head_date = Column(Date, nullable=True)
    hr_signature = Column(String(64), nullable=True)
    hr_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="conversion")
