"""第二期业务表 —— 试用期计划/周评/转正 + 培训课程/进度 + 带教记录。

员工类型: tech / non_tech 走不同评分维度。
带教周期由 match_level 决定: high=4w / experienced=6w / fresh=8w / core=8w。
"""
import uuid
from datetime import datetime, date
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Date, ForeignKey, Boolean, Numeric, JSON, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# ═══════════════════════════════════════════════
# 试用期计划
# ═══════════════════════════════════════════════

class ProbationPlan(Base):
    """每个员工一份试用期计划(tech/non_tech)。"""
    __tablename__ = "probation_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True, index=True)
    type = Column(String(16), nullable=False)            # tech / non_tech
    total_weeks = Column(Integer, nullable=False, default=4)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    status = Column(String(16), nullable=False, default="active")   # active / completed / extended
    ai_generated = Column(Boolean, default=False)
    weeks = Column(JSON, nullable=True)                  # [ProbationWeekPlan{week,title,focus,goals[],trainingItems[]}]
    created_by_id = Column(UUID(as_uuid=True), nullable=True)       # FK users(id)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", foreign_keys=[employee_id],
                            back_populates="plan", uselist=False)


# ═══════════════════════════════════════════════
# 试用期周评
# ═══════════════════════════════════════════════

class ProbationWeekReview(Base):
    """每周考核评分(第1周固定维度，第2-4周项目维度)。"""
    __tablename__ = "probation_week_reviews"
    __table_args__ = (
        UniqueConstraint("employee_id", "week_number", name="uq_week_review"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    week_number = Column(Integer, nullable=False)   # 1-4
    dimensions = Column(JSON, nullable=False)       # {dimKey: score, ...}
    total_score = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    reviewer_id = Column(UUID(as_uuid=True), nullable=True)    # FK users(id)
    reviewer_name = Column(String(64), nullable=True)
    reviewed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", foreign_keys=[employee_id],
                            back_populates="week_reviews")


# ═══════════════════════════════════════════════
# 转正审批
# ═══════════════════════════════════════════════

class ConfirmationReview(Base):
    """转正综合考核(替代旧 ProbationConversion 表)。"""
    __tablename__ = "confirmation_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True, index=True)
    # 各周汇总分
    week1_score = Column(Numeric(5, 2), nullable=True)
    week2_score = Column(Numeric(5, 2), nullable=True)
    week3_score = Column(Numeric(5, 2), nullable=True)
    week4_score = Column(Numeric(5, 2), nullable=True)
    mentor_score = Column(Numeric(5, 2), nullable=True)
    discipline_score = Column(Numeric(5, 2), nullable=True, default=85)
    overall_score = Column(Numeric(5, 2), nullable=True)
    # 公式: week1×0.15 + week2×0.20 + week3×0.20 + week4×0.25 + mentor×0.10 + discipline×0.10

    ai_report = Column(JSON, nullable=True)              # AIAdvice
    employee_summary = Column(Text, nullable=True)        # 员工自评
    project_results = Column(JSON, nullable=True)         # [{name,result,score}]
    ability_gaps = Column(JSON, nullable=True)            # [{tag,gap}]
    next_90days_goals = Column(JSON, nullable=True)       # [{goal,measure}]

    recommendation = Column(String(16), nullable=True)    # excellent / normal / conditional / reject
    status = Column(String(16), nullable=False, default="pending")
    # pending / approved / rejected / conditional

    mentor_comment = Column(Text, nullable=True)
    manager_comment = Column(Text, nullable=True)

    manager_approved = Column(Boolean, default=False)
    manager_approved_at = Column(DateTime, nullable=True)
    # 转正审批: manager 单人审批(原型口径)

    # 人工覆盖 AI 建议结论时的留痕（合规要求：必须记录谁、何时、为什么改）
    override_reason = Column(Text, nullable=True)
    override_by = Column(UUID(as_uuid=True), nullable=True)
    override_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", foreign_keys=[employee_id],
                            back_populates="confirmation_review", uselist=False)


# ═══════════════════════════════════════════════
# 培训课程
# ═══════════════════════════════════════════════

class TrainingCourse(Base):
    """标准 5 天培训课程目录(静态种子数据)。"""
    __tablename__ = "training_courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    day = Column(Integer, nullable=False)               # 1-5
    title = Column(String(128), nullable=False)
    content = Column(Text, nullable=True)
    duration = Column(String(32), nullable=True)        # e.g. "2h" "半天"
    type = Column(String(16), nullable=False, default="lecture")   # lecture / practice / exam


class EmployeeTrainingProgress(Base):
    """每个员工的培训完成进度(入职时自动创建 5 天课程清单)。"""
    __tablename__ = "employee_training_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True, index=True)
    courses = Column(JSON, nullable=True)               # [{courseId, completed:bool, score:int?}]
    overall_rate = Column(Numeric(5, 2), default=0)     # 0-100
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("Employee", foreign_keys=[employee_id],
                            back_populates="training_progress", uselist=False)


# ═══════════════════════════════════════════════
# 带教记录
# ═══════════════════════════════════════════════

class MentorRecord(Base):
    """每周带教记录 —— 由带教人填写，员工确认。"""
    __tablename__ = "mentor_records"
    __table_args__ = (
        UniqueConstraint("employee_id", "week", name="uq_mentor_week"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    week = Column(Integer, nullable=False)
    training_content = Column(Text, nullable=True)      # 本期带教内容
    mastered_skills = Column(JSON, nullable=True)       # [str]
    pending_skills = Column(JSON, nullable=True)        # [str]
    completed_tasks = Column(JSON, nullable=True)       # [str]
    issues = Column(JSON, nullable=True)                # [str]
    improvement_plan = Column(Text, nullable=True)
    mentor_score = Column(Integer, nullable=True)       # 0-100
    employee_confirmed = Column(Boolean, default=False)
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), nullable=True)  # mentor user id

    employee = relationship("Employee", back_populates="mentor_records")
