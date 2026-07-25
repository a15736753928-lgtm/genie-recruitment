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
    """试用期任务 —— 第二期扩充为可验收任务(对齐前端 ProbationTask)。"""
    __tablename__ = "probation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(256), nullable=False)              # 前端 name
    week_number = Column(Integer, nullable=True)             # 前端 week
    description = Column(Text, nullable=True)
    # 状态统一到前端 TaskStatus: draft/pending_confirm/in_progress/pending_review/passed/rework/closed
    status = Column(String(24), default="in_progress")
    deadline = Column(Date, nullable=True)
    review_notes = Column(Text, nullable=True)

    # ── 第二期扩充:可验收任务字段(对齐前端 ProbationTask) ──
    objective = Column(Text, nullable=True)                  # 任务目标
    assignee = Column(String(64), nullable=True)             # 负责人/执行人
    input_materials = Column(Text, nullable=True)            # 输入材料
    deliverables = Column(Text, nullable=True)               # 交付物
    quality_standard = Column(Text, nullable=True)           # 质量标准
    test_standard = Column(Text, nullable=True)              # 测试标准
    reviewer = Column(String(64), nullable=True)             # 验收人
    expected_points = Column(Integer, nullable=True)         # 预期积分
    risk_note = Column(Text, nullable=True)                  # 风险说明
    score = Column(Numeric(5, 2), nullable=True)             # 验收得分
    project_scores = Column(JSON, nullable=True)             # 项目 8 维评分 {dimKey: score}
    submitted_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="tasks")
