import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Integer, Date, DateTime, ForeignKey, Text, Boolean, Numeric, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)
    name = Column(String(64), nullable=False)
    gender = Column(String(4))
    age = Column(Integer)
    department = Column(String(64))
    join_date = Column(Date)
    probation_end = Column(Date)
    status = Column(String(16), default="assessing")
    ai_score = Column(Integer)
    ai_result = Column(String(32))
    # Mentor
    mentor_name = Column(String(64), nullable=True)
    mentor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    # Week 1 assessment
    week1_score = Column(Integer, nullable=True)
    week1_passed = Column(Boolean, nullable=True)
    # Conversion
    conversion_score = Column(Numeric(5, 2), nullable=True)
    conversion_decision = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    position = relationship("Position", back_populates="employees")
    tasks = relationship("ProbationTask", back_populates="employee", cascade="all, delete-orphan")
    performance_records = relationship("PerformanceRecord", back_populates="employee", cascade="all, delete-orphan")
    week1_assessment = relationship("ProbationWeek1Assessment", back_populates="employee", uselist=False, cascade="all, delete-orphan")
    conversion = relationship("ProbationConversion", back_populates="employee", uselist=False, cascade="all, delete-orphan")


class ProbationTask(Base):
    __tablename__ = "probation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"))
    title = Column(String(256), nullable=False)
    week_number = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(16), default="pending")
    deadline = Column(Date, nullable=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="tasks")


class ProbationWeek1Assessment(Base):
    """Week 1 project reproduction assessment (试用期第一周考核)."""
    __tablename__ = "probation_week1_assessments"
    __table_args__ = ({"extend_existing": True},)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, unique=True)
    dimension_completion = Column(Integer, default=0)       # 项目复现完整度: max 30
    dimension_fidelity = Column(Integer, default=0)          # 代码/方案还原度: max 25
    dimension_problem_solving = Column(Integer, default=0)   # 独立解决问题能力: max 25
    dimension_standards = Column(Integer, default=0)         # 规范性与总结: max 20
    total_score = Column(Integer, default=0)                 # Sum of 4 dimensions, max 100
    deduction_reasons = Column(JSON, nullable=True)
    assessor_signature = Column(String(64), nullable=True)
    dept_head_signature = Column(String(64), nullable=True)
    assessor_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="week1_assessment")


class ProbationConversion(Base):
    """Final probation conversion evaluation (试用期转正考核)."""
    __tablename__ = "probation_conversions"
    __table_args__ = ({"extend_existing": True},)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, unique=True)
    # Dimension 1: Project performance (后三周真实项目表现) 60%
    project_performance_score = Column(Integer, default=0)
    project_performance_weight = Column(Numeric(3, 2), default=0.60)
    # Dimension 2: Tech capability (技术能力与业务产出) 20%
    tech_capability_score = Column(Integer, default=0)
    tech_capability_weight = Column(Numeric(3, 2), default=0.20)
    # Dimension 3: Collaboration (团队协作与综合素养) 20%
    collaboration_score = Column(Integer, default=0)
    collaboration_weight = Column(Numeric(3, 2), default=0.20)
    # Computed
    total_score = Column(Numeric(5, 2), default=0)
    decision = Column(String(16), nullable=True)  # converted / extended / rejected
    # Signatures
    mentor_comments = Column(Text, nullable=True)
    mentor_signature = Column(String(64), nullable=True)
    mentor_date = Column(Date, nullable=True)
    dept_head_signature = Column(String(64), nullable=True)
    dept_head_date = Column(Date, nullable=True)
    hr_signature = Column(String(64), nullable=True)
    hr_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="conversion")
