import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Integer, Date, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    name = Column(String(64), nullable=False)
    gender = Column(String(4))
    age = Column(Integer)
    department = Column(String(64))
    join_date = Column(Date)
    probation_end = Column(Date)
    status = Column(String(16), default="assessing")
    ai_score = Column(Integer)
    ai_result = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)

    tasks = relationship("ProbationTask", back_populates="employee", cascade="all, delete-orphan")
    performance_records = relationship("PerformanceRecord", back_populates="employee", cascade="all, delete-orphan")


class ProbationTask(Base):
    __tablename__ = "probation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"))
    title = Column(String(256), nullable=False)
    status = Column(String(16), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="tasks")
