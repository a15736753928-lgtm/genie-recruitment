import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Date, DateTime, ForeignKey, Numeric, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class PerformanceQuarter(Base):
    __tablename__ = "performance_quarters"

    quarter = Column(String(8), primary_key=True)
    status = Column(String(16), default="draft")
    bonus_pool = Column(Numeric(14, 2))
    distributed = Column(Numeric(14, 2), default=0)
    initiated_at = Column(DateTime)


class PerformanceRecord(Base):
    __tablename__ = "performance_records"
    __table_args__ = (UniqueConstraint("employee_id", "quarter"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"))
    quarter = Column(String(8), nullable=False)
    tasks_completed = Column(Integer)
    quality = Column(Integer)
    speed = Column(Integer)
    compliance = Column(Integer)
    total_score = Column(Integer)
    grade = Column(String(4))
    bonus = Column(Numeric(12, 2))
    rank = Column(Integer)

    employee = relationship("Employee", back_populates="performance_records")
