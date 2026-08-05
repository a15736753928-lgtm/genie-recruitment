"""第三期业务表 —— 工作任务 / 验收 / 积分 / 奖惩 / 申诉。

积分公式: actual = round(base × completion × quality × time + bonus − penalty)
任务等级: S 500-1000 / A 200-500 / B 80-200 / C 20-80 / D 5-20
重大扣分: ≥200 分(可配) → 双人审批
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, Numeric, JSON,
    ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


# ═══════════════════════════════════════════════
# 工作任务
# ═══════════════════════════════════════════════

class WorkTask(Base):
    """工作任务 —— 由 project_lead 创建,员工认领执行。"""
    __tablename__ = "work_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False)
    goal = Column(Text, nullable=True)
    owner_id = Column(UUID(as_uuid=True), nullable=False)          # FK users(employee)
    collaborator_ids = Column(JSON, nullable=True)                  # [user_id, ...]
    start_at = Column(DateTime, nullable=True)
    deadline = Column(DateTime, nullable=True)
    submitted_at = Column(DateTime, nullable=True)          # 员工提交验收时间(按时交付率)
    priority = Column(String(8), nullable=False, default="normal")  # low/normal/high/urgent
    difficulty = Column(String(8), nullable=True)                   # easy/medium/hard/expert
    level = Column(String(2), nullable=False, default="D")          # S/A/B/C/D
    base_points = Column(Integer, nullable=False, default=0)        # 基础积分
    acceptance_criteria = Column(Text, nullable=False)              # 验收标准(必填)
    acceptor_id = Column(UUID(as_uuid=True), nullable=False)        # 验收人 FK users
    deliverables = Column(Text, nullable=True)
    risk_note = Column(Text, nullable=True)
    department = Column(String(64), nullable=True)
    position_id = Column(UUID(as_uuid=True), nullable=True)
    status = Column(String(16), nullable=False, default="draft")
    # draft / pending / in_progress / pending_accept / passed / rework / closed
    rework_count = Column(Integer, default=0)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════
# 任务验收
# ═══════════════════════════════════════════════

class TaskAcceptance(Base):
    """任务验收记录 —— 每个任务一条,含系数和加减分。"""
    __tablename__ = "task_acceptances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("work_tasks.id", ondelete="CASCADE"),
                     nullable=False, unique=True, index=True)
    acceptor_id = Column(UUID(as_uuid=True), nullable=False)         # FK users

    completion_coeff = Column(Numeric(3, 2), nullable=False, default=1.0)   # 0-1.3
    quality_coeff = Column(Numeric(3, 2), nullable=False, default=1.0)      # 0-1.2
    timeliness_coeff = Column(Numeric(3, 2), nullable=False, default=1.0)   # 0-1.1

    collaboration_points = Column(Integer, default=0)
    innovation_points = Column(Integer, default=0)
    penalty_points = Column(Integer, default=0)

    notes = Column(Text, nullable=True)
    accepted_at = Column(DateTime, default=datetime.utcnow)


# ═══════════════════════════════════════════════
# 积分记录
# ═══════════════════════════════════════════════

class PointRecord(Base):
    """积分流水 —— 来源: 任务验收/奖励/扣分/手动。"""
    __tablename__ = "point_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), nullable=True)           # 关联任务(可为空)
    base_points = Column(Integer, nullable=False, default=0)
    actual_points = Column(Numeric(8, 2), nullable=False)        # 实际生效积分
    ai_advice = Column(JSON, nullable=True)                      # AIAdvice
    status = Column(String(16), nullable=False, default="pending")
    # pending / confirmed / appealed / cancelled
    source = Column(String(16), nullable=False, default="task")  # task / reward / penalty / manual
    description = Column(Text, nullable=True)
    confirmed_by = Column(UUID(as_uuid=True), nullable=True)      # FK users
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ═══════════════════════════════════════════════
# 奖惩记录
# ═══════════════════════════════════════════════

class RewardPenaltyRecord(Base):
    """奖励/扣分记录 —— 大额扣分(≥200)需双人审批。"""
    __tablename__ = "reward_penalty_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    type = Column(String(8), nullable=False)        # reward / penalty
    points = Column(Integer, nullable=False)         # 正数(数额)
    reason = Column(Text, nullable=False)
    rule_ref = Column(String(128), nullable=True)    # 依据条款
    evidence = Column(JSON, nullable=True)           # [{type, ref}],扣分必须非空

    # 双人审批
    requires_dual_approval = Column(Boolean, default=False)
    approver_ids = Column(JSON, nullable=True)       # [uuid, uuid] 两个不同审批人
    dual_approved = Column(Boolean, default=False)

    employee_notified = Column(Boolean, default=False)   # 员工知情确认
    employee_acknowledged = Column(Boolean, default=False)
    employee_ack_at = Column(DateTime, nullable=True)

    status = Column(String(16), nullable=False, default="pending")
    # pending / confirmed / dual_pending / cancelled

    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════
# 申诉
# ═══════════════════════════════════════════════

class Appeal(Base):
    """积分/扣分申诉 —— 员工提交,project_lead 处理。"""
    __tablename__ = "appeals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    target_type = Column(String(16), nullable=False)    # point / penalty
    target_id = Column(UUID(as_uuid=True), nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String(16), nullable=False, default="open")
    # open / reviewing / resolved / rejected
    handler_id = Column(UUID(as_uuid=True), nullable=True)  # FK users
    resolution = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
