"""第四期业务表 —— 人才画像 / 能力标签 / 晋级 / 期权 / 风险。

能力等级: L1(了解需指导)→L5(制定标准并带教),只升不降。
晋级六项条件全 True 方可审批(即使双人通过也拦截)。
期权: 长期贡献×30%、核心项目×25%、专业能力×15%、协作带教×15%、责任价值观×15%。
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, Numeric, JSON,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# ═══════════════════════════════════════════════
# 人才画像
# ═══════════════════════════════════════════════

class TalentProfile(Base):
    """每位正式员工一份画像,第二期转入 formal 时自动创建。"""
    __tablename__ = "talent_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True, index=True)
    current_position = Column(String(128), nullable=True)
    position_level = Column(String(16), nullable=True)          # 职级
    ability_level = Column(String(2), nullable=False, default="L1")  # L1-L5
    skills = Column(JSON, nullable=True)                        # [str]
    department = Column(String(64), nullable=True)

    # 指标(第三期聚合后刷新)
    task_success_rate = Column(Numeric(5, 2), nullable=True)    # 任务一次通过率%
    on_time_rate = Column(Numeric(5, 2), nullable=True)         # 按时率%
    first_pass_rate = Column(Numeric(5, 2), nullable=True)      # 首次通过率%
    total_confirmed_points = Column(Numeric(10, 2), nullable=True)
    task_count = Column(Integer, default=0)
    reward_count = Column(Integer, default=0)
    penalty_count = Column(Integer, default=0)
    rework_rate = Column(Numeric(5, 2), nullable=True)

    # 能力与协作
    trainable_skills = Column(JSON, nullable=True)              # [str]
    assignable_tasks = Column(JSON, nullable=True)              # [{level, count}]
    can_mentor = Column(Boolean, default=False)

    # 晋升与风险
    promotion_readiness = Column(Numeric(5, 2), nullable=True)  # 0-100
    talent_risk = Column(String(8), nullable=True, default="normal")  # normal/watch/high

    ai_analysis = Column(JSON, nullable=True)                   # AI 综合评估
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════
# 能力标签
# ═══════════════════════════════════════════════

class AbilityTag(Base):
    """每个员工的每项能力标签,evidence 必填非空。"""
    __tablename__ = "ability_tags"
    __table_args__ = (UniqueConstraint("employee_id", "tag", name="uq_ability_tag"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    tag = Column(String(64), nullable=False)
    level = Column(String(2), nullable=False, default="L1")    # L1-L5
    evidence = Column(JSON, nullable=False)                    # [{type,ref}] 非空
    confirmed_by = Column(UUID(as_uuid=True), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ═══════════════════════════════════════════════
# 晋级记录
# ═══════════════════════════════════════════════

class PromotionRecord(Base):
    """晋级申请 —— 六项条件全 True 方可审批。"""
    __tablename__ = "promotion_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    from_level = Column(String(2), nullable=False)             # L1
    to_level = Column(String(2), nullable=False)               # L2
    from_position = Column(String(128), nullable=True)
    to_position = Column(String(128), nullable=True)

    # 六项条件判定
    criteria_check = Column(JSON, nullable=False)              # {points_met:true, core_ability_met:true, stable_delivery:true, no_major_violation:true, higher_task_capable:true, review_passed:true}
    all_criteria_met = Column(Boolean, default=False)

    # AI 建议
    ai_recommendation = Column(JSON, nullable=True)            # AIAdvice

    # 双人审批
    review_result = Column(String(16), nullable=True)          # recommended / pending / approved / rejected
    approver_ids = Column(JSON, nullable=True)                  # [uuid, uuid]
    reject_reason = Column(Text, nullable=True)

    status = Column(String(16), nullable=False, default="pending")
    # pending / approved / rejected

    created_by = Column(UUID(as_uuid=True), nullable=False)    # 发起人
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════
# 期权记录
# ═══════════════════════════════════════════════

class EquityRecord(Base):
    """期权分配 —— 5 维加权分,CEO+期权委员会联签。"""
    __tablename__ = "equity_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, unique=True, index=True)

    # 五维原始分(0-100)
    long_term_points_score = Column(Numeric(5, 2), nullable=True)
    core_project_score = Column(Numeric(5, 2), nullable=True)
    professional_score = Column(Numeric(5, 2), nullable=True)
    collaboration_score = Column(Numeric(5, 2), nullable=True)
    responsibility_score = Column(Numeric(5, 2), nullable=True)

    # 加权综合分
    equity_score = Column(Numeric(5, 2), nullable=True)
    # = long_term × 0.30 + core_project × 0.25 + professional × 0.15
    #   + collaboration × 0.15 + responsibility × 0.15

    ai_recommendation = Column(JSON, nullable=True)            # AIAdvice

    # CEO + 委员会双人审批
    approver_ids = Column(JSON, nullable=True)                  # [uuid, uuid] CEO+committee
    reject_reason = Column(Text, nullable=True)

    status = Column(String(16), nullable=False, default="pending")
    # pending / approved / rejected

    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════
# 项目 & 项目人员分配(人员推荐组队)
# ═══════════════════════════════════════════════

class Project(Base):
    """业务项目需求 —— AI 按技能匹配人才，项目负责人确认组队。

    注意与 `app/models/agent_session.py::AgentProject` 区分：那个是 AI 对话工作台的
    会话文件夹，跟人员组队毫无关系。
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    required_skills = Column(JSON, nullable=True)               # [str]
    required_level = Column(String(2), nullable=True, default="L3")   # L1-L5
    headcount = Column(Integer, nullable=False, default=1)
    department = Column(String(64), nullable=True)
    status = Column(String(16), nullable=False, default="recruiting")  # recruiting / staffed / closed
    created_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    assignments = relationship("ProjectAssignment", back_populates="project",
                               cascade="all, delete-orphan")


class ProjectAssignment(Base):
    """项目人员分配 —— 一个员工在一个项目里最多一条。"""
    __tablename__ = "project_assignments"
    __table_args__ = (UniqueConstraint("project_id", "employee_id", name="uq_project_member"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    employee_name = Column(String(64), nullable=True)           # 冗余便于列表展示
    assigned_by = Column(String(64), nullable=True)             # 操作人名
    assigned_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="assignments")
