"""第一期业务表 —— 招聘需求 / 岗位胜任力 / 简历评分 / 面试 / 录用审批。

所有写操作经 core.state_machine.transition() 驱动状态变更。
AI 输出存为 JSON 列,格式遵循 app.schemas.ai_advice.AIAdvice (snake_case)。
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, Numeric, JSON,
    ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# ═══════════════════════════════════════════════
# 模块一: 招聘需求
# ═══════════════════════════════════════════════

class RecruitmentRequest(Base):
    """招聘需求单 —— 从提交、AI 草稿、双确认到发布岗位的全流程。"""
    __tablename__ = "recruitment_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 18 个字段（2026-07-25 改造）
    position_name = Column(String(128), nullable=False)
    headcount = Column(Integer, nullable=False)                     # ≥1
    department_id = Column(UUID(as_uuid=True), ForeignKey("departments.id"), nullable=True)
    work_experience = Column(Text, nullable=False, default="")
    education_requirement = Column(Text, nullable=False, default="")
    job_responsibilities = Column(Text, nullable=False, default="")
    job_description = Column(Text, nullable=False, default="")
    job_requirements = Column(Text, nullable=False, default="")
    bonus_items = Column(Text, nullable=True, default="")
    core_tasks = Column(Text, nullable=False)
    required_skills = Column(JSON, nullable=False)                  # list[str], 非空
    preferred_skills = Column(JSON, nullable=True)
    deliverable_req = Column(Text, nullable=False)
    salary_range = Column(String(64), nullable=False)               # 脱敏字段
    probation_goal = Column(Text, nullable=False)
    elimination_criteria = Column(Text, nullable=False)
    interviewer_ids = Column(JSON, nullable=False)                  # [user_id 或姓名, ...]
    direct_manager_id = Column(UUID(as_uuid=True), nullable=True)   # 可选：系统用户 UUID
    direct_manager_name = Column(String(64), nullable=True)         # 表单填写的负责人姓名

    # 提交人
    submitter_id = Column(UUID(as_uuid=True), nullable=False)       # FK users

    # 状态(经 state_machine 驱动)
    status = Column(String(24), nullable=False, default="draft")
    # draft / ai_generated / hr_confirmed / dept_confirmed / published / closed

    # AI 生成的 7 件产品
    ai_draft = Column(JSON, nullable=True)

    # 双确认
    hr_confirmed_by = Column(UUID(as_uuid=True), nullable=True)
    hr_confirmed_at = Column(DateTime, nullable=True)
    dept_confirmed_by = Column(UUID(as_uuid=True), nullable=True)
    dept_confirmed_at = Column(DateTime, nullable=True)

    # 发布后回填
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PositionCompetency(Base):
    """岗位胜任力维度 —— 发布后锁定,6 个维度合计权重 100%。"""
    __tablename__ = "position_competency"
    __table_args__ = (
        UniqueConstraint("position_id", "dimension", name="uq_pos_competency"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    dimension = Column(String(32), nullable=False)
    weight = Column(Numeric(4, 2), nullable=False)   # 百分比,如 30.00
    locked = Column(Boolean, default=False)          # 发布后 True,不可修改


# ═══════════════════════════════════════════════
# 模块二: 简历评分
# ═══════════════════════════════════════════════

class ResumeScore(Base):
    """简历 8 维评分 + AI 建议 + ABCD 等级。"""
    __tablename__ = "resume_scores"
    __table_args__ = (
        UniqueConstraint("candidate_id", "position_id", name="uq_resume_score"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

    # 8 维分值(满分见括号)
    skill_match = Column(Integer, default=0)        # 必备技能匹配 max25
    project_match = Column(Integer, default=0)      # 项目经验匹配 max20
    position_exp = Column(Integer, default=0)       # 岗位经验匹配 max15
    achievement = Column(Integer, default=0)        # 成果证明 max15
    industry_exp = Column(Integer, default=0)       # 行业经验 max10
    learning = Column(Integer, default=0)           # 学习与成长 max5
    stability = Column(Integer, default=0)          # 稳定性 max5
    bonus_skill = Column(Integer, default=0)        # 加分技能 max5
    total = Column(Integer, default=0)              # 合计 0-100
    grade = Column(String(1), nullable=True)        # A/B/C/D

    advice = Column(JSON, nullable=True)            # AIAdvice (snake_case)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════
# 模块三: 面试
# ═══════════════════════════════════════════════

class Interview(Base):
    """面试场次 —— 每个候选人每轮一条记录。"""
    __tablename__ = "interviews"
    __table_args__ = (
        UniqueConstraint("candidate_id", "round", name="uq_interview_cand_round"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)
    round = Column(String(8), nullable=False)        # r1 / r2
    scheduled_at = Column(DateTime, nullable=True)
    interviewer_ids = Column(JSON, nullable=False)   # [uuid, ...]
    status = Column(String(16), nullable=False, default="scheduled")
    # scheduled / in_progress / completed / cancelled

    # R2 实操成果分 (D9)
    practical_score = Column(Integer, nullable=True)

    # 汇总(由结论接口写入)
    composite_score = Column(Numeric(5, 2), nullable=True)  # R1 or R2 综合分
    conclusion = Column(String(16), nullable=True)           # advance/review/reject/recommend

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    scores = relationship("InterviewerScore", back_populates="interview",
                          cascade="all, delete-orphan")
    ai_report = relationship("AIInterviewReport", back_populates="interview",
                             uselist=False, cascade="all, delete-orphan")


class InterviewerScore(Base):
    """面试官维度评分 —— 每位面试官提交一次。"""
    __tablename__ = "interviewer_scores"
    __table_args__ = (
        UniqueConstraint("interview_id", "interviewer_id", name="uq_interviewer_score"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interview_id = Column(UUID(as_uuid=True), ForeignKey("interviews.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    interviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    dimensions = Column(JSON, nullable=False)   # {dimensionKey: score, ...}
    total = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    is_submitted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    interview = relationship("Interview", back_populates="scores")


class AIInterviewReport(Base):
    """AI 面试分析报告 —— 基于转写内容生成,含 7 项布尔质量维度。"""
    __tablename__ = "ai_interview_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    interview_id = Column(UUID(as_uuid=True), ForeignKey("interviews.id", ondelete="CASCADE"),
                          nullable=True, index=True)
    round = Column(String(8), nullable=False)
    score = Column(Integer, nullable=True)              # AI 内容分析评分 0-100
    authenticity_score = Column(Integer, nullable=True) # 经历真实性专项 0-100
    advice = Column(JSON, nullable=False)               # AIAdvice (snake_case)

    # 7 项布尔质量维度(§5.5 dev doc 口径)
    answered_directly = Column(Boolean, nullable=True)
    role_clear = Column(Boolean, nullable=True)
    concrete_result = Column(Boolean, nullable=True)
    process_described = Column(Boolean, nullable=True)
    contradiction_found = Column(Boolean, nullable=True)
    avoided_key = Column(Boolean, nullable=True)
    logical = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    interview = relationship("Interview", back_populates="ai_report")


# ═══════════════════════════════════════════════
# 模块四: 录用审批
# ═══════════════════════════════════════════════

class OfferApproval(Base):
    """录用审批单 —— AI 建议 + 五分加权 + 经理审批。

    审批决策: 仅经理(manager)可批准,AI 只产建议。
    人工确认硬约束: offer/淘汰/入职均需此接口写入终态,AI 不直接写。
    """
    __tablename__ = "offer_approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

    # 五项分值(0-100,已标准化)
    resume_score = Column(Numeric(5, 2), nullable=True)    # 来自 resume_scores.total
    r1_score = Column(Numeric(5, 2), nullable=True)        # R1 综合分
    r2_score = Column(Numeric(5, 2), nullable=True)        # R2 综合分
    practical_score = Column(Numeric(5, 2), nullable=True) # D9: interviews.practical_score
    team_score = Column(Numeric(5, 2), nullable=True)      # D8: 部门负责人手填 0-100
    final_score = Column(Numeric(5, 2), nullable=True)     # 加权 = 0.15+0.25+0.40+0.10+0.10

    ai_advice = Column(JSON, nullable=True)                # AIAdvice (snake_case)

    # 9 项审批填写字段(§6.3 全部非空方可提交审批)
    strengths = Column(Text, nullable=True)
    capability_gaps = Column(Text, nullable=True)
    risks_note = Column(Text, nullable=True)
    suggested_salary = Column(String(64), nullable=True)   # 脱敏字段(salary:view)
    probation_goal = Column(Text, nullable=True)
    training_plan = Column(Text, nullable=True)
    mentor_id = Column(UUID(as_uuid=True), nullable=True)  # FK users
    conversion_criteria = Column(Text, nullable=True)
    elimination_criteria = Column(Text, nullable=True)

    # 审批人(经理一人审批,原型口径 managerOnly)
    approver_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    reject_reason = Column(Text, nullable=True)

    # AI 建议结果 vs 人工最终结果
    ai_result = Column(String(16), nullable=True)   # priority/recommend/conditional/reserve/reject
    result = Column(String(16), nullable=True)      # 人工最终决定

    status = Column(String(16), nullable=False, default="pending")
    # pending / approved / rejected

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
