"""Offer 管理模块附属表 —— 模板 / 审批流 / 逐级审批记录 / 附件。

主表 OfferApproval 仍在 models/phase1.py（扩展字段见其注释）。
所有状态变更经 core.state_machine.transition() 驱动。
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, JSON,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


# ═══════════════════════════════════════════════
# Offer 模板
# ═══════════════════════════════════════════════

class OfferTemplate(Base):
    """Offer 模板 —— 固定文本 + 变量占位符，新建 Offer 时选择并填充。

    content 为 HTML 片段，占位符形如 {{name}} / {{position}} / {{baseSalary}}
    / {{onboardDate}}；variables 列出 content 实际用到的占位符名，供前端动态填充与预览。
    """
    __tablename__ = "offer_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), nullable=False)
    code = Column(String(64), unique=True, nullable=False)   # offer_letter 等
    category = Column(String(24), nullable=False, default="offer_letter")
    # offer_letter | salary_confirmation | supplemental_agreement
    content = Column(Text, nullable=False, default="")
    variables = Column(JSON, nullable=True)                  # list[str]
    is_default = Column(Boolean, nullable=False, default=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════
# 审批流配置
# ═══════════════════════════════════════════════

class OfferApprovalFlow(Base):
    """审批流配置 —— steps 为有序角色链快照。

    steps JSON: [{"step": 1, "role_code": "manager", "role_name": "部门负责人", "approver_id": null}, ...]
    approver_id 为空 = 该角色任一用户可批；非空 = 仅指定人可批。
    提交审批时把所选流 steps 快照到 offer_approval_records，改配置不影响已提交 Offer。
    """
    __tablename__ = "offer_approval_flows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(64), nullable=False)
    code = Column(String(64), unique=True, nullable=False)   # standard
    is_default = Column(Boolean, nullable=False, default=False)
    enabled = Column(Boolean, nullable=False, default=True)
    description = Column(Text, nullable=True)
    steps = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ═══════════════════════════════════════════════
# 逐级审批记录
# ═══════════════════════════════════════════════

class OfferApprovalRecord(Base):
    """审批流实例 —— 每级一条，提交时按所选流快照生成。"""
    __tablename__ = "offer_approval_records"
    __table_args__ = (
        UniqueConstraint("offer_id", "step", name="uq_offer_record_step"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    offer_id = Column(UUID(as_uuid=True), ForeignKey("offer_approvals.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    step = Column(Integer, nullable=False)                   # 1 基
    role_code = Column(String(32), nullable=False)           # manager / hr / ceo
    role_name = Column(String(32), nullable=True)
    approver_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approver_name = Column(String(64), nullable=True)
    action = Column(String(12), nullable=True)               # approve | reject
    opinion = Column(Text, nullable=True)
    decided_at = Column(DateTime, nullable=True)
    status = Column(String(12), nullable=False, default="pending")
    # pending | approved | rejected | cancelled
    created_at = Column(DateTime, default=datetime.utcnow)


# ═══════════════════════════════════════════════
# Offer 附件
# ═══════════════════════════════════════════════

class OfferAttachment(Base):
    """Offer 附件 —— 薪资确认单 / 附加协议等，文件存 MinIO（object_key）。"""
    __tablename__ = "offer_attachments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    offer_id = Column(UUID(as_uuid=True), ForeignKey("offer_approvals.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    category = Column(String(24), nullable=False, default="other")
    # salary_confirmation | supplemental_agreement | other
    filename = Column(String(255), nullable=False)
    object_key = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=True)
    size = Column(Integer, nullable=True, default=0)
    uploaded_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
