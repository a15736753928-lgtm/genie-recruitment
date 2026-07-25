"""第三期: 工作任务 / 验收 / 积分 / 奖惩 / 申诉 API。

积分公式: actual = round(base × completion × quality × time + bonus − penalty)
任务等级区间: S 500-1000 / A 200-500 / B 80-200 / C 20-80 / D 5-20
重大扣分阈值: system_settings.major_penalty_threshold (default 200)
"""
from __future__ import annotations
import json, uuid, math
from datetime import datetime
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.phase3 import WorkTask, TaskAcceptance, PointRecord, RewardPenaltyRecord, Appeal
from app.models.settings import SystemSetting
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.core.exceptions import push_exception
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit

router = APIRouter(tags=["任务积分(Phase3)"])

# ── 等级点范围 ──────────────────────────────────────────────
LEVEL_RANGES = {"S": (500, 1000), "A": (200, 500), "B": (80, 200), "C": (20, 80), "D": (5, 20)}

async def _get_setting(db: AsyncSession, key: str, default: Any) -> Any:
    row = await db.execute(select(SystemSetting).where(SystemSetting.key == key))
    s = row.scalar_one_or_none()
    return s.value if s else default

# ── 序列化 ───────────────────────────────────────────────────

def _serialize_task(t: WorkTask) -> dict:
    return {"id": str(t.id), "name": t.name, "goal": t.goal, "ownerId": str(t.owner_id) if t.owner_id else None,
            "collaboratorIds": t.collaborator_ids, "startAt": t.start_at.isoformat() if t.start_at else None,
            "deadline": t.deadline.isoformat() if t.deadline else None, "priority": t.priority,
            "difficulty": t.difficulty, "level": t.level, "basePoints": t.base_points,
            "acceptanceCriteria": t.acceptance_criteria, "acceptorId": str(t.acceptor_id) if t.acceptor_id else None,
            "deliverables": t.deliverables, "riskNote": t.risk_note, "department": t.department,
            "status": t.status, "reworkCount": t.rework_count, "createdBy": str(t.created_by) if t.created_by else None,
            "createdAt": t.created_at.isoformat() if t.created_at else None}

def _serialize_acceptance(a: TaskAcceptance) -> dict:
    return {"id": str(a.id), "taskId": str(a.task_id), "acceptorId": str(a.acceptor_id) if a.acceptor_id else None,
            "completionCoeff": float(a.completion_coeff), "qualityCoeff": float(a.quality_coeff),
            "timelinessCoeff": float(a.timeliness_coeff), "collaborationPoints": a.collaboration_points,
            "innovationPoints": a.innovation_points, "penaltyPoints": a.penalty_points,
            "notes": a.notes, "acceptedAt": a.accepted_at.isoformat() if a.accepted_at else None}

def _serialize_point(p: PointRecord) -> dict:
    return {"id": str(p.id), "employeeId": str(p.employee_id), "taskId": str(p.task_id) if p.task_id else None,
            "basePoints": p.base_points, "actualPoints": float(p.actual_points), "aiAdvice": p.ai_advice,
            "status": p.status, "source": p.source, "description": p.description,
            "confirmedBy": str(p.confirmed_by) if p.confirmed_by else None,
            "confirmedAt": p.confirmed_at.isoformat() if p.confirmed_at else None,
            "createdAt": p.created_at.isoformat() if p.created_at else None}

def _serialize_rp(rp: RewardPenaltyRecord) -> dict:
    return {"id": str(rp.id), "employeeId": str(rp.employee_id), "type": rp.type, "points": rp.points,
            "reason": rp.reason, "ruleRef": rp.rule_ref, "evidence": rp.evidence,
            "requiresDualApproval": rp.requires_dual_approval, "approverIds": rp.approver_ids,
            "dualApproved": rp.dual_approved, "employeeNotified": rp.employee_notified,
            "employeeAcknowledged": rp.employee_acknowledged, "status": rp.status,
            "createdBy": str(rp.created_by) if rp.created_by else None,
            "createdAt": rp.created_at.isoformat() if rp.created_at else None}

def _serialize_appeal(a: Appeal) -> dict:
    return {"id": str(a.id), "employeeId": str(a.employee_id), "targetType": a.target_type,
            "targetId": str(a.target_id), "content": a.content, "status": a.status,
            "handlerId": str(a.handler_id) if a.handler_id else None, "resolution": a.resolution,
            "resolvedAt": a.resolved_at.isoformat() if a.resolved_at else None,
            "createdAt": a.created_at.isoformat() if a.created_at else None}


# ═══════════════════════════════════════════════
# 工作任务
# ═══════════════════════════════════════════════

class CreateTaskRequest(BaseModel):
    model_config = {"populate_by_name": True}
    name: str
    goal: Optional[str] = None
    owner_id: str = Field(alias="ownerId")
    collaborator_ids: Optional[List[str]] = Field(None, alias="collaboratorIds")
    deadline: Optional[str] = None
    priority: str = "normal"
    difficulty: Optional[str] = None
    level: str = "D"
    base_points: int = Field(0, alias="basePoints")
    acceptance_criteria: str = Field(alias="acceptanceCriteria")
    acceptor_id: str = Field(alias="acceptorId")
    deliverables: Optional[str] = None
    risk_note: Optional[str] = Field(None, alias="riskNote")
    department: Optional[str] = None


@router.post("/tasks")
async def create_task(
    body: CreateTaskRequest,
    current: CurrentUser = Depends(require_permission("task:create")),
    db: AsyncSession = Depends(get_db),
):
    # 验收标准必填
    if not body.acceptance_criteria.strip():
        await push_exception(db, entity_type="task", entity_id=uuid.uuid4(),
                             exception_type="no_acceptance_criteria", detail="无验收标准不得发布", severity="block")
        return fail(422, "验收标准不可为空")

    # 等级→点数范围校验
    lo, hi = LEVEL_RANGES.get(body.level, (0, 0))
    if not (lo <= body.base_points <= hi):
        return fail(400, f"{body.level}级任务基础积分须在 {lo}-{hi} 之间，当前: {body.base_points}")

    # S级任务: 验收人不能是任务负责人
    if body.level == "S" and body.acceptor_id == body.owner_id:
        return fail(422, "S 级任务负责人与验收人不能为同一人")

    try:
        deadline_dt = datetime.fromisoformat(body.deadline.replace("Z", "+00:00")) if body.deadline else None
    except (ValueError, AttributeError):
        deadline_dt = None

    task = WorkTask(
        name=body.name, goal=body.goal,
        owner_id=uuid.UUID(body.owner_id),
        collaborator_ids=body.collaborator_ids or [],
        deadline=deadline_dt, priority=body.priority, difficulty=body.difficulty,
        level=body.level, base_points=body.base_points,
        acceptance_criteria=body.acceptance_criteria,
        acceptor_id=uuid.UUID(body.acceptor_id),
        deliverables=body.deliverables, risk_note=body.risk_note,
        department=body.department, created_by=current.id, status="draft",
    )
    db.add(task)
    await db.flush()
    await write_audit(db, actor=current.username, action=f"创建任务: {body.name}", section="tasks")
    return ok(_serialize_task(task))


class UpdateTaskRequest(BaseModel):
    model_config = {"populate_by_name": True}
    name: Optional[str] = None; goal: Optional[str] = None
    level: Optional[str] = None; base_points: Optional[int] = Field(None, alias="basePoints")
    acceptance_criteria: Optional[str] = Field(None, alias="acceptanceCriteria")
    deadline: Optional[str] = None; priority: Optional[str] = None
    deliverables: Optional[str] = None; risk_note: Optional[str] = Field(None, alias="riskNote")


@router.put("/tasks/{task_id}")
async def update_task(
    task_id: str, body: UpdateTaskRequest,
    current: CurrentUser = Depends(require_permission("task:create")),
    db: AsyncSession = Depends(get_db),
):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status in ("closed", "passed"):
        return fail(409, "已完成或已关闭的任务不可修改")
    for field, val in body.model_dump(exclude_unset=True, exclude={"base_points"}).items():
        if val is not None and hasattr(t, field):
            setattr(t, field, val)
    if body.base_points is not None:
        lo, hi = LEVEL_RANGES.get(t.level, (0, 0))
        if not (lo <= body.base_points <= hi):
            return fail(400, f"{t.level}级任务基础积分须在 {lo}-{hi} 之间")
        t.base_points = body.base_points
    await db.flush()
    return ok(_serialize_task(t))


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current: CurrentUser = Depends(require_permission("task:create")),
    db: AsyncSession = Depends(get_db),
):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status not in ("draft", "pending"):
        return fail(409, "只有草稿或待认领状态的任务可以删除")
    await db.delete(t)
    await db.flush()
    return ok(None)


@router.get("/tasks")
async def list_tasks(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    status: Optional[str] = None, level: Optional[str] = None,
    department: Optional[str] = None, owner_id: Optional[str] = Query(None, alias="ownerId"),
    current: CurrentUser = Depends(require_permission("task:view")),
    db: AsyncSession = Depends(get_db),
):
    # 员工数据隔离: employee 只看到自己
    if current.permissions and "self:points:view" in current.permissions and "task:accept" not in current.permissions:
        if current.employee_id:
            owner_id = str(current.employee_id)

    q = select(WorkTask)
    if status: q = q.where(WorkTask.status == status)
    if level: q = q.where(WorkTask.level == level)
    if department: q = q.where(WorkTask.department == department)
    if owner_id:
        try: q = q.where(WorkTask.owner_id == uuid.UUID(owner_id))
        except ValueError: pass
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(WorkTask.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_task(t) for t in rows], "total": total, "page": page, "pageSize": page_size})


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str, current: CurrentUser = Depends(require_permission("task:view")), db: AsyncSession = Depends(get_db),
):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    # 加载验收记录
    acc = (await db.execute(select(TaskAcceptance).where(TaskAcceptance.task_id == tid))).scalar_one_or_none()
    return ok({"task": _serialize_task(t), "acceptance": _serialize_acceptance(acc) if acc else None})


@router.post("/tasks/{task_id}/publish")
async def publish_task(task_id: str, current: CurrentUser = Depends(require_permission("task:create")), db: AsyncSession = Depends(get_db)):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status != "draft": return fail(409, "只有草稿可发布")
    try: await transition(db, "task", t, "pending", actor_id=current.id, actor_name=current.username, skip_block_check=True)
    except StateError as e: return fail(409, e.message)
    await write_audit(db, actor=current.username, action=f"发布任务: {t.name}", section="tasks")
    return ok(_serialize_task(t))


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: str, current: CurrentUser = Depends(require_permission("task:submit")), db: AsyncSession = Depends(get_db)):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status != "pending": return fail(409, "只有待认领的任务可开始")
    try: await transition(db, "task", t, "in_progress", actor_id=current.id, actor_name=current.username, skip_block_check=True)
    except StateError as e: return fail(409, e.message)
    return ok(_serialize_task(t))


@router.post("/tasks/{task_id}/submit")
async def submit_task(task_id: str, body: dict, current: CurrentUser = Depends(require_permission("task:submit")), db: AsyncSession = Depends(get_db)):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status != "in_progress": return fail(409, "只有进行中的任务可提交")
    t.deliverables = body.get("deliverables") or t.deliverables
    try: await transition(db, "task", t, "pending_accept", actor_id=current.id, actor_name=current.username, skip_block_check=True)
    except StateError as e: return fail(409, e.message)
    await write_audit(db, actor=current.username, action=f"提交任务: {t.name}", section="tasks")
    return ok(_serialize_task(t))


# ═══════════════════════════════════════════════
# 任务验收
# ═══════════════════════════════════════════════

class AcceptTaskRequest(BaseModel):
    model_config = {"populate_by_name": True}
    completion_coeff: float = Field(1.0, alias="completionCoeff")
    quality_coeff: float = Field(1.0, alias="qualityCoeff")
    timeliness_coeff: float = Field(1.0, alias="timelinessCoeff")
    collaboration_points: int = Field(0, alias="collaborationPoints")
    innovation_points: int = Field(0, alias="innovationPoints")
    penalty_points: int = Field(0, alias="penaltyPoints")
    notes: Optional[str] = None


@router.post("/tasks/{task_id}/acceptance")
async def accept_task(
    task_id: str, body: AcceptTaskRequest,
    current: CurrentUser = Depends(require_permission("task:accept")),
    db: AsyncSession = Depends(get_db),
):
    try: tid = uuid.UUID(task_id)
    except ValueError: return not_found("任务不存在")
    t = (await db.execute(select(WorkTask).where(WorkTask.id == tid))).scalar_one_or_none()
    if not t: return not_found("任务不存在")
    if t.status != "pending_accept": return fail(409, "只有待验收的任务可验收")

    # 系数范围校验
    if not (0.0 <= body.completion_coeff <= 1.3): return fail(400, "完成系数须在0-1.3之间")
    if not (0.0 <= body.quality_coeff <= 1.2): return fail(400, "质量系数须在0-1.2之间")
    if not (0.0 <= body.timeliness_coeff <= 1.1): return fail(400, "时效系数须在0-1.1之间")

    # 计算实际积分
    actual = round(
        t.base_points * body.completion_coeff * body.quality_coeff * body.timeliness_coeff
        + body.collaboration_points + body.innovation_points - body.penalty_points, 2
    )

    # 写验收记录
    acc = TaskAcceptance(
        task_id=tid, acceptor_id=current.id,
        completion_coeff=body.completion_coeff, quality_coeff=body.quality_coeff,
        timeliness_coeff=body.timeliness_coeff, collaboration_points=body.collaboration_points,
        innovation_points=body.innovation_points, penalty_points=body.penalty_points,
        notes=body.notes,
    )
    db.add(acc)

    # 创建待确认积分记录
    pr = PointRecord(
        employee_id=t.owner_id, task_id=tid,
        base_points=t.base_points, actual_points=actual,
        source="task", status="pending",
        description=f"任务: {t.name}",
    )
    db.add(pr)

    t.status = "passed"
    await db.flush()
    await write_audit(db, actor=current.username, action=f"验收任务: {t.name} 得分: {actual}", section="tasks")
    return ok({"task": _serialize_task(t), "acceptance": _serialize_acceptance(acc),
               "pointRecord": _serialize_point(pr)})


# ═══════════════════════════════════════════════
# 积分
# ═══════════════════════════════════════════════

@router.get("/points")
async def list_points(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    employee_id: Optional[str] = Query(None, alias="employeeId"),
    status: Optional[str] = None, source: Optional[str] = None,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 员工自助: 只能看自己
    if current.permissions and "self:points:view" in current.permissions and "points:confirm" not in current.permissions:
        if current.employee_id: employee_id = str(current.employee_id)
        else: return fail(403, "无法识别员工身份")

    q = select(PointRecord)
    if status: q = q.where(PointRecord.status == status)
    if source: q = q.where(PointRecord.source == source)
    if employee_id:
        try: q = q.where(PointRecord.employee_id == uuid.UUID(employee_id))
        except ValueError: pass
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(PointRecord.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_point(p) for p in rows], "total": total, "page": page, "pageSize": page_size})


@router.get("/employees/{employee_id}/points/summary")
async def points_summary(
    employee_id: str, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    try: eid = uuid.UUID(employee_id)
    except ValueError: return fail(400, "无效ID")
    confirmed = (await db.execute(
        select(func.sum(PointRecord.actual_points))
        .where(PointRecord.employee_id == eid, PointRecord.status == "confirmed")
    )).scalar() or 0
    pending = (await db.execute(
        select(func.sum(PointRecord.actual_points))
        .where(PointRecord.employee_id == eid, PointRecord.status == "pending")
    )).scalar() or 0
    return ok({"totalConfirmed": float(confirmed), "totalPending": float(pending), "employeeId": str(eid)})


@router.post("/points/{point_id}/confirm")
async def confirm_points(
    point_id: str, current: CurrentUser = Depends(require_permission("points:confirm")), db: AsyncSession = Depends(get_db),
):
    try: pid = uuid.UUID(point_id)
    except ValueError: return not_found("积分记录不存在")
    pr = (await db.execute(select(PointRecord).where(PointRecord.id == pid))).scalar_one_or_none()
    if not pr: return not_found("积分记录不存在")
    if pr.status == "appealed": return fail(409, "该记录正在申诉中，不可确认")
    pr.status = "confirmed"; pr.confirmed_by = current.id; pr.confirmed_at = datetime.utcnow()
    await db.flush()
    return ok(_serialize_point(pr))


@router.post("/points/{point_id}/cancel")
async def cancel_points(
    point_id: str, current: CurrentUser = Depends(require_permission("points:confirm")), db: AsyncSession = Depends(get_db),
):
    try: pid = uuid.UUID(point_id)
    except ValueError: return not_found("积分记录不存在")
    pr = (await db.execute(select(PointRecord).where(PointRecord.id == pid))).scalar_one_or_none()
    if not pr: return not_found("积分记录不存在")
    pr.status = "cancelled"
    await db.flush()
    return ok(_serialize_point(pr))


@router.post("/points/{point_id}/appeal")
async def appeal_points(
    point_id: str, body: dict, current: CurrentUser = Depends(require_permission("self:appeal")), db: AsyncSession = Depends(get_db),
):
    try: pid = uuid.UUID(point_id)
    except ValueError: return not_found("积分记录不存在")
    pr = (await db.execute(select(PointRecord).where(PointRecord.id == pid))).scalar_one_or_none()
    if not pr: return not_found("积分记录不存在")
    if pr.status != "confirmed": return fail(409, "只有已确认的积分可以申诉")
    pr.status = "appealed"
    # 创建申诉
    appeal = Appeal(employee_id=pr.employee_id, target_type="point", target_id=pid, content=body.get("content", ""))
    db.add(appeal)
    await push_exception(db, entity_type="point", entity_id=pid, exception_type="appeal_submitted", detail=body.get("content"), severity="warn")
    await db.flush()
    return ok({"point": _serialize_point(pr), "appeal": _serialize_appeal(appeal)})


# ═══════════════════════════════════════════════
# 奖惩
# ═══════════════════════════════════════════════

class CreateRPRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    type: str       # reward / penalty
    points: int     # 正数
    reason: str
    rule_ref: Optional[str] = Field(None, alias="ruleRef")
    evidence: Optional[List[dict]] = None

@router.post("/reward-penalties")
async def create_rp(
    body: CreateRPRequest, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    # 根据类型检查权限(两种都需检查)
    perm = "points:deduct" if body.type == "penalty" else "points:reward"
    if not current.has(perm): return fail(403, f"缺少权限: {perm}")

    if body.points <= 0: return fail(400, "分数必须为正整数")
    if not body.reason.strip(): return fail(400, "原因不可为空")

    # 扣分必须提供证据
    if body.type == "penalty" and (not body.evidence or len(body.evidence) == 0):
        await push_exception(db, entity_type="reward_penalty", entity_id=uuid.uuid4(),
                             exception_type="deduction_no_evidence", detail="扣分缺少证据", severity="block")
        return fail(422, "扣分须提供证据")

    threshold = await _get_setting(db, "major_penalty_threshold", 200)
    rp = RewardPenaltyRecord(
        employee_id=uuid.UUID(body.employee_id), type=body.type, points=body.points,
        reason=body.reason, rule_ref=body.rule_ref, evidence=body.evidence,
        requires_dual_approval=(body.type == "penalty" and body.points >= threshold),
        status="dual_pending" if (body.type == "penalty" and body.points >= threshold) else "confirmed",
        created_by=current.id,
    )
    db.add(rp)

    # 小额自动确认：同时创建积分记录
    if rp.status == "confirmed":
        sign = 1 if body.type == "reward" else -1
        db.add(PointRecord(employee_id=rp.employee_id, base_points=body.points, actual_points=body.points * sign,
                           status="confirmed", source=body.type, description=body.reason, confirmed_by=current.id, confirmed_at=datetime.utcnow()))
    await db.flush()
    action = f"{'奖励' if body.type == 'reward' else '扣分'}: {body.points}分"
    await write_audit(db, actor=current.username, action=action, section="points")
    return ok(_serialize_rp(rp))


@router.post("/reward-penalties/{rp_id}/employee-confirm")
async def employee_confirm_rp(
    rp_id: str, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    try: rid = uuid.UUID(rp_id)
    except ValueError: return not_found("记录不存在")
    rp = (await db.execute(select(RewardPenaltyRecord).where(RewardPenaltyRecord.id == rid))).scalar_one_or_none()
    if not rp: return not_found("记录不存在")
    rp.employee_acknowledged = True; rp.employee_notified = True; rp.employee_ack_at = datetime.utcnow()
    await db.flush()
    return ok(_serialize_rp(rp))


@router.post("/reward-penalties/{rp_id}/dual-approve")
async def dual_approve_rp(
    rp_id: str, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    try: rid = uuid.UUID(rp_id)
    except ValueError: return not_found("记录不存在")
    rp = (await db.execute(select(RewardPenaltyRecord).where(RewardPenaltyRecord.id == rid))).scalar_one_or_none()
    if not rp: return not_found("记录不存在")
    if not rp.requires_dual_approval: return fail(409, "该记录无需双人审批")
    if rp.dual_approved: return fail(409, "已双人审批")

    approvers: list = list(rp.approver_ids or [])
    if str(current.id) in approvers: return fail(409, "不能重复审批")
    approvers.append(str(current.id))
    rp.approver_ids = approvers

    if len(approvers) >= 2:
        rp.dual_approved = True; rp.status = "confirmed"; rp.confirmed_at = datetime.utcnow()
        # 双审通过 → 创建积分记录(扣分)
        sign = 1 if rp.type == "reward" else -1
        db.add(PointRecord(employee_id=rp.employee_id, base_points=rp.points, actual_points=rp.points * sign,
                           status="confirmed", source=rp.type, description=rp.reason, confirmed_by=current.id, confirmed_at=datetime.utcnow()))
    await db.flush()
    return ok(_serialize_rp(rp))


@router.get("/reward-penalties")
async def list_rp(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    type: Optional[str] = None, employee_id: Optional[str] = Query(None, alias="employeeId"),
    current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    q = select(RewardPenaltyRecord)
    if type: q = q.where(RewardPenaltyRecord.type == type)
    if employee_id:
        try: q = q.where(RewardPenaltyRecord.employee_id == uuid.UUID(employee_id))
        except ValueError: pass
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(RewardPenaltyRecord.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_rp(r) for r in rows], "total": total, "page": page, "pageSize": page_size})


# ═══════════════════════════════════════════════
# 申诉
# ═══════════════════════════════════════════════

@router.get("/appeals")
async def list_appeals(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    status: Optional[str] = None, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    q = select(Appeal)
    # 员工自助: 只能看自己
    if current.permissions and "self:appeal" in current.permissions and "points:confirm" not in current.permissions:
        if current.employee_id: q = q.where(Appeal.employee_id == current.employee_id)
    if status: q = q.where(Appeal.status == status)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(Appeal.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_appeal(a) for a in rows], "total": total, "page": page, "pageSize": page_size})


class HandleAppealRequest(BaseModel):
    model_config = {"populate_by_name": True}
    resolution: str
    action: str = "resolved"  # resolved / rejected


@router.put("/appeals/{appeal_id}/handle")
async def handle_appeal(
    appeal_id: str, body: HandleAppealRequest, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    try: aid = uuid.UUID(appeal_id)
    except ValueError: return not_found("申诉不存在")
    a = (await db.execute(select(Appeal).where(Appeal.id == aid))).scalar_one_or_none()
    if not a: return not_found("申诉不存在")
    if a.status in ("resolved", "rejected"): return fail(409, "申诉已处理")

    a.status = body.action; a.handler_id = current.id; a.resolution = body.resolution; a.resolved_at = datetime.utcnow()
    await db.flush()

    # 申诉通过 → 恢复原积分为 pending
    if body.action == "resolved" and a.target_type == "point":
        try:
            pr = (await db.execute(select(PointRecord).where(PointRecord.id == a.target_id))).scalar_one_or_none()
            if pr: pr.status = "pending"
        except (ValueError, Exception): pass

    await write_audit(db, actor=current.username, action=f"处理申诉: {a.target_type}", section="appeals")
    return ok(_serialize_appeal(a))


# ═══════════════════════════════════════════════
# 积分看板数据
# ═══════════════════════════════════════════════

@router.get("/dashboard/points")
async def points_dashboard(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # 月度排行 top 10
    top_rows = (await db.execute(
        select(PointRecord.employee_id, func.sum(PointRecord.actual_points).label("total"))
        .where(PointRecord.status == "confirmed")
        .group_by(PointRecord.employee_id).order_by(func.sum(PointRecord.actual_points).desc()).limit(10)
    )).all()
    ranking = [{"employeeId": str(r[0]), "totalPoints": float(r[1] or 0)} for r in top_rows]

    # 按类型汇总
    task_total = (await db.execute(select(func.sum(PointRecord.actual_points)).where(PointRecord.source == "task", PointRecord.status == "confirmed"))).scalar() or 0
    reward_total = (await db.execute(select(func.sum(PointRecord.actual_points)).where(PointRecord.source == "reward", PointRecord.status == "confirmed"))).scalar() or 0
    penalty_total = (await db.execute(select(func.sum(PointRecord.actual_points)).where(PointRecord.source == "penalty", PointRecord.status == "confirmed"))).scalar() or 0

    # 待确认/异常
    pending_count = (await db.execute(select(func.count()).select_from(PointRecord).where(PointRecord.status == "pending"))).scalar() or 0
    appealed_count = (await db.execute(select(func.count()).select_from(PointRecord).where(PointRecord.status == "appealed"))).scalar() or 0

    return ok({
        "ranking": ranking,
        "byType": {"task": float(task_total), "reward": float(reward_total), "penalty": float(penalty_total)},
        "pendingCount": pending_count, "appealedCount": appealed_count,
    })
