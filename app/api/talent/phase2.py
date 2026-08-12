"""第二期: 试用期计划/任务/周评/转正 + 培训 + 带教 API。

转正公式: week1×0.15 + week2×0.20 + week3×0.20 + week4×0.25 + mentor×0.10 + discipline×0.10
阈值: ≥85 excellent / 75-84 normal / 65-74 conditional / <65 reject
带教周期: high=4w / experienced=6w / fresh=8w / core=8w
"""
from __future__ import annotations

from app.prompts import render_prompt
import json
import logging
import uuid
import math
from datetime import datetime, date, timedelta
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.probation import Employee
from app.models.phase2 import (
    ProbationPlan, ProbationWeekReview, ConfirmationReview,
    TrainingCourse, EmployeeTrainingProgress, MentorRecord,
)
from app.models.settings import SystemSetting
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.core.exceptions import push_exception
from app.utils.responses import ok, fail, not_found, conflict
from app.utils.audit import write_audit
from app.utils.clock import iso_utc
from app.utils.llm_json import extract_json_array

logger = logging.getLogger(__name__)

router = APIRouter(tags=["试用期&培训(Phase2)"])

# ── 常量 ─────────────────────────────────────────────────────
#
# 这里曾有 TECH_W1 / NONTECH_W1 / W2_4_DIMS_TECH / W2_4_DIMS_NONTECH 四组周评维度定义，
# 但全仓从无任何代码读取它们（save_week_review 只对 dimensions 求和校验 0-100，不校验键名），
# 而前端 constants/businessRules.ts 另有一套键名不同的维度在真正驱动周评表单——
# 两套定义长期并存且互不知情。已删除这四组死常量，避免后来者误以为后端在做维度校验。
#
# 周评维度的真源：前端 constants/businessRules.ts（PROBATION_WEEK1_TECH / PROBATION_NON_TECH）。
# ProbationWeekReview.dimensions 是自由 JSON 列，按岗位可配置的那套评分标准另存在
# positions 表（见 services/recruitment/seed_data_v2.py 的 WEEK1_COMMON），属岗位级配置数据。
#
# 面试维度不同——那套有后端强校验，真源在 phase1_interview.py 的 R1_DIMS/R2_DIMS，
# 前端通过 GET /api/interviews/dimensions 取用。


from app.services.system.system_settings import get_system_setting as _get_setting


# ── 序列化 ───────────────────────────────────────────────────

def _serialize_plan(p: ProbationPlan) -> dict:
    # employee 关系已 selectinload 时带上员工名/岗位
    emp_name = emp_position = ""
    try:
        if p.employee is not None:
            emp_name = p.employee.name or ""
            if p.employee.position is not None:
                emp_position = p.employee.position.name or ""
    except Exception:
        pass
    return {"id": str(p.id), "employeeId": str(p.employee_id),
            "employeeName": emp_name, "position": emp_position, "type": p.type,
            "totalWeeks": p.total_weeks, "startDate": iso_utc(p.start_date),
            "endDate": iso_utc(p.end_date), "status": p.status,
            "aiGenerated": p.ai_generated, "weeks": p.weeks,
            "createdById": str(p.created_by_id) if p.created_by_id else None,
            "createdAt": iso_utc(p.created_at)}

def _serialize_review(r: ProbationWeekReview) -> dict:
    return {"id": str(r.id), "employeeId": str(r.employee_id), "weekNumber": r.week_number,
            "dimensions": r.dimensions, "totalScore": r.total_score, "comment": r.comment,
            "reviewerId": str(r.reviewer_id) if r.reviewer_id else None, "reviewerName": r.reviewer_name,
            "reviewedAt": iso_utc(r.reviewed_at)}

def _serialize_confirmation(cr: ConfirmationReview) -> dict:
    # employee 关系已 selectinload 时带上员工信息，供前端列表/详情展示
    emp_name = emp_dept = emp_mentor = emp_manager = ""
    emp_position = ""
    try:
        if cr.employee is not None:
            emp_name = cr.employee.name or ""
            emp_dept = cr.employee.department or ""
            emp_mentor = cr.employee.mentor or ""
            emp_manager = cr.employee.manager or ""
            if cr.employee.position is not None:
                emp_position = cr.employee.position.name or ""
    except Exception:
        pass
    return {"id": str(cr.id), "employeeId": str(cr.employee_id),
            "employeeName": emp_name, "position": emp_position,
            "department": emp_dept, "mentor": emp_mentor, "manager": emp_manager,
            "week1Score": float(cr.week1_score) if cr.week1_score else None,
            "week2Score": float(cr.week2_score) if cr.week2_score else None,
            "week3Score": float(cr.week3_score) if cr.week3_score else None,
            "week4Score": float(cr.week4_score) if cr.week4_score else None,
            "mentorScore": float(cr.mentor_score) if cr.mentor_score else None,
            "disciplineScore": float(cr.discipline_score) if cr.discipline_score else None,
            "overallScore": float(cr.overall_score) if cr.overall_score else None,
            "aiReport": cr.ai_report, "employeeSummary": cr.employee_summary,
            "projectResults": cr.project_results, "abilityGaps": cr.ability_gaps,
            "next90DaysGoals": cr.next_90days_goals,
            "recommendation": cr.recommendation, "status": cr.status,
            "mentorComment": cr.mentor_comment, "managerComment": cr.manager_comment,
            "managerApproved": cr.manager_approved,
            "overrideReason": cr.override_reason,
            "overrideBy": str(cr.override_by) if cr.override_by else None,
            "overrideAt": iso_utc(cr.override_at),
            "createdAt": iso_utc(cr.created_at)}

def _serialize_employee(e: Employee) -> dict:
    return {"id": str(e.id), "name": e.name, "department": e.department, "status": e.status,
            "employeeType": e.employee_type, "matchLevel": e.match_level,
            "currentWeek": e.current_week, "totalWeeks": e.total_weeks,
            "mentor": e.mentor, "mentorId": str(e.mentor_id) if e.mentor_id else None,
            "manager": e.manager, "managerId": str(e.manager_id) if e.manager_id else None,
            "onboardDate": iso_utc(e.onboard_date),
            "probationEndDate": iso_utc(e.probation_end_date),
            "overallScore": float(e.overall_score) if e.overall_score else None,
            "riskLevel": e.risk_level}


# ═══════════════════════════════════════════════
# 试用期计划
# ═══════════════════════════════════════════════

class CreatePlanRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    type: str = "tech"
    total_weeks: int = Field(4, alias="totalWeeks")
    start_date: Optional[str] = Field(None, alias="startDate")


@router.post("/probation/plans")
async def create_plan(
    body: CreatePlanRequest,
    current: CurrentUser = Depends(require_permission("probation:manage")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(body.employee_id)
    except ValueError:
        return not_found("员工不存在")
    emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()
    if not emp:
        return not_found("员工不存在")

    dup = (await db.execute(select(ProbationPlan).where(ProbationPlan.employee_id == eid))).scalar_one_or_none()
    if dup:
        return fail(409, "该员工已有试用期计划")

    # 试用期长度由 match_level 决定
    weeks_map = {"high": 4, "experienced": 6, "fresh": 8, "core": 8}
    total_weeks = weeks_map.get(emp.match_level, 4)

    sd = None
    if body.start_date:
        try:
            sd = date.fromisoformat(body.start_date)
        except ValueError:
            sd = date.today()

    # AI 生成周计划框架
    try:
        from app.services.ai import llm_chat
        prompt = (
            render_prompt('talent/phase2.md', {'plan_type': body.type, 'total_weeks': total_weeks, 'department': emp.department or '技术'}, 'Prompt 1')
        )
        resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=2048)
        weeks_data = (extract_json_array(resp) or [])[:total_weeks]
    except Exception:
        weeks_data = []

    plan = ProbationPlan(
        employee_id=eid,
        type=body.type,
        total_weeks=total_weeks,
        start_date=sd or date.today(),
        end_date=(sd or date.today()) + timedelta(weeks=total_weeks),
        status="active",
        ai_generated=bool(weeks_data),
        weeks=weeks_data,
        created_by_id=current.id,
    )
    db.add(plan)
    await db.flush()

    # 创建培训进度
    await _ensure_training_progress(db, eid)

    # 员工 onboarding → training → probation
    if emp.status == "pending_onboard":
        try:
            emp.onboard_date = sd or date.today()
            await transition(db, "employee", emp, "training", actor_id=current.id,
                             actor_name=current.username, skip_block_check=True)
        except StateError as e:
            logger.warning("员工 %s 入职后无法进入培训: %s", eid, e.message)

    await write_audit(db, actor=current.username, action="创建试用期计划", section="probation")
    return ok(_serialize_plan(plan))


@router.get("/probation/plans/list/all")
async def list_plans(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查全部试用期计划(前端计划列表),带员工名/岗位,按创建时间倒序。"""
    rows = (await db.execute(
        select(ProbationPlan)
        .options(selectinload(ProbationPlan.employee).selectinload(Employee.position))
        .order_by(ProbationPlan.created_at.desc())
    )).scalars().all()
    return ok([_serialize_plan(p) for p in rows])


@router.get("/probation/plans/{employee_id}")
async def get_plan(
    employee_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    p = (await db.execute(
        select(ProbationPlan)
        .options(selectinload(ProbationPlan.employee).selectinload(Employee.position))
        .where(ProbationPlan.employee_id == eid)
    )).scalar_one_or_none()
    if not p:
        return not_found("未找到试用期计划")
    return ok(_serialize_plan(p))


@router.put("/probation/plans/{plan_id}")
async def update_plan(
    plan_id: str,
    body: dict,
    current: CurrentUser = Depends(require_permission("probation:manage")),
    db: AsyncSession = Depends(get_db),
):
    try:
        pid = uuid.UUID(plan_id)
    except ValueError:
        return not_found("计划不存在")
    p = (await db.execute(select(ProbationPlan).where(ProbationPlan.id == pid))).scalar_one_or_none()
    if not p:
        return not_found("计划不存在")
    if "weeks" in body:
        p.weeks = body["weeks"]
    if "type" in body:
        p.type = body["type"]
    await db.flush()
    return ok(_serialize_plan(p))


# ═══════════════════════════════════════════════
# 试用期周评
# ═══════════════════════════════════════════════

class CreateWeekReviewRequest(BaseModel):
    model_config = {"populate_by_name": True}
    week_number: int = Field(alias="weekNumber")
    dimensions: dict
    comment: Optional[str] = None


@router.post("/probation/{employee_id}/week-review")
async def save_week_review(
    employee_id: str,
    body: CreateWeekReviewRequest,
    current: CurrentUser = Depends(require_permission("probation:accept")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()
    if not emp:
        return not_found("员工不存在")

    total = sum(int(v) for v in body.dimensions.values())
    if not (0 <= total <= 100):
        return fail(400, "总分须在0-100之间")

    # upsert
    existing = (await db.execute(
        select(ProbationWeekReview).where(
            ProbationWeekReview.employee_id == eid,
            ProbationWeekReview.week_number == body.week_number,
        )
    )).scalar_one_or_none()
    review = existing or ProbationWeekReview(employee_id=eid, week_number=body.week_number)
    if not existing:
        db.add(review)
    review.dimensions = body.dimensions
    review.total_score = total
    review.comment = body.comment
    review.reviewer_id = current.id
    review.reviewer_name = current.display_name
    review.reviewed_at = datetime.utcnow()

    # 更新员工当前周
    emp.current_week = max(emp.current_week or 0, body.week_number)

    await db.flush()
    await write_audit(db, actor=current.username, action=f"第{body.week_number}周考核", section="probation")
    return ok(_serialize_review(review))


@router.get("/probation/{employee_id}/week-reviews")
async def list_week_reviews(
    employee_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return ok([])
    rows = (await db.execute(
        select(ProbationWeekReview).where(ProbationWeekReview.employee_id == eid)
        .order_by(ProbationWeekReview.week_number)
    )).scalars().all()
    return ok([_serialize_review(r) for r in rows])


# ═══════════════════════════════════════════════
# 转正审批
# ═══════════════════════════════════════════════

@router.get("/probation/confirmations/list")
async def list_confirmations(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查全部转正审批单（前端转正审批页列表），带员工信息，按创建时间倒序。"""
    rows = (await db.execute(
        select(ConfirmationReview)
        .options(selectinload(ConfirmationReview.employee).selectinload(Employee.position))
        .order_by(ConfirmationReview.created_at.desc())
    )).scalars().all()
    return ok([_serialize_confirmation(cr) for cr in rows])


@router.post("/probation/{employee_id}/confirmation/sync")
async def sync_confirmation(
    employee_id: str,
    current: CurrentUser = Depends(require_permission("probation:manage")),
    db: AsyncSession = Depends(get_db),
):
    """从周评和带教记录汇总生成转正审批单。"""
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")

    # 汇总四周评分
    reviews = (await db.execute(
        select(ProbationWeekReview).where(ProbationWeekReview.employee_id == eid)
    )).scalars().all()
    week_scores: dict[int, float] = {}
    for rv in reviews:
        week_scores[rv.week_number] = float(rv.total_score)

    # 带教汇总
    mentor_recs = (await db.execute(
        select(MentorRecord).where(MentorRecord.employee_id == eid)
    )).scalars().all()
    mentor_avg = round(sum(r.mentor_score or 0 for r in mentor_recs) / max(len(mentor_recs), 1), 2)

    # 纪律分默认 85
    discipline = 85

    w1 = week_scores.get(1, 0); w2 = week_scores.get(2, 0)
    w3 = week_scores.get(3, 0); w4 = week_scores.get(4, 0)
    overall = round(w1 * 0.15 + w2 * 0.20 + w3 * 0.20 + w4 * 0.25 + mentor_avg * 0.10 + discipline * 0.10, 2)

    thresholds = await _get_setting(db, "conversion_thresholds", {"excellent": 85, "normal": 75, "conditional": 65})
    if overall >= thresholds.get("excellent", 85):
        rec = "excellent"
    elif overall >= thresholds.get("normal", 75):
        rec = "normal"
    elif overall >= thresholds.get("conditional", 65):
        rec = "conditional"
    else:
        rec = "reject"

    # upsert
    existing = (await db.execute(
        select(ConfirmationReview).where(ConfirmationReview.employee_id == eid)
    )).scalar_one_or_none()
    cr = existing or ConfirmationReview(employee_id=eid)
    if not existing:
        db.add(cr)
    cr.week1_score = w1; cr.week2_score = w2; cr.week3_score = w3; cr.week4_score = w4
    cr.mentor_score = mentor_avg; cr.discipline_score = discipline
    cr.overall_score = overall; cr.recommendation = rec
    cr.status = "pending"

    # 推进员工到 pending_confirmation
    emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()
    if emp and emp.status == "probation":
        try:
            await transition(db, "employee", emp, "pending_confirmation",
                             actor_id=current.id, actor_name=current.username, skip_block_check=True)
        except StateError as e:
            logger.warning("员工 %s 无法进入待转正: %s", eid, e.message)

    await db.flush()
    await write_audit(db, actor=current.username, action="同步转正评分", section="confirmation")
    return ok(_serialize_confirmation(cr))


class ApproveConfirmationRequest(BaseModel):
    model_config = {"populate_by_name": True}
    recommendation: str   # excellent / normal / conditional / reject
    manager_comment: Optional[str] = Field(None, alias="managerComment")
    mentor_comment: Optional[str] = Field(None, alias="mentorComment")


@router.post("/probation/{employee_id}/confirmation/approve")
async def approve_confirmation(
    employee_id: str,
    body: ApproveConfirmationRequest,
    current: CurrentUser = Depends(require_permission("confirmation:approve")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    cr = (await db.execute(
        select(ConfirmationReview).where(ConfirmationReview.employee_id == eid)
    )).scalar_one_or_none()
    if not cr:
        return not_found("无转正审批记录，请先 sync")

    if cr.status in ("approved", "rejected"):
        return fail(409, "转正审批已处理")

    emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()
    if not emp:
        return not_found("员工不存在")

    rejected = body.recommendation == "reject"
    target_status = "probation" if rejected else "formal"

    # 员工状态迁移放在写审批单之前：迁移失败要整单拒绝并明确报错。
    # 此处过去是 except StateError: pass —— 审批单被写成「已通过」、员工却没转正，
    # 接口还返回 200，用户在审批页看到成功、员工列表里人还挂在试用期。
    try:
        await transition(db, "employee", emp, target_status,
                         actor_id=current.id, actor_name=current.username,
                         reason="转正未通过" if rejected else "转正通过",
                         skip_block_check=True)
    except StateError as e:
        await db.rollback()
        return conflict(e.message)

    cr.recommendation = body.recommendation
    cr.manager_comment = body.manager_comment
    cr.mentor_comment = body.mentor_comment
    cr.manager_approved = True
    cr.manager_approved_at = datetime.utcnow()
    cr.status = "rejected" if rejected else "approved"

    if not rejected:
        # P4 钩子: 创建人才画像
        try:
            from app.services.talent.talent_profile import ensure_talent_profile
            await ensure_talent_profile(db, eid)
        except Exception:
            pass

    emp.overall_score = cr.overall_score
    await write_audit(db, actor=current.username, action=f"转正审批: {body.recommendation}", section="confirmation")
    return ok(_serialize_confirmation(cr))


@router.put("/probation/{employee_id}/confirmation/recommendation")
async def update_recommendation(
    employee_id: str,
    body: dict,
    current: CurrentUser = Depends(require_permission("probation:manage")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    cr = (await db.execute(
        select(ConfirmationReview).where(ConfirmationReview.employee_id == eid)
    )).scalar_one_or_none()
    if not cr:
        return not_found("无转正审批记录")
    # camelCase(前端) → snake_case(ORM 列) 显式映射，避免 setattr 写到无效动态属性上
    field_map = {
        "employeeSummary": "employee_summary",
        "projectResults": "project_results",
        "abilityGaps": "ability_gaps",
        "next90DaysGoals": "next_90days_goals",
        "recommendation": "recommendation",
    }
    # 人工覆盖 AI 建议结论时必须留下原因（合规要求）：前端一直在收集这个原因，
    # 但此前既不在白名单里、也没有落库字段，等于走了个形式。
    old_rec = cr.recommendation
    new_rec = body.get("recommendation")
    override_reason = (body.get("overrideReason") or body.get("reason") or "").strip()
    is_override = bool(new_rec) and new_rec != old_rec
    if is_override and not override_reason:
        return fail(400, "人工调整转正结论必须填写原因")

    for field, column in field_map.items():
        if field in body:
            setattr(cr, column, body[field])

    if is_override:
        cr.override_reason = override_reason
        cr.override_by = current.id
        cr.override_at = datetime.utcnow()
        await write_audit(
            db, actor=current.username,
            action=f"人工调整转正结论 {old_rec or '—'} → {new_rec}（{override_reason}）",
            section="confirmation",
        )

    await db.flush()
    return ok(_serialize_confirmation(cr))


@router.get("/probation/{employee_id}/report")
async def get_report(
    employee_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    emp_row = await db.execute(select(Employee).where(Employee.id == eid))
    emp = emp_row.scalar_one_or_none()
    plan_row = await db.execute(select(ProbationPlan).where(ProbationPlan.employee_id == eid))
    plan = plan_row.scalar_one_or_none()
    review_rows = await db.execute(
        select(ProbationWeekReview).where(ProbationWeekReview.employee_id == eid)
        .order_by(ProbationWeekReview.week_number)
    )
    reviews = review_rows.scalars().all()
    cr_row = await db.execute(select(ConfirmationReview).where(ConfirmationReview.employee_id == eid))
    cr = cr_row.scalar_one_or_none()
    mentor_rows = await db.execute(
        select(MentorRecord).where(MentorRecord.employee_id == eid).order_by(MentorRecord.week)
    )
    mentors = mentor_rows.scalars().all()
    training_row = await db.execute(
        select(EmployeeTrainingProgress).where(EmployeeTrainingProgress.employee_id == eid)
    )
    training = training_row.scalar_one_or_none()

    return ok({
        "employee": _serialize_employee(emp) if emp else None,
        "plan": _serialize_plan(plan) if plan else None,
        "weekReviews": [_serialize_review(r) for r in reviews],
        "confirmation": _serialize_confirmation(cr) if cr else None,
        "mentorRecords": [_serialize_mentor(m) for m in mentors],
        "trainingProgress": _serialize_training_progress(training) if training else None,
    })


# ═══════════════════════════════════════════════
# 培训课程 + 进度
# ═══════════════════════════════════════════════

@router.get("/training/courses")
async def list_courses(
    current: CurrentUser = Depends(require_permission("training:view")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(TrainingCourse).order_by(TrainingCourse.day, TrainingCourse.title))
            ).scalars().all()
    if not rows:
        # 种子未写入则自动补
        await _seed_training_courses(db)
        rows = (await db.execute(select(TrainingCourse).order_by(TrainingCourse.day, TrainingCourse.title))
                ).scalars().all()
    return ok([{"id": str(c.id), "day": c.day, "title": c.title, "content": c.content,
                "duration": c.duration, "type": c.type} for c in rows])


async def _ensure_training_progress(db: AsyncSession, employee_id: uuid.UUID) -> EmployeeTrainingProgress:
    existing = (await db.execute(
        select(EmployeeTrainingProgress).where(EmployeeTrainingProgress.employee_id == employee_id)
    )).scalar_one_or_none()
    if existing:
        return existing

    courses = (await db.execute(select(TrainingCourse).order_by(TrainingCourse.day, TrainingCourse.title))
               ).scalars().all()
    if not courses:
        await _seed_training_courses(db)
        courses = (await db.execute(select(TrainingCourse).order_by(TrainingCourse.day, TrainingCourse.title))
                   ).scalars().all()

    tp = EmployeeTrainingProgress(
        employee_id=employee_id,
        courses=[{"courseId": str(c.id), "completed": False, "score": None} for c in courses],
        overall_rate=0,
    )
    db.add(tp)
    await db.flush()
    return tp


@router.get("/training/progress/{employee_id}")
async def get_training_progress(
    employee_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return fail(400, "无效ID")
    tp = (await db.execute(
        select(EmployeeTrainingProgress)
        .options(selectinload(EmployeeTrainingProgress.employee).selectinload(Employee.position))
        .where(EmployeeTrainingProgress.employee_id == eid)
    )).scalar_one_or_none()
    if not tp:
        tp = await _ensure_training_progress(db, eid)
    return ok(_serialize_training_progress(tp))


@router.post("/training/progress/{employee_id}/complete")
async def complete_course(
    employee_id: str,
    body: dict,
    current: CurrentUser = Depends(require_permission("training:confirm")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return fail(400, "无效ID")
    tp = (await db.execute(
        select(EmployeeTrainingProgress).where(EmployeeTrainingProgress.employee_id == eid)
    )).scalar_one_or_none()
    if not tp:
        tp = await _ensure_training_progress(db, eid)

    course_id = body.get("courseId")
    completed = body.get("completed", True)
    score = body.get("score")

    courses = list(tp.courses or [])
    for c in courses:
        if c.get("courseId") == course_id:
            c["completed"] = completed
            if score is not None:
                c["score"] = score
            break
    tp.courses = courses

    total = len(courses)
    done = sum(1 for c in courses if c.get("completed"))
    tp.overall_rate = round(done / max(total, 1) * 100, 2)
    if done == total:
        tp.completed_at = datetime.utcnow()
        # 培训完成 → 推进到试用期
        emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()
        if emp and emp.status == "training":
            try:
                await transition(db, "employee", emp, "probation",
                                 actor_id=current.id, actor_name=current.username, skip_block_check=True)
                emp.current_week = 1
            except StateError as e:
                logger.warning("员工 %s 培训完成后无法进入试用期: %s", eid, e.message)

    await db.flush()
    return ok(_serialize_training_progress(tp))


def _serialize_training_progress(tp: EmployeeTrainingProgress | None) -> dict | None:
    if not tp:
        return None
    # employee 关系已 selectinload 时带上员工名/岗位，供前端展示
    emp_name = emp_position = ""
    try:
        if tp.employee is not None:
            emp_name = tp.employee.name or ""
            if tp.employee.position is not None:
                emp_position = tp.employee.position.name or ""
    except Exception:
        pass
    return {"id": str(tp.id), "employeeId": str(tp.employee_id),
            "employeeName": emp_name, "position": emp_position,
            "courses": tp.courses,
            "overallRate": float(tp.overall_rate) if tp.overall_rate else 0,
            "completedAt": iso_utc(tp.completed_at)}


# ═══════════════════════════════════════════════
# 带教记录
# ═══════════════════════════════════════════════

def _serialize_mentor(m: MentorRecord) -> dict:
    # employee 关系已 selectinload 时带上员工名，供前端列表展示
    employee_name = ""
    try:
        if m.employee is not None:
            employee_name = m.employee.name or ""
    except Exception:
        employee_name = ""
    return {"id": str(m.id), "employeeId": str(m.employee_id),
            "employeeName": employee_name, "week": m.week,
            "trainingContent": m.training_content,
            "masteredSkills": m.mastered_skills, "pendingSkills": m.pending_skills,
            "completedTasks": m.completed_tasks, "issues": m.issues,
            "improvementPlan": m.improvement_plan, "mentorScore": m.mentor_score,
            "employeeConfirmed": m.employee_confirmed,
            "confirmedAt": iso_utc(m.confirmed_at),
            "createdAt": iso_utc(m.created_at)}


class CreateMentorRecordRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    week: int
    training_content: Optional[str] = Field(None, alias="trainingContent")
    mastered_skills: Optional[List[str]] = Field(None, alias="masteredSkills")
    pending_skills: Optional[List[str]] = Field(None, alias="pendingSkills")
    completed_tasks: Optional[List[str]] = Field(None, alias="completedTasks")
    issues: Optional[List[str]] = None
    improvement_plan: Optional[str] = Field(None, alias="improvementPlan")
    mentor_score: Optional[int] = Field(None, alias="mentorScore")


@router.get("/mentor-records")
async def list_all_mentor_records(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查全部带教记录（前端无参调用），带员工名，按创建时间倒序。"""
    rows = (await db.execute(
        select(MentorRecord)
        .options(selectinload(MentorRecord.employee))
        .order_by(MentorRecord.created_at.desc())
    )).scalars().all()
    return ok([_serialize_mentor(m) for m in rows])


@router.get("/mentor-records/{employee_id}")
async def get_mentor_records(
    employee_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(employee_id)
    except ValueError:
        return not_found("员工不存在")
    rows = (await db.execute(
        select(MentorRecord)
        .options(selectinload(MentorRecord.employee))
        .where(MentorRecord.employee_id == eid)
        .order_by(MentorRecord.week)
    )).scalars().all()
    return ok([_serialize_mentor(m) for m in rows])


@router.post("/mentor-records")
async def create_mentor_record(
    body: CreateMentorRecordRequest,
    current: CurrentUser = Depends(require_permission("mentor:record")),
    db: AsyncSession = Depends(get_db),
):
    try:
        eid = uuid.UUID(body.employee_id)
    except ValueError:
        return not_found("员工不存在")
    dup = (await db.execute(
        select(MentorRecord).where(MentorRecord.employee_id == eid, MentorRecord.week == body.week)
    )).scalar_one_or_none()
    if dup:
        return fail(409, f"第{body.week}周带教记录已存在，请更新而非新建")

    mr = MentorRecord(
        employee_id=eid, week=body.week,
        training_content=body.training_content,
        mastered_skills=body.mastered_skills, pending_skills=body.pending_skills,
        completed_tasks=body.completed_tasks, issues=body.issues,
        improvement_plan=body.improvement_plan, mentor_score=body.mentor_score,
        created_by=current.id,
    )
    db.add(mr)
    await db.flush()
    await write_audit(db, actor=current.username, action=f"撰写第{body.week}周带教记录", section="mentor")
    return ok(_serialize_mentor(mr))


@router.put("/mentor-records/{record_id}/confirm")
async def confirm_mentor_record(
    record_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        return not_found("带教记录不存在")
    mr = (await db.execute(select(MentorRecord).where(MentorRecord.id == rid))).scalar_one_or_none()
    if not mr:
        return not_found("带教记录不存在")
    mr.employee_confirmed = True
    mr.confirmed_at = datetime.utcnow()
    await db.flush()
    return ok(_serialize_mentor(mr))


# ═══════════════════════════════════════════════
# 种子
# ═══════════════════════════════════════════════

async def _seed_training_courses(db: AsyncSession):
    DEFAULT_COURSES = [
        (1, "公司文化与价值观", "企业使命、愿景、核心价值观、行为准则", "半天", "lecture"),
        (1, "组织架构与团队", "公司组织架构、各部门职能、关键联系人", "1h", "lecture"),
        (2, "岗位技能基础", "岗位所需技术栈/工具链/工作流程实操", "全天", "practice"),
        (3, "产品与业务", "公司产品线、商业模式、客户场景", "半天", "lecture"),
        (4, "合规与安全", "信息安全、数据隐私、合规制度", "2h", "lecture"),
        (5, "入职考核", "综合笔试+实操考核", "半天", "exam"),
    ]
    for day, title, content, duration, ctype in DEFAULT_COURSES:
        existing = (await db.execute(
            select(TrainingCourse).where(TrainingCourse.day == day, TrainingCourse.title == title)
        )).scalar_one_or_none()
        if not existing:
            db.add(TrainingCourse(day=day, title=title, content=content, duration=duration, type=ctype))
    await db.flush()
