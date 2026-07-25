"""第四期: 人才画像 / 能力标签 / 晋级 / 期权 / 风险 / 看板。

能力等级: L1-L5 只升不降。晋级六项条件全 True 方可审批。
期权: 长期贡献×30% + 核心项目×25% + 专业×15% + 协作×15% + 责任×15%。
"""
from __future__ import annotations
import json, uuid, math
from datetime import datetime
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.phase4 import TalentProfile, AbilityTag, PromotionRecord, EquityRecord
from app.models.phase3 import PointRecord, WorkTask, RewardPenaltyRecord
from app.models.phase2 import MentorRecord
from app.models.probation import Employee
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.exceptions import push_exception
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit

router = APIRouter(tags=["人才池&晋级&期权(Phase4)"])


# ── 序列化 ───────────────────────────────────────────────────

def _serialize_profile(p: TalentProfile) -> dict:
    return {"id": str(p.id), "employeeId": str(p.employee_id),
            "currentPosition": p.current_position, "positionLevel": p.position_level,
            "abilityLevel": p.ability_level, "skills": p.skills, "department": p.department,
            "taskSuccessRate": float(p.task_success_rate) if p.task_success_rate else None,
            "onTimeRate": float(p.on_time_rate) if p.on_time_rate else None,
            "firstPassRate": float(p.first_pass_rate) if p.first_pass_rate else None,
            "totalConfirmedPoints": float(p.total_confirmed_points) if p.total_confirmed_points else None,
            "taskCount": p.task_count, "rewardCount": p.reward_count,
            "penaltyCount": p.penalty_count, "reworkRate": float(p.rework_rate) if p.rework_rate else None,
            "trainableSkills": p.trainable_skills, "assignableTasks": p.assignable_tasks,
            "canMentor": p.can_mentor,
            "promotionReadiness": float(p.promotion_readiness) if p.promotion_readiness else None,
            "talentRisk": p.talent_risk, "aiAnalysis": p.ai_analysis,
            "createdAt": p.created_at.isoformat() if p.created_at else None}

def _serialize_tag(t: AbilityTag) -> dict:
    return {"id": str(t.id), "employeeId": str(t.employee_id), "tag": t.tag, "level": t.level,
            "evidence": t.evidence, "confirmedBy": str(t.confirmed_by) if t.confirmed_by else None,
            "confirmedAt": t.confirmed_at.isoformat() if t.confirmed_at else None}

def _serialize_promotion(pr: PromotionRecord) -> dict:
    return {"id": str(pr.id), "employeeId": str(pr.employee_id),
            "fromLevel": pr.from_level, "toLevel": pr.to_level,
            "fromPosition": pr.from_position, "toPosition": pr.to_position,
            "criteriaCheck": pr.criteria_check, "allCriteriaMet": pr.all_criteria_met,
            "aiRecommendation": pr.ai_recommendation,
            "reviewResult": pr.review_result, "approverIds": pr.approver_ids,
            "rejectReason": pr.reject_reason, "status": pr.status,
            "createdBy": str(pr.created_by) if pr.created_by else None,
            "createdAt": pr.created_at.isoformat() if pr.created_at else None}

def _serialize_equity(e: EquityRecord) -> dict:
    return {"id": str(e.id), "employeeId": str(e.employee_id),
            "longTermPointsScore": float(e.long_term_points_score) if e.long_term_points_score else None,
            "coreProjectScore": float(e.core_project_score) if e.core_project_score else None,
            "professionalScore": float(e.professional_score) if e.professional_score else None,
            "collaborationScore": float(e.collaboration_score) if e.collaboration_score else None,
            "responsibilityScore": float(e.responsibility_score) if e.responsibility_score else None,
            "equityScore": float(e.equity_score) if e.equity_score else None,
            "aiRecommendation": e.ai_recommendation,
            "approverIds": e.approver_ids, "rejectReason": e.reject_reason,
            "status": e.status, "createdBy": str(e.created_by) if e.created_by else None,
            "createdAt": e.created_at.isoformat() if e.created_at else None}


# ═══════════════════════════════════════════════
# 人才画像
# ═══════════════════════════════════════════════

@router.post("/employees/{employee_id}/talent-profile/refresh")
async def refresh_talent_profile(
    employee_id: str,
    current: CurrentUser = Depends(require_permission("talent:manage")),
    db: AsyncSession = Depends(get_db),
):
    """聚合 P2/P3 数据更新人才画像。"""
    try: eid = uuid.UUID(employee_id)
    except ValueError: return not_found("员工不存在")

    # 确保画像存在
    existing = (await db.execute(
        select(TalentProfile).where(TalentProfile.employee_id == eid)
    )).scalar_one_or_none()
    profile = existing or TalentProfile(employee_id=eid)
    if not existing: db.add(profile)

    emp = (await db.execute(select(Employee).where(Employee.id == eid))).scalar_one_or_none()

    # 聚合 PointRecord
    confirmed = (await db.execute(
        select(func.sum(PointRecord.actual_points)).where(
            PointRecord.employee_id == eid, PointRecord.status == "confirmed"
        )
    )).scalar() or 0
    profile.total_confirmed_points = confirmed

    # 任务统计
    all_tasks = (await db.execute(
        select(WorkTask).where(WorkTask.owner_id == eid)
    )).scalars().all()
    total_tasks = len(all_tasks)
    passed = sum(1 for t in all_tasks if t.status == "passed")
    rework = sum(1 for t in all_tasks if t.rework_count > 0)
    profile.task_count = total_tasks
    profile.task_success_rate = round(passed / max(total_tasks, 1) * 100, 2)
    profile.rework_rate = round(rework / max(total_tasks, 1) * 100, 2)

    # 奖惩
    rew = (await db.execute(
        select(func.count()).select_from(RewardPenaltyRecord).where(
            RewardPenaltyRecord.employee_id == eid, RewardPenaltyRecord.type == "reward",
            RewardPenaltyRecord.status == "confirmed"
        )
    )).scalar() or 0
    pen = (await db.execute(
        select(func.count()).select_from(RewardPenaltyRecord).where(
            RewardPenaltyRecord.employee_id == eid, RewardPenaltyRecord.type == "penalty",
            RewardPenaltyRecord.status == "confirmed"
        )
    )).scalar() or 0
    profile.reward_count = rew; profile.penalty_count = pen

    # 晋升就绪度
    profile.promotion_readiness = round(
        float(profile.task_success_rate or 0) * 0.6 + (float(confirmed or 0) / 100) * 0.4, 2
    )
    profile.promotion_readiness = min(100, profile.promotion_readiness)

    # 风险判定
    profile.talent_risk = _assess_risk(profile, all_tasks)

    # 部门/岗位
    if emp:
        profile.department = emp.department
        profile.current_position = emp.position_id and str(emp.position_id) or None

    await db.flush()
    return ok(_serialize_profile(profile))


def _assess_risk(profile: TalentProfile, tasks: list[WorkTask]) -> str:
    rework_rate = float(profile.rework_rate or 0)
    if rework_rate > 25 or (profile.task_success_rate or 100) < 70:
        return "high"
    if rework_rate > 10 or (profile.task_success_rate or 100) < 85:
        return "watch"
    return "normal"


@router.get("/employees/{employee_id}/talent-profile")
async def get_profile(employee_id: str, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try: eid = uuid.UUID(employee_id)
    except ValueError: return not_found("无效ID")
    p = (await db.execute(select(TalentProfile).where(TalentProfile.employee_id == eid))).scalar_one_or_none()
    if not p: return not_found("未找到人才画像")
    return ok(_serialize_profile(p))


@router.put("/employees/{employee_id}/talent-profile")
async def update_profile(employee_id: str, body: dict, current: CurrentUser = Depends(require_permission("talent:manage")), db: AsyncSession = Depends(get_db)):
    try: eid = uuid.UUID(employee_id)
    except ValueError: return not_found("无效ID")
    p = (await db.execute(select(TalentProfile).where(TalentProfile.employee_id == eid))).scalar_one_or_none()
    if not p: return not_found("未找到人才画像")
    for field in ("skills", "trainableSkills", "assignableTasks", "canMentor", "positionLevel"):
        if field in body:
            setattr(p, field if field != "trainableSkills" else "trainable_skills", body[field])
            if field == "trainableSkills": setattr(p, "trainable_skills", body[field])
            elif field == "assignableTasks": setattr(p, "assignable_tasks", body[field])
            elif field == "canMentor": setattr(p, "can_mentor", body[field])
            elif field == "positionLevel": setattr(p, "position_level", body[field])
    if "abilityLevel" in body:
        new_lv = body["abilityLevel"]
        if p.ability_level and p.ability_level > new_lv:
            return fail(422, "能力等级只升不降")
        p.ability_level = new_lv
    if body.get("skills") is not None: p.skills = body["skills"]
    await db.flush()
    return ok(_serialize_profile(p))


# ═══════════════════════════════════════════════
# 能力标签
# ═══════════════════════════════════════════════

@router.get("/employees/{employee_id}/ability-tags")
async def list_tags(employee_id: str, current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try: eid = uuid.UUID(employee_id)
    except ValueError: return ok([])
    rows = (await db.execute(select(AbilityTag).where(AbilityTag.employee_id == eid))).scalars().all()
    return ok([_serialize_tag(t) for t in rows])


class CreateTagRequest(BaseModel):
    model_config = {"populate_by_name": True}
    tag: str; level: str = "L1"
    evidence: list   # 非空

@router.post("/employees/{employee_id}/ability-tags")
async def add_tag(employee_id: str, body: CreateTagRequest, current: CurrentUser = Depends(require_permission("talent:manage")), db: AsyncSession = Depends(get_db)):
    try: eid = uuid.UUID(employee_id)
    except ValueError: return not_found("无效ID")
    if not body.evidence or len(body.evidence) == 0:
        return fail(422, "能力标签必须提供证据(evidence 不可为空)")
    dup = (await db.execute(
        select(AbilityTag).where(AbilityTag.employee_id == eid, AbilityTag.tag == body.tag)
    )).scalar_one_or_none()
    if dup:
        # 只升不降
        if dup.level > body.level:
            return fail(422, f"能力标签只升不降({dup.level} > {body.level})")
        dup.level = body.level; dup.evidence = body.evidence
        await db.flush()
        return ok(_serialize_tag(dup))
    tag = AbilityTag(employee_id=eid, tag=body.tag, level=body.level, evidence=body.evidence, confirmed_by=current.id, confirmed_at=datetime.utcnow())
    db.add(tag); await db.flush()
    return ok(_serialize_tag(tag))


@router.delete("/employees/{employee_id}/ability-tags/{tag_id}")
async def remove_tag(employee_id: str, tag_id: str, current: CurrentUser = Depends(require_permission("talent:manage")), db: AsyncSession = Depends(get_db)):
    try: tid = uuid.UUID(tag_id)
    except ValueError: return not_found("标签不存在")
    tag = (await db.execute(select(AbilityTag).where(AbilityTag.id == tid))).scalar_one_or_none()
    if not tag: return not_found("标签不存在")
    await db.delete(tag); await db.flush()
    return ok(None)


# ═══════════════════════════════════════════════
# 人才池
# ═══════════════════════════════════════════════

@router.get("/talent-pool")
async def list_talent_pool(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    ability_level: Optional[str] = Query(None, alias="abilityLevel"),
    search: Optional[str] = None, department: Optional[str] = None,
    current: CurrentUser = Depends(require_permission("talent:view")),
    db: AsyncSession = Depends(get_db),
):
    q = select(TalentProfile)
    if ability_level: q = q.where(TalentProfile.ability_level == ability_level)
    if department: q = q.where(TalentProfile.department == department)
    if search:
        q = q.where(TalentProfile.skills.cast(String).ilike(f"%{search}%"))
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_profile(p) for p in rows], "total": total, "page": page, "pageSize": page_size})


@router.get("/talent-pool/search")
async def search_talent(q: str = Query(default=""), current: CurrentUser = Depends(require_permission("talent:view")), db: AsyncSession = Depends(get_db)):
    """按标签/技能搜索匹配人才(用于项目人员推荐)。"""
    if not q.strip(): return ok([])
    pattern = f"%{q.strip()}%"
    rows = (await db.execute(
        select(TalentProfile).where(
            or_(
                TalentProfile.skills.cast(String).ilike(pattern),
                TalentProfile.trainable_skills.cast(String).ilike(pattern),
            )
        ).limit(20)
    )).scalars().all()
    return ok([{"employeeId": str(p.employee_id), "abilityLevel": p.ability_level,
                "skills": p.skills, "department": p.department,
                "taskSuccessRate": float(p.task_success_rate) if p.task_success_rate else None}
               for p in rows])


@router.get("/talent-pool/risks")
async def talent_risks(current: CurrentUser = Depends(require_permission("talent:view")), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(TalentProfile).where(
            TalentProfile.talent_risk.in_(["watch", "high"])
        ).order_by(TalentProfile.talent_risk.desc())
    )).scalars().all()
    return ok([_serialize_profile(p) for p in rows])


# ═══════════════════════════════════════════════
# 晋级
# ═══════════════════════════════════════════════

class CreatePromotionRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    from_level: str = Field(alias="fromLevel")
    to_level: str = Field(alias="toLevel")
    from_position: Optional[str] = Field(None, alias="fromPosition")
    to_position: Optional[str] = Field(None, alias="toPosition")
    criteria_check: dict = Field(alias="criteriaCheck")
    # {pointsMet:true, coreAbilityMet:true, stableDelivery:true,
    #  noMajorViolation:true, higherTaskCapable:true, reviewPassed:true}

@router.post("/promotion/requests")
async def create_promotion(
    body: CreatePromotionRequest,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 六项条件校验
    criteria = body.criteria_check
    required = ["pointsMet", "coreAbilityMet", "stableDelivery", "noMajorViolation", "higherTaskCapable", "reviewPassed"]
    all_met = all(criteria.get(k, False) for k in required)
    if not all_met:
        return fail(422, "六项条件未全部满足: " + ", ".join(k for k in required if not criteria.get(k, False)))

    try: eid = uuid.UUID(body.employee_id)
    except ValueError: return not_found("员工不存在")

    pr = PromotionRecord(
        employee_id=eid, from_level=body.from_level, to_level=body.to_level,
        from_position=body.from_position, to_position=body.to_position,
        criteria_check=criteria, all_criteria_met=True,
        status="pending", created_by=current.id,
    )
    db.add(pr)
    await db.flush()
    await write_audit(db, actor=current.username, action=f"发起晋级: {body.from_level}→{body.to_level}", section="promotion")
    return ok(_serialize_promotion(pr))


@router.post("/promotion/requests/{request_id}/approve")
async def approve_promotion(
    request_id: str,
    current: CurrentUser = Depends(require_permission("promotion:approve")),
    db: AsyncSession = Depends(get_db),
):
    try: rid = uuid.UUID(request_id)
    except ValueError: return not_found("晋级申请不存在")
    pr = (await db.execute(select(PromotionRecord).where(PromotionRecord.id == rid))).scalar_one_or_none()
    if not pr: return not_found("晋级申请不存在")
    if pr.status in ("approved", "rejected"): return fail(409, "已审批")

    # 再次校验六项条件(防止篡改后依然通过)
    if not pr.all_criteria_met:
        return fail(422, "六项条件未全部满足,无法审批通过")

    # 双人审批: 记录审批人
    approvers: list = list(pr.approver_ids or [])
    if str(current.id) in approvers: return fail(409, "不能重复审批")
    approvers.append(str(current.id))
    pr.approver_ids = approvers

    if len(approvers) >= 2:
        pr.status = "approved"; pr.approved_at = datetime.utcnow()
        # 更新画像职级
        profile = (await db.execute(
            select(TalentProfile).where(TalentProfile.employee_id == pr.employee_id)
        )).scalar_one_or_none()
        if profile:
            profile.ability_level = pr.to_level
            profile.position_level = pr.to_position
            profile.promotion_readiness = 0
    else:
        pr.status = "pending"  # 等待第二位审批人

    await db.flush()
    await write_audit(db, actor=current.username, action=f"审批晋级: {pr.status}", section="promotion")
    return ok(_serialize_promotion(pr))


@router.get("/promotion/requests")
async def list_promotions(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    status: Optional[str] = None,
    current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    q = select(PromotionRecord)
    if status: q = q.where(PromotionRecord.status == status)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(PromotionRecord.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_promotion(p) for p in rows], "total": total, "page": page, "pageSize": page_size})


@router.get("/promotion/candidates")
async def promotion_candidates(current: CurrentUser = Depends(require_permission("promotion:approve")), db: AsyncSession = Depends(get_db)):
    """列出晋升就绪度≥60 且无高风险标记的员工画像。"""
    rows = (await db.execute(
        select(TalentProfile).where(
            TalentProfile.promotion_readiness >= 60,
            TalentProfile.talent_risk != "high",
        ).order_by(TalentProfile.promotion_readiness.desc()).limit(50)
    )).scalars().all()
    return ok([_serialize_profile(p) for p in rows])


# ═══════════════════════════════════════════════
# 期权
# ═══════════════════════════════════════════════

class CreateEquityRequest(BaseModel):
    model_config = {"populate_by_name": True}
    employee_id: str = Field(alias="employeeId")
    long_term_points_score: float = Field(0, alias="longTermPointsScore")
    core_project_score: float = Field(0, alias="coreProjectScore")
    professional_score: float = Field(0, alias="professionalScore")
    collaboration_score: float = Field(0, alias="collaborationScore")
    responsibility_score: float = Field(0, alias="responsibilityScore")

@router.post("/equity/records")
async def create_equity(
    body: CreateEquityRequest,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try: eid = uuid.UUID(body.employee_id)
    except ValueError: return not_found("员工不存在")
    dup = (await db.execute(select(EquityRecord).where(EquityRecord.employee_id == eid))).scalar_one_or_none()
    if dup: return fail(409, "该员工已有期权记录")

    equity_score = (
        body.long_term_points_score * 0.30 + body.core_project_score * 0.25 +
        body.professional_score * 0.15 + body.collaboration_score * 0.15 +
        body.responsibility_score * 0.15
    )
    er = EquityRecord(
        employee_id=eid, long_term_points_score=body.long_term_points_score,
        core_project_score=body.core_project_score, professional_score=body.professional_score,
        collaboration_score=body.collaboration_score, responsibility_score=body.responsibility_score,
        equity_score=round(equity_score, 2), status="pending", created_by=current.id,
    )
    db.add(er)
    await db.flush()
    return ok(_serialize_equity(er))


@router.post("/equity/records/{record_id}/approve")
async def approve_equity(
    record_id: str,
    current: CurrentUser = Depends(require_permission("equity:approve")),
    db: AsyncSession = Depends(get_db),
):
    try: rid = uuid.UUID(record_id)
    except ValueError: return not_found("期权记录不存在")
    er = (await db.execute(select(EquityRecord).where(EquityRecord.id == rid))).scalar_one_or_none()
    if not er: return not_found("期权记录不存在")
    if er.status in ("approved", "rejected"): return fail(409, "已审批")

    approvers: list = list(er.approver_ids or [])
    if str(current.id) in approvers: return fail(409, "不能重复审批")
    approvers.append(str(current.id))
    er.approver_ids = approvers

    if len(approvers) >= 2:
        er.status = "approved"; er.approved_at = datetime.utcnow()
    await db.flush()
    await write_audit(db, actor=current.username, action=f"审批期权: {er.status}", section="equity")
    return ok(_serialize_equity(er))


@router.get("/equity/records")
async def list_equity(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    status: Optional[str] = None,
    current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    q = select(EquityRecord)
    if status: q = q.where(EquityRecord.status == status)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(q.order_by(EquityRecord.created_at.desc()).offset((page-1)*page_size).limit(page_size))).scalars().all()
    return ok({"list": [_serialize_equity(e) for e in rows], "total": total, "page": page, "pageSize": page_size})


@router.get("/equity/candidates")
async def equity_candidates(current: CurrentUser = Depends(require_permission("equity:view")), db: AsyncSession = Depends(get_db)):
    """推荐期权候选人: 总积分≥500 且人才风险正常。"""
    rows = (await db.execute(
        select(TalentProfile).where(
            TalentProfile.total_confirmed_points >= 500,
            TalentProfile.talent_risk == "normal",
        ).order_by(TalentProfile.total_confirmed_points.desc()).limit(20)
    )).scalars().all()
    return ok([_serialize_profile(p) for p in rows])


# ═══════════════════════════════════════════════
# 综合看板
# ═══════════════════════════════════════════════

@router.get("/dashboard/talent")
async def talent_dashboard(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # 能力分布
    level_dist = (await db.execute(
        select(TalentProfile.ability_level, func.count())
        .group_by(TalentProfile.ability_level).order_by(TalentProfile.ability_level)
    )).all()
    ability_distribution = [{"level": l, "count": c} for l, c in level_dist]

    # 高潜(晋升就绪≥80)
    high_pot = (await db.execute(
        select(func.count()).where(TalentProfile.promotion_readiness >= 80)
    )).scalar() or 0

    # 核心成员(能力L4/L5)
    core = (await db.execute(
        select(func.count()).where(TalentProfile.ability_level.in_(["L4", "L5"]))
    )).scalar() or 0

    # 可带教人数
    mentors = (await db.execute(
        select(func.count()).where(TalentProfile.can_mentor == True)
    )).scalar() or 0

    # 风险人员
    at_risk = (await db.execute(
        select(func.count()).where(TalentProfile.talent_risk.in_(["watch", "high"]))
    )).scalar() or 0

    # 按部门
    dept_dist = (await db.execute(
        select(TalentProfile.department, func.count())
        .where(TalentProfile.department.isnot(None))
        .group_by(TalentProfile.department)
    )).all()
    by_department = [{"department": d, "count": c} for d, c in dept_dist]

    # 能力缺口(哪些技能普遍缺失 — 简化版)
    tag_rows = (await db.execute(
        select(AbilityTag.tag).group_by(AbilityTag.tag)
    )).scalars().all()

    return ok({
        "abilityDistribution": ability_distribution,
        "highPotential": high_pot, "coreMembers": core,
        "mentors": mentors, "atRisk": at_risk,
        "byDepartment": by_department,
        "totalProfiles": sum(c for _, c in level_dist),
        "knownTags": list(tag_rows)[:20],
    })


@router.get("/dashboard/recruitment")
async def recruitment_dashboard(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """招聘漏斗 + 转化率。"""
    from app.models.recruitment import Candidate
    total_c = (await db.execute(select(func.count()).select_from(Candidate))).scalar() or 0
    hired = (await db.execute(select(func.count()).where(Candidate.status == "hired"))).scalar() or 0

    return ok({
        "totalCandidates": total_c, "hired": hired,
        "conversionRate": round(hired / max(total_c, 1) * 100, 1),
    })


@router.get("/dashboard/probation")
async def probation_dashboard(current: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """试用期概览。"""
    total_emp = (await db.execute(select(func.count()).select_from(Employee))).scalar() or 0
    in_prob = (await db.execute(
        select(func.count()).where(Employee.status.in_(["training", "probation", "pending_confirmation"]))
    )).scalar() or 0
    formal = (await db.execute(
        select(func.count()).where(Employee.status == "formal")
    )).scalar() or 0

    from app.models.phase2 import ConfirmationReview
    excellent = (await db.execute(
        select(func.count()).where(ConfirmationReview.recommendation == "excellent")
    )).scalar() or 0
    normal = (await db.execute(
        select(func.count()).where(ConfirmationReview.recommendation == "normal")
    )).scalar() or 0

    return ok({
        "totalEmployees": total_emp, "inProbation": in_prob, "formal": formal,
        "conversionExcellent": excellent, "conversionNormal": normal,
    })
