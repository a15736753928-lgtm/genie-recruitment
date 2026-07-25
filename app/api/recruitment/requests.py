"""模块一: 招聘需求管理 API。"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Position
from app.models.phase1 import RecruitmentRequest, PositionCompetency
from app.models.settings import SystemSetting
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit

router = APIRouter(tags=["招聘需求"])

DEFAULT_COMPETENCIES = [
    ("专业技能", 30), ("项目经验", 20), ("任务交付能力", 15),
    ("问题解决能力", 15), ("学习能力", 10), ("沟通协作与责任感", 10),
]


def _serialize_req(req: RecruitmentRequest, viewer: CurrentUser | None = None) -> dict:
    salary = req.salary_range
    masked = False
    if viewer is None or not viewer.has("salary:view"):
        salary = None
        masked = True
    return {
        "id": str(req.id),
        "positionName": req.position_name,
        "headcount": req.headcount,
        "positionGoal": req.position_goal,
        "coreTasks": req.core_tasks,
        "requiredSkills": req.required_skills,
        "preferredSkills": req.preferred_skills,
        "projectExperienceReq": req.project_experience_req,
        "deliverableReq": req.deliverable_req,
        "salaryRange": salary,
        "salaryMasked": masked,
        "probationGoal": req.probation_goal,
        "eliminationCriteria": req.elimination_criteria,
        "interviewerIds": req.interviewer_ids,
        "directManagerId": str(req.direct_manager_id) if req.direct_manager_id else None,
        "submitterId": str(req.submitter_id) if req.submitter_id else None,
        "status": req.status,
        "aiDraft": req.ai_draft,
        "hrConfirmedBy": str(req.hr_confirmed_by) if req.hr_confirmed_by else None,
        "hrConfirmedAt": req.hr_confirmed_at.isoformat() if req.hr_confirmed_at else None,
        "deptConfirmedBy": str(req.dept_confirmed_by) if req.dept_confirmed_by else None,
        "deptConfirmedAt": req.dept_confirmed_at.isoformat() if req.dept_confirmed_at else None,
        "positionId": str(req.position_id) if req.position_id else None,
        "createdAt": req.created_at.isoformat() if req.created_at else None,
        "updatedAt": req.updated_at.isoformat() if req.updated_at else None,
    }


class CreateReqRequest(BaseModel):
    model_config = {"populate_by_name": True}
    position_name: str = Field(alias="positionName")
    headcount: int
    position_goal: str = Field(alias="positionGoal")
    core_tasks: str = Field(alias="coreTasks")
    required_skills: List[str] = Field(alias="requiredSkills")
    preferred_skills: Optional[List[str]] = Field(None, alias="preferredSkills")
    project_experience_req: str = Field(alias="projectExperienceReq")
    deliverable_req: str = Field(alias="deliverableReq")
    salary_range: str = Field(alias="salaryRange")
    probation_goal: str = Field(alias="probationGoal")
    elimination_criteria: str = Field(alias="eliminationCriteria")
    interviewer_ids: List[str] = Field(alias="interviewerIds")
    direct_manager_id: str = Field(alias="directManagerId")


@router.post("/recruitment-requests")
async def create_request(
    body: CreateReqRequest,
    current: CurrentUser = Depends(require_permission("recruitment_request:create")),
    db: AsyncSession = Depends(get_db),
):
    if body.headcount < 1:
        return fail(400, "招聘人数须≥1")
    if not body.required_skills:
        return fail(400, "必备技能不可为空")
    if not body.interviewer_ids:
        return fail(400, "必须指定至少一位面试官")
    req = RecruitmentRequest(
        position_name=body.position_name,
        headcount=body.headcount,
        position_goal=body.position_goal,
        core_tasks=body.core_tasks,
        required_skills=body.required_skills,
        preferred_skills=body.preferred_skills or [],
        project_experience_req=body.project_experience_req,
        deliverable_req=body.deliverable_req,
        salary_range=body.salary_range,
        probation_goal=body.probation_goal,
        elimination_criteria=body.elimination_criteria,
        interviewer_ids=body.interviewer_ids,
        direct_manager_id=uuid.UUID(body.direct_manager_id),
        submitter_id=current.id,
        status="draft",
    )
    db.add(req)
    await db.flush()
    await write_audit(db, actor=current.username, action="创建招聘需求", section="recruitment")
    return ok(_serialize_req(req, current))


@router.post("/recruitment-requests/{req_id}/ai-generate")
async def ai_generate(
    req_id: str,
    current: CurrentUser = Depends(require_permission("recruitment_request:create")),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    if req.status not in ("draft", "ai_generated"):
        return fail(409, "当前状态不允许 AI 生成")

    try:
        from app.services.ai import llm_chat
        prompt = (
            f"你是专业 HR 顾问。请根据以下招聘需求,生成 7 件结构化产品,以 JSON 格式返回。\n"
            f"岗位: {req.position_name}\n目标: {req.position_goal}\n核心任务: {req.core_tasks}\n"
            f"必备技能: {req.required_skills}\n\n"
            "返回 JSON 结构:\n"
            '{"jobDescription":"...","competencyModel":[{"dimension":"...","weight":30}],'
            '"resumeScoringRules":{"skillMatch":25,"projectMatch":20,"positionExp":15,'
            '"achievement":15,"industryExp":10,"learning":5,"stability":5,"bonusSkill":5},'
            '"interviewDimensionsR1":[{"dimension":"...","score":15}],'
            '"interviewDimensionsR2":[{"dimension":"...","score":20}],'
            '"probationFramework":{"week1":"...","weeks24":"..."},'
            '"trainingSuggestions":["..."]}'
        )
        import json
        resp_text = await llm_chat([{"role": "user", "content": prompt}])
        # extract JSON from response
        start = resp_text.find("{")
        end = resp_text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON found")
        ai_draft = json.loads(resp_text[start:end])
    except Exception as e:
        return fail(500, f"AI 生成失败，请重试: {e}")

    req.ai_draft = ai_draft
    if req.status == "draft":
        try:
            await transition(db, "recruitment_request", req, "ai_generated",
                             actor_id=current.id, actor_name=current.username,
                             skip_block_check=True)
        except StateError:
            req.status = "ai_generated"
    await write_audit(db, actor=current.username, action="AI 生成招聘草稿", section="recruitment")
    return ok({"aiDraft": ai_draft})


class UpdateReqRequest(BaseModel):
    model_config = {"populate_by_name": True}
    position_name: Optional[str] = Field(None, alias="positionName")
    headcount: Optional[int] = None
    position_goal: Optional[str] = Field(None, alias="positionGoal")
    core_tasks: Optional[str] = Field(None, alias="coreTasks")
    required_skills: Optional[List[str]] = Field(None, alias="requiredSkills")
    preferred_skills: Optional[List[str]] = Field(None, alias="preferredSkills")
    project_experience_req: Optional[str] = Field(None, alias="projectExperienceReq")
    deliverable_req: Optional[str] = Field(None, alias="deliverableReq")
    salary_range: Optional[str] = Field(None, alias="salaryRange")
    probation_goal: Optional[str] = Field(None, alias="probationGoal")
    elimination_criteria: Optional[str] = Field(None, alias="eliminationCriteria")
    interviewer_ids: Optional[List[str]] = Field(None, alias="interviewerIds")
    ai_draft: Optional[dict] = Field(None, alias="aiDraft")


@router.put("/recruitment-requests/{req_id}")
async def update_request(
    req_id: str,
    body: UpdateReqRequest,
    current: CurrentUser = Depends(require_permission("recruitment_request:create")),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    if req.status in ("published", "closed"):
        return fail(409, "已发布，不可修改")
    for field, value in body.model_dump(exclude_unset=True, by_alias=False).items():
        if hasattr(req, field) and value is not None:
            setattr(req, field, value)
    await db.flush()
    return ok(_serialize_req(req, current))


class ConfirmReqRequest(BaseModel):
    role: str   # hr / dept


@router.post("/recruitment-requests/{req_id}/confirm")
async def confirm_request(
    req_id: str,
    body: ConfirmReqRequest,
    current: CurrentUser = Depends(require_permission("recruitment_request:confirm")),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    if req.status in ("published", "closed"):
        return fail(409, "已发布或关闭，不可再确认")

    now = datetime.utcnow()
    if body.role == "hr":
        req.hr_confirmed_by = current.id
        req.hr_confirmed_at = now
        if req.status == "ai_generated":
            try:
                await transition(db, "recruitment_request", req, "hr_confirmed",
                                 actor_id=current.id, actor_name=current.username,
                                 skip_block_check=True)
            except StateError:
                req.status = "hr_confirmed"
    elif body.role == "dept":
        req.dept_confirmed_by = current.id
        req.dept_confirmed_at = now
    else:
        return fail(400, "role 须为 hr 或 dept")

    # 双方均确认 → dept_confirmed
    if req.hr_confirmed_by and req.dept_confirmed_by and req.status not in ("dept_confirmed", "published", "closed"):
        try:
            await transition(db, "recruitment_request", req, "dept_confirmed",
                             actor_id=current.id, actor_name=current.username,
                             skip_block_check=True)
        except StateError:
            req.status = "dept_confirmed"

    await write_audit(db, actor=current.username, action="确认招聘需求", section="recruitment")
    return ok(_serialize_req(req, current))


@router.post("/recruitment-requests/{req_id}/publish")
async def publish_request(
    req_id: str,
    current: CurrentUser = Depends(require_permission("position:publish")),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    if req.status != "dept_confirmed":
        return fail(409, "需 HR 与部门负责人双方确认后才能发布")

    draft = req.ai_draft or {}
    position = Position(
        name=req.position_name,
        department=draft.get("department"),
        jd_content=draft.get("jobDescription"),
        screening_criteria=draft.get("resumeScoringRules"),
        interview_criteria_r1=draft.get("interviewDimensionsR1"),
        interview_criteria_r2=draft.get("interviewDimensionsR2"),
        week1_project_requirement=draft.get("probationFramework"),
        conversion_criteria=draft.get("trainingSuggestions"),
    )
    db.add(position)
    await db.flush()

    competency_src = draft.get("competencyModel") or []
    if competency_src:
        for c in competency_src:
            db.add(PositionCompetency(
                position_id=position.id,
                dimension=c.get("dimension", ""),
                weight=c.get("weight", 0),
                locked=True,
            ))
    else:
        for dim, weight in DEFAULT_COMPETENCIES:
            db.add(PositionCompetency(position_id=position.id, dimension=dim, weight=weight, locked=True))

    req.position_id = position.id
    await transition(db, "recruitment_request", req, "published",
                     actor_id=current.id, actor_name=current.username, skip_block_check=True)
    await write_audit(db, actor=current.username, action=f"发布招聘需求 → 岗位: {req.position_name}", section="recruitment")
    return ok({"positionId": str(position.id)})


@router.get("/recruitment-requests")
async def list_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(RecruitmentRequest)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(
        q.order_by(RecruitmentRequest.created_at.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()
    return ok({
        "list": [_serialize_req(r, current) for r in rows],
        "total": total, "page": page, "pageSize": page_size,
    })


@router.get("/recruitment-requests/{req_id}")
async def get_request(
    req_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    return ok(_serialize_req(req, current))
