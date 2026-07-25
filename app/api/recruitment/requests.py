"""模块一: 招聘需求管理 API。"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, List, Union, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Position, Department
from app.models.phase1 import RecruitmentRequest, PositionCompetency
from app.models.settings import SystemSetting
from app.models.auth import User
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit

router = APIRouter(tags=["招聘需求"])

DEFAULT_COMPETENCIES = [
    ("专业技能", 30), ("项目经验", 20), ("任务交付能力", 15),
    ("问题解决能力", 15), ("学习能力", 10), ("沟通协作与责任感", 10),
]


def _core_tasks_to_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    for sep in ("\n", "；", ";"):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text]


def _core_tasks_to_text(raw: Any) -> str:
    return "\n".join(_core_tasks_to_list(raw))


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value).strip())
    except ValueError:
        return None


def _format_ai_outputs(draft: Any) -> dict | None:
    """把 ai_draft 规范成前端 aiOutputs 展示结构。"""
    if not isinstance(draft, dict) or not draft:
        return None

    def dims(key_a: str, key_b: str) -> list[str]:
        raw = draft.get(key_a) or draft.get(key_b) or []
        if isinstance(raw, list):
            out: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    name = str(item.get("dimension") or item.get("name") or "").strip()
                    score = item.get("score")
                    out.append(f"{name}（{score}）" if name and score is not None else name or str(item))
                else:
                    text = str(item).strip()
                    if text:
                        out.append(text)
            return out
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return []

    rules = draft.get("resumeScoringRules")
    if isinstance(rules, dict):
        rules_text = "、".join(f"{k} {v}%" for k, v in rules.items())
    else:
        rules_text = str(rules or "")

    framework = draft.get("probationFramework")
    if isinstance(framework, dict):
        framework_text = "；".join(f"{k}: {v}" for k, v in framework.items() if v)
    else:
        framework_text = str(framework or "")

    training = draft.get("trainingSuggestions") or []
    if isinstance(training, str):
        training = [training] if training.strip() else []
    elif not isinstance(training, list):
        training = []

    competencies = draft.get("competencyModel") or []
    if not isinstance(competencies, list):
        competencies = []

    return {
        "jobDescription": str(draft.get("jobDescription") or ""),
        "competencyModel": [
            {"dimension": str(c.get("dimension") or ""), "weight": int(c.get("weight") or 0)}
            for c in competencies if isinstance(c, dict)
        ],
        "resumeScoringRules": rules_text,
        "round1Dimensions": dims("round1Dimensions", "interviewDimensionsR1"),
        "round2Dimensions": dims("round2Dimensions", "interviewDimensionsR2"),
        "probationFramework": framework_text,
        "trainingSuggestions": [str(x).strip() for x in training if str(x).strip()],
    }


def _serialize_req(req: RecruitmentRequest, viewer: CurrentUser | None = None) -> dict:
    salary = req.salary_range
    masked = False
    if viewer is None or not viewer.has("salary:view"):
        salary = None
        masked = True
    interviewers = req.interviewer_ids if isinstance(req.interviewer_ids, list) else []
    return {
        "id": str(req.id),
        "positionName": req.position_name,
        "departmentId": str(req.department_id) if req.department_id else None,
        "headcount": req.headcount,
        "salaryRange": salary,
        "salaryMasked": masked,
        "workExperience": req.work_experience or "",
        "educationRequirement": req.education_requirement or "",
        "jobDescription": req.job_description or "",
        "jobResponsibilities": req.job_responsibilities or "",
        "jobRequirements": req.job_requirements or "",
        "bonusItems": req.bonus_items or "",
        "coreTasks": _core_tasks_to_list(req.core_tasks),
        "requiredSkills": req.required_skills or [],
        "preferredSkills": req.preferred_skills or [],
        "deliverableReq": req.deliverable_req,
        "deliverableRequirements": req.deliverable_req,
        "probationGoal": req.probation_goal,
        "probationGoals": req.probation_goal,
        "eliminationCriteria": req.elimination_criteria,
        "interviewerIds": interviewers,
        "interviewers": interviewers,
        "directManagerId": str(req.direct_manager_id) if req.direct_manager_id else None,
        "directManager": req.direct_manager_name or "",
        "submitterId": str(req.submitter_id) if req.submitter_id else None,
        "status": req.status,
        "aiDraft": req.ai_draft,
        "aiOutputs": _format_ai_outputs(req.ai_draft),
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
    department_id: str = Field(alias="departmentId")
    headcount: int
    salary_range: str = Field(alias="salaryRange")
    work_experience: str = Field(alias="workExperience")
    education_requirement: str = Field(alias="educationRequirement")
    job_description: str = Field(alias="jobDescription")
    job_responsibilities: str = Field(alias="jobResponsibilities")
    job_requirements: str = Field(alias="jobRequirements")
    bonus_items: Optional[str] = Field(None, alias="bonusItems")
    core_tasks: Union[str, List[str]] = Field(alias="coreTasks")
    required_skills: List[str] = Field(alias="requiredSkills")
    preferred_skills: Optional[List[str]] = Field(None, alias="preferredSkills")
    deliverable_req: str = Field(alias="deliverableReq")
    probation_goal: str = Field(alias="probationGoal")
    elimination_criteria: str = Field(alias="eliminationCriteria")
    interviewer_ids: List[str] = Field(alias="interviewerIds")
    direct_manager_id: str = Field(alias="directManagerId")

    @field_validator("core_tasks", mode="before")
    @classmethod
    def _normalize_core_tasks(cls, v: Any) -> str:
        return _core_tasks_to_text(v)

    @field_validator("required_skills", "preferred_skills", "interviewer_ids", mode="before")
    @classmethod
    def _normalize_str_lists(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            return [p.strip() for p in v.replace("，", ",").split(",") if p.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return v


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
    core_tasks_text = _core_tasks_to_text(body.core_tasks)
    if not core_tasks_text:
        return fail(400, "工作任务不可为空")
    try:
        dept_id = uuid.UUID(body.department_id) if body.department_id else None
    except ValueError:
        return fail(400, "部门 ID 格式无效")

    manager_raw = (body.direct_manager_id or "").strip()
    if not manager_raw:
        return fail(400, "直属负责人不可为空")
    manager_id = _parse_uuid(manager_raw)
    manager_name = manager_raw
    if manager_id is None:
        row = await db.execute(select(User).where(User.display_name == manager_raw))
        matched = row.scalar_one_or_none()
        if matched is not None:
            manager_id = matched.id
            manager_name = matched.display_name
    else:
        row = await db.execute(select(User).where(User.id == manager_id))
        matched = row.scalar_one_or_none()
        manager_name = matched.display_name if matched else manager_raw

    req = RecruitmentRequest(
        position_name=body.position_name,
        headcount=body.headcount,
        department_id=dept_id,
        salary_range=body.salary_range,
        work_experience=body.work_experience,
        education_requirement=body.education_requirement,
        job_description=body.job_description,
        job_responsibilities=body.job_responsibilities,
        job_requirements=body.job_requirements,
        bonus_items=body.bonus_items or "",
        core_tasks=core_tasks_text,
        required_skills=body.required_skills,
        preferred_skills=body.preferred_skills or [],
        deliverable_req=body.deliverable_req,
        probation_goal=body.probation_goal,
        elimination_criteria=body.elimination_criteria,
        interviewer_ids=body.interviewer_ids,
        direct_manager_id=manager_id,
        direct_manager_name=manager_name,
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
            f"岗位: {req.position_name}\n"
            f"岗位描述: {req.job_description}\n工作职责: {req.job_responsibilities}\n"
            f"任职要求: {req.job_requirements}\n工作经验要求: {req.work_experience}\n"
            f"学历要求: {req.education_requirement}\n加分项: {req.bonus_items}\n"
            f"核心任务: {req.core_tasks}\n必备技能: {req.required_skills}\n\n"
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
    return ok(_serialize_req(req, current))


class UpdateReqRequest(BaseModel):
    model_config = {"populate_by_name": True}
    position_name: Optional[str] = Field(None, alias="positionName")
    department_id: Optional[str] = Field(None, alias="departmentId")
    headcount: Optional[int] = None
    salary_range: Optional[str] = Field(None, alias="salaryRange")
    work_experience: Optional[str] = Field(None, alias="workExperience")
    education_requirement: Optional[str] = Field(None, alias="educationRequirement")
    job_description: Optional[str] = Field(None, alias="jobDescription")
    job_responsibilities: Optional[str] = Field(None, alias="jobResponsibilities")
    job_requirements: Optional[str] = Field(None, alias="jobRequirements")
    bonus_items: Optional[str] = Field(None, alias="bonusItems")
    core_tasks: Optional[Union[str, List[str]]] = Field(None, alias="coreTasks")
    required_skills: Optional[List[str]] = Field(None, alias="requiredSkills")
    preferred_skills: Optional[List[str]] = Field(None, alias="preferredSkills")
    deliverable_req: Optional[str] = Field(None, alias="deliverableReq")
    probation_goal: Optional[str] = Field(None, alias="probationGoal")
    elimination_criteria: Optional[str] = Field(None, alias="eliminationCriteria")
    interviewer_ids: Optional[List[str]] = Field(None, alias="interviewerIds")
    direct_manager_id: Optional[str] = Field(None, alias="directManagerId")
    direct_manager_name: Optional[str] = Field(None, alias="directManager")
    ai_draft: Optional[dict] = Field(None, alias="aiDraft")

    @field_validator("core_tasks", mode="before")
    @classmethod
    def _normalize_core_tasks_update(cls, v: Any) -> Any:
        if v is None:
            return v
        return _core_tasks_to_text(v)


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
    data = body.model_dump(exclude_unset=True, by_alias=False)
    if "direct_manager_id" in data:
        raw = (data.pop("direct_manager_id") or "").strip()
        parsed = _parse_uuid(raw)
        data["direct_manager_id"] = parsed
        if "direct_manager_name" not in data:
            data["direct_manager_name"] = raw if parsed is None else data.get("direct_manager_name")
            if parsed is not None and not data.get("direct_manager_name"):
                urow = await db.execute(select(User).where(User.id == parsed))
                matched = urow.scalar_one_or_none()
                data["direct_manager_name"] = matched.display_name if matched else raw
    for field, value in data.items():
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

    # Resolve department name for Position record
    dept_name = None
    if req.department_id:
        dept_row = await db.execute(
            select(Department).where(Department.id == req.department_id)
        )
        dept = dept_row.scalar_one_or_none()
        dept_name = dept.name if dept else None

    draft = req.ai_draft or {}
    position = Position(
        name=req.position_name,
        department=dept_name,
        jd_content=draft.get("jobDescription"),
        jd_responsibilities=req.job_responsibilities or "",
        jd_requirements=req.job_requirements or "",
        jd_preferred=req.bonus_items or "",
        education_requirement=req.education_requirement or "",
        experience_requirement=req.work_experience or "",
        salary_range=req.salary_range,
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


@router.get("/departments")
async def list_departments(
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Department).order_by(Department.name)
    )).scalars().all()
    return ok([
        {"id": str(d.id), "name": d.name, "description": d.description}
        for d in rows
    ])
