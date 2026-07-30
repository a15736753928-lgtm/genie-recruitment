"""模块一: 招聘需求管理 API。"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, List, Union, Any
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Position, Department
from app.models.phase1 import RecruitmentRequest, PositionCompetency
from app.models.phase1_ai import PositionAIArtifact, ARTIFACT_TYPES
from app.models.settings import SystemSetting
from app.models.auth import User
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
from app.utils.clock import iso_utc
from app.utils.llm_json import extract_json_object

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
    """把 ai_draft 规范成前端 aiOutputs 展示结构。

    保留完整的结构化数据（description, sampleQuestions, phases 等），
    不再扁平化为纯文本，让前端可以丰富渲染。
    """
    if not isinstance(draft, dict) or not draft:
        return None

    def normalize_dims(raw: Any) -> list[dict]:
        """将面试维度/能力模型列表规范化为结构化数组。"""
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if isinstance(item, dict):
                out.append({
                    "dimension": str(item.get("dimension") or item.get("name") or "").strip(),
                    "weight": int(item.get("weight") or item.get("score") or 0),
                    "description": str(item.get("description") or "").strip(),
                    "sampleQuestions": item.get("sampleQuestions") or [],
                })
            elif isinstance(item, str) and item.strip():
                out.append({"dimension": item.strip(), "weight": 0, "description": "", "sampleQuestions": []})
        return out

    def normalize_framework(raw: Any) -> dict:
        """将试用期考核框架规范化。"""
        if isinstance(raw, dict):
            phases = raw.get("phases")
            if isinstance(phases, list):
                return {"phases": [
                    {
                        "name": str(p.get("name") or ""),
                        "duration": str(p.get("duration") or ""),
                        "goals": p.get("goals") or [],
                        "criteria": p.get("criteria") or "",
                        "evaluationMethod": str(p.get("evaluationMethod") or p.get("evaluation_method") or ""),
                    } for p in phases if isinstance(p, dict)
                ]}
            # 兼容旧格式 {week1: ..., weeks24: ...}
            return {"phases": [
                {"name": "第1周", "duration": "1周", "goals": [str(raw.get("week1", ""))], "evaluationMethod": ""},
                {"name": "第2-4周", "duration": "3周", "goals": [str(raw.get("weeks24", ""))], "evaluationMethod": ""},
            ]}
        if isinstance(raw, str) and raw.strip():
            return {"phases": [{"name": "试用期", "duration": "", "goals": [raw.strip()], "evaluationMethod": ""}]}
        return {"phases": []}

    def normalize_training(raw: Any) -> list[dict]:
        """将培训内容建议规范化。"""
        if not isinstance(raw, list):
            if isinstance(raw, str) and raw.strip():
                return [{"title": raw.strip(), "duration": "", "content": "", "objectives": [], "method": ""}]
            return []
        out = []
        for item in raw:
            if isinstance(item, dict):
                out.append({
                    "title": str(item.get("title") or ""),
                    "duration": str(item.get("duration") or ""),
                    "content": str(item.get("content") or ""),
                    "objectives": item.get("objectives") or [],
                    "method": str(item.get("method") or ""),
                })
            elif isinstance(item, str) and item.strip():
                out.append({"title": item.strip(), "duration": "", "content": "", "objectives": [], "method": ""})
        return out

    rules = draft.get("resumeScoringRules")
    if isinstance(rules, dict):
        # 保留完整结构，同时提供预览文本
        rules_normalized = {k: int(v) for k, v in rules.items() if isinstance(v, (int, float))}
        rules_preview = "、".join(f"{k} {v}%" for k, v in rules.items())
    else:
        rules_normalized = {}
        rules_preview = str(rules or "")

    competencies = draft.get("competencyModel") or []

    # jobDescription 支持三种格式：
    # 1. 新格式: draft.jobDescription = { basicInfo: {...}, mission: "...", ... }
    # 2. 扁平格式: draft = { basicInfo: {...}, mission: "...", ... } (AI 可能不嵌套)
    # 3. 旧格式: draft.jobDescription = "纯文本..."
    job_desc = draft.get("jobDescription")
    if isinstance(job_desc, dict):
        # 新格式，jobDescription 是结构化对象
        job_desc_formatted = job_desc
    elif isinstance(draft.get("basicInfo"), dict):
        # 扁平格式，basicInfo 直接在顶层，整个 draft 就是岗位说明书
        job_desc_formatted = {
            "basicInfo": draft.get("basicInfo"),
            "mission": draft.get("mission"),
            "responsibilities": draft.get("responsibilities"),
            "qualifications": draft.get("qualifications"),
            "permissions": draft.get("permissions"),
            "collaborations": draft.get("collaborations"),
            "workEnvironment": draft.get("workEnvironment"),
            "kpi": draft.get("kpi"),
            "careerPath": draft.get("careerPath"),
        }
    else:
        # 旧的纯文本格式，转为字符串
        job_desc_formatted = str(job_desc or "")

    return {
        "jobDescription": job_desc_formatted,
        "competencyModel": normalize_dims(competencies),
        "resumeScoringRules": rules_normalized,
        "resumeScoringRulesPreview": rules_preview,
        "round1Dimensions": normalize_dims(draft.get("interviewDimensionsR1") or draft.get("round1Dimensions")),
        "round2Dimensions": normalize_dims(draft.get("interviewDimensionsR2") or draft.get("round2Dimensions")),
        "probationFramework": normalize_framework(draft.get("probationFramework")),
        "trainingPlan": normalize_training(draft.get("trainingPlan") or draft.get("trainingSuggestions")),
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
        "hrConfirmedAt": iso_utc(req.hr_confirmed_at),
        "deptConfirmedBy": str(req.dept_confirmed_by) if req.dept_confirmed_by else None,
        "deptConfirmedAt": iso_utc(req.dept_confirmed_at),
        "positionId": str(req.position_id) if req.position_id else None,
        "createdAt": iso_utc(req.created_at),
        "updatedAt": iso_utc(req.updated_at),
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
        import asyncio
        import logging
        from app.services.ai import llm_chat
        from app.utils.llm_json import extract_json_object, extract_json_array
        _log = logging.getLogger(__name__)

        # ── 基础信息（所有 prompt 共用）────────────────────────────
        base_info = (
            f"岗位名称: {req.position_name}\n"
            f"部门: {req.department_id or '待定'}\n"
            f"招聘人数: {req.headcount}\n"
            f"岗位描述: {req.job_description}\n"
            f"工作职责: {req.job_responsibilities}\n"
            f"任职要求: {req.job_requirements}\n"
            f"工作经验要求: {req.work_experience}\n"
            f"学历要求: {req.education_requirement}\n"
            f"加分项: {req.bonus_items}\n"
            f"核心任务: {req.core_tasks}\n"
            f"必备技能: {', '.join(req.required_skills or [])}\n"
            f"优先技能: {', '.join(req.preferred_skills or [])}\n"
            f"薪资范围: {req.salary_range}\n"
            f"试用期目标: {req.probation_goal}\n"
            f"淘汰条件: {req.elimination_criteria}"
        )

        # ── 7 个独立的生成函数 ─────────────────────────────────────

        async def gen_job_description():
            """生成岗位说明书（结构化 JSON）"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成结构化的岗位说明书。

## 招聘需求
{base_info}

## 返回 JSON（只返回 JSON，不要其他内容）
{{
  "basicInfo": {{
    "positionName": "岗位名称",
    "department": "所属部门",
    "headcount": 1,
    "reportTo": "汇报对象",
    "salaryRange": "薪资范围",
    "workLocation": "工作地点",
    "employmentType": "正式员工",
    "level": "岗位等级"
  }},
  "mission": "岗位使命和核心价值（100字以上）",
  "responsibilities": ["主要工作职责1（含量化指标）", "职责2"],
  "qualifications": {{
    "required": ["必须具备的条件1", "条件2"],
    "preferred": ["优先条件1", "加分项2"]
  }},
  "permissions": ["岗位工作权限1", "权限2"],
  "collaborations": {{
    "internal": ["内部协作部门/角色1", "协作2"],
    "external": ["外部协作对象1"]
  }},
  "workEnvironment": {{
    "officeType": "办公室/远程/混合",
    "workingHours": "工作时间",
    "overtime": "加班情况",
    "travel": "出差要求"
  }},
  "kpi": ["核心绩效考核指标1", "指标2", "指标3"],
  "careerPath": "职业发展通道描述"
}}"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=2048)
            return ("jobDescription", extract_json_object(resp))

        async def gen_competency_model():
            """生成能力模型"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成岗位能力模型。

## 招聘需求
{base_info}

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {{"dimension": "专业技能", "weight": 30, "description": "评估标准描述"}},
  {{"dimension": "项目经验", "weight": 20, "description": "评估标准描述"}},
  {{"dimension": "任务交付能力", "weight": 15, "description": "评估标准描述"}},
  {{"dimension": "问题解决能力", "weight": 15, "description": "评估标准描述"}},
  {{"dimension": "学习能力", "weight": 10, "description": "评估标准描述"}},
  {{"dimension": "沟通协作与责任感", "weight": 10, "description": "评估标准描述"}}
]"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=1024)
            return ("competencyModel", extract_json_array(resp))

        async def gen_resume_scoring():
            """生成简历评分规则"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成简历评分规则。

## 招聘需求
{base_info}

## 返回 JSON（只返回 JSON，不要其他内容）
{{
  "skillMatch": 25,
  "projectMatch": 20,
  "positionExp": 15,
  "achievement": 15,
  "industryExp": 10,
  "learning": 5,
  "stability": 5,
  "bonusSkill": 5
}}"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=512)
            return ("resumeScoringRules", extract_json_object(resp))

        async def gen_interview_r1():
            """生成一面维度"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成第一轮面试（初试）的评估维度。

## 招聘需求
{base_info}

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {{"dimension": "经历真实性", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]}},
  {{"dimension": "专业基础", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]}},
  {{"dimension": "项目经验", "weight": 20, "description": "评估要点描述", "sampleQuestions": ["问题1"]}},
  {{"dimension": "学习能力", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]}},
  {{"dimension": "沟通表达", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]}},
  {{"dimension": "稳定性与动机", "weight": 10, "description": "评估要点描述", "sampleQuestions": ["问题1"]}}
]"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=1500)
            return ("interviewDimensionsR1", extract_json_array(resp))

        async def gen_interview_r2():
            """生成二面维度"""
            prompt = f"""你是资深技术面试官。根据以下招聘需求，生成第二轮面试（复试）的评估维度。

## 招聘需求
{base_info}

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {{"dimension": "系统设计能力", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]}},
  {{"dimension": "技术深度", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]}},
  {{"dimension": "问题解决思路", "weight": 20, "description": "评估要点描述", "sampleQuestions": ["问题1"]}},
  {{"dimension": "技术视野", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]}},
  {{"dimension": "领导力/影响力", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]}}
]"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=1500)
            return ("interviewDimensionsR2", extract_json_array(resp))

        async def gen_probation_framework():
            """生成试用期框架"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成试用期考核框架。

## 招聘需求
{base_info}
- 试用期目标: {req.probation_goal}
- 淘汰条件: {req.elimination_criteria}

## 返回 JSON（只返回 JSON，不要其他内容）
{{
  "phases": [
    {{"name": "第1周", "duration": "1周", "goals": ["熟悉环境和团队", "了解项目背景"], "evaluationMethod": "导师评价"}},
    {{"name": "第2-4周", "duration": "3周", "goals": ["独立完成小型任务", "熟悉开发流程"], "evaluationMethod": "任务完成度评估"}},
    {{"name": "第2-3个月", "duration": "2个月", "goals": ["独立负责模块开发", "达到试用期目标"], "evaluationMethod": "转正评审"}}
  ]
}}"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=1024)
            return ("probationFramework", extract_json_object(resp))

        async def gen_training_plan():
            """生成培训建议"""
            prompt = f"""你是资深 HR 顾问。根据以下招聘需求，生成入职培训计划。

## 招聘需求
{base_info}

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {{"title": "公司文化与价值观", "duration": "半天", "content": "企业使命、愿景、核心价值观", "objectives": ["了解公司文化", "认同核心价值观"], "method": "讲座"}},
  {{"title": "产品与业务介绍", "duration": "1天", "content": "公司产品线、业务模式、客户群体", "objectives": ["理解产品定位", "了解业务流程"], "method": "讲解+演示"}},
  {{"title": "技术架构概览", "duration": "1天", "content": "技术栈、系统架构、开发规范", "objectives": ["熟悉技术栈", "了解代码规范"], "method": "技术分享"}},
  {{"title": "开发工具与流程", "duration": "半天", "content": "Git、CI/CD、项目管理工具", "objectives": ["掌握开发工具", "熟悉发布流程"], "method": "实操"}},
  {{"title": "团队协作规范", "duration": "半天", "content": "代码评审、文档规范、沟通机制", "objectives": ["了解协作流程", "掌握规范要求"], "method": "讲解"}}
]"""
            resp = await llm_chat([{"role": "user", "content": prompt}], max_tokens=1500)
            return ("trainingPlan", extract_json_array(resp))

        # ── 并发执行所有 7 个生成任务 ──────────────────────────────
        _log.info("开始并发生成 7 件产品...")
        results = await asyncio.gather(
            gen_job_description(),
            gen_competency_model(),
            gen_resume_scoring(),
            gen_interview_r1(),
            gen_interview_r2(),
            gen_probation_framework(),
            gen_training_plan(),
            return_exceptions=True,
        )

        # ── 合并结果 ──────────────────────────────────────────────
        ai_draft = {}
        for r in results:
            if isinstance(r, Exception):
                _log.error("AI 生成失败: %s", r)
                continue
            key, value = r
            if value is not None:
                ai_draft[key] = value
            else:
                _log.warning("AI 生成 %s 返回空", key)

        if not ai_draft:
            raise ValueError("所有 AI 生成任务均失败，请重试")

        _log.info("AI 生成完成，成功 %d/7 项: %s", len(ai_draft), list(ai_draft.keys()))
    except Exception as e:
        _log.exception("AI 生成招聘需求失败")
        return fail(500, f"AI 生成失败，请重试: {e}")

    # 状态迁移放在赋值之前：迁移失败要整单拒绝，否则 ai_draft 会被静默提交
    if req.status == "draft":
        try:
            await transition(db, "recruitment_request", req, "ai_generated",
                             actor_id=current.id, actor_name=current.username,
                             skip_block_check=True)
        except StateError as e:
            await db.rollback()
            return fail(409, e.message)
    req.ai_draft = ai_draft

    # 将 7 项 AI 产物分别写入 position_ai_artifacts 表，供下游复用
    artifact_mapping = {
        "job_description": ai_draft.get("jobDescription"),
        "competency_model": ai_draft.get("competencyModel"),
        "resume_scoring_rules": ai_draft.get("resumeScoringRules"),
        "interview_r1": ai_draft.get("interviewDimensionsR1") or ai_draft.get("interviewDimensionsR1"),
        "interview_r2": ai_draft.get("interviewDimensionsR2") or ai_draft.get("interviewDimensionsR2"),
        "probation_framework": ai_draft.get("probationFramework"),
        "training_plan": ai_draft.get("trainingPlan") or ai_draft.get("trainingSuggestions"),
    }
    for artifact_type, content in artifact_mapping.items():
        if content is None:
            continue
        # 如果已存在同类型记录，更新版本
        existing = await db.execute(
            select(PositionAIArtifact).where(
                PositionAIArtifact.recruitment_request_id == req.id,
                PositionAIArtifact.artifact_type == artifact_type,
            )
        )
        existing_record = existing.scalar_one_or_none()
        if existing_record:
            existing_record.content = content
            existing_record.version = (existing_record.version or 1) + 1
        else:
            db.add(PositionAIArtifact(
                recruitment_request_id=req.id,
                artifact_type=artifact_type,
                content=content,
            ))

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

    # 注意：hr_confirmed_by/dept_confirmed_by 的赋值发生在状态迁移校验之前；
    # get_db() 只在路由抛出异常时才 rollback（见 app/database.py:28-37），
    # 若 transition() 校验失败后直接 return fail()，这两个字段的赋值仍会被
    # 悄悄提交——曾导致"一次不合法的 dept 确认"残留 dept_confirmed_by，
    # 污染后续请求让状态直接跳级（跳过 hr_confirmed 直接到 dept_confirmed）。
    # 因此任何一次 transition() 失败都必须显式 rollback。
    now = datetime.utcnow()
    if body.role == "hr":
        req.hr_confirmed_by = current.id
        req.hr_confirmed_at = now
        # draft 也要能进 hr_confirmed：AI 生成不是必经步骤，需求可以手工填完直接确认。
        # 此前只认 ai_generated，导致「部门先确认 → HR 再确认」这条路上状态一直停在
        # draft，最后那步 draft → dept_confirmed 必然非法，双方都确认了却永远发布不了。
        if req.status in ("draft", "ai_generated"):
            try:
                await transition(db, "recruitment_request", req, "hr_confirmed",
                                 actor_id=current.id, actor_name=current.username,
                                 skip_block_check=True)
            except StateError as e:
                await db.rollback()
                return fail(409, e.message)
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
        except StateError as e:
            await db.rollback()
            return fail(409, e.message)

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

    # positions.name 有唯一约束：重名时此处会抛 IntegrityError 变成裸 500，
    # 用户只看到「服务器内部错误」，完全不知道是岗位重名。先查再给明确提示。
    dup = await db.execute(select(Position).where(Position.name == req.position_name))
    existing_position = dup.scalar_one_or_none()
    if existing_position is not None:
        return fail(409, f"岗位「{req.position_name}」已存在，请修改需求里的岗位名称后再发布")

    # Resolve department name for Position record
    dept_name = None
    if req.department_id:
        dept_row = await db.execute(
            select(Department).where(Department.id == req.department_id)
        )
        dept = dept_row.scalar_one_or_none()
        dept_name = dept.name if dept else None

    draft = req.ai_draft or {}
    import json as _json
    # jd_content 是 VARCHAR，结构化格式的 jobDescription 需要转为 JSON 字符串
    jd_content_raw = draft.get("jobDescription")
    if isinstance(jd_content_raw, dict):
        jd_content_str = _json.dumps(jd_content_raw, ensure_ascii=False)
    elif jd_content_raw:
        jd_content_str = str(jd_content_raw)
    else:
        jd_content_str = ""

    position = Position(
        name=req.position_name,
        department=dept_name,
        jd_content=jd_content_str,
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

    # 将所有 AI 产物关联到新创建的岗位
    artifacts_result = await db.execute(
        select(PositionAIArtifact).where(
            PositionAIArtifact.recruitment_request_id == req.id
        )
    )
    for artifact in artifacts_result.scalars().all():
        artifact.position_id = position.id

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
    # 1) 已有的招聘需求
    q = select(RecruitmentRequest)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
    rows = (await db.execute(
        q.order_by(RecruitmentRequest.created_at.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()
    result = [_serialize_req(r, current) for r in rows]

    # 2) 把 positions 里还没有 recruitment_request 关联的岗位也列出来，
    #    标记 status="uninitiated"，让前端能看到并一键发起招聘。
    published_position_ids = await db.execute(
        select(RecruitmentRequest.position_id).where(
            RecruitmentRequest.position_id.isnot(None),
            RecruitmentRequest.status == "published",
        )
    )
    published_ids = {r[0] for r in published_position_ids.all()}

    # 已经在 recruitment_requests 中引用（但未发布或已关闭）的岗位，也要补充进来
    all_linked_position_ids = await db.execute(
        select(RecruitmentRequest.position_id).where(
            RecruitmentRequest.position_id.isnot(None)
        )
    )
    linked_ids = {r[0] for r in all_linked_position_ids.all()}

    # 查所有 positions，排除已经有已发布请求的
    all_positions = (await db.execute(
        select(Position).where(Position.id.notin_(linked_ids - published_ids)).order_by(Position.created_at.desc())
    )).scalars().all()

    # 构建已有请求中引用的 position_id → request_id 的映射，给未发布但已关联的请求提供入口
    linked_request_map = {}
    if all_linked_position_ids:
        linked_rows = await db.execute(
            select(RecruitmentRequest.position_id, RecruitmentRequest.id).where(
                RecruitmentRequest.position_id.isnot(None)
            )
        )
        linked_request_map = {str(r[0]): str(r[1]) for r in linked_rows.all()}

    result_position_ids = {
        str(r["positionId"]) for r in result
        if r.get("positionId") is not None
    }

    for pos in all_positions:
        pid = str(pos.id)
        # 跳过已有请求且已发布、或已在本次请求列表中的
        if pid in result_position_ids:
            continue
        # 跳过已有请求且已发布的
        if pid in published_ids:
            continue

        # 构建虚拟请求实体
        existing_req_id = linked_request_map.get(pid)
        result.append({
            "id": existing_req_id or pid,  # 有已有请求就用请求id，否则用岗位id（前端用作唯一标识）
            "positionName": pos.name,
            "departmentId": None,
            "headcount": 1,
            "salaryRange": pos.salary_range or "",
            "salaryMasked": False,
            "workExperience": pos.experience_requirement or "",
            "educationRequirement": pos.education_requirement or "",
            "jobDescription": pos.jd_content or "",
            "jobResponsibilities": pos.jd_responsibilities or "",
            "jobRequirements": pos.jd_requirements or "",
            "bonusItems": pos.jd_preferred or "",
            "coreTasks": [],
            "requiredSkills": [],
            "preferredSkills": [],
            "deliverableReq": "",
            "deliverableRequirements": "",
            "probationGoal": "",
            "probationGoals": "",
            "eliminationCriteria": "",
            "interviewerIds": [],
            "interviewers": [],
            "directManagerId": None,
            "directManager": "",
            "submitterId": None,
            "status": "uninitiated" if not existing_req_id else "draft",
            "aiDraft": None,
            "aiOutputs": None,
            "hrConfirmedBy": None,
            "hrConfirmedAt": None,
            "deptConfirmedBy": None,
            "deptConfirmedAt": None,
            "positionId": pid,
            "createdAt": pos.created_at.isoformat() if pos.created_at else None,
            "updatedAt": pos.updated_at.isoformat() if pos.updated_at else None,
        })

    return ok({
        "list": result,
        "total": total + len(result) - len(rows),
        "page": page,
        "pageSize": page_size,
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


@router.post("/positions/{position_id}/initiate-request")
async def initiate_request_from_position(
    position_id: str,
    current: CurrentUser = Depends(require_permission("recruitment_request:create")),
    db: AsyncSession = Depends(get_db),
):
    """从已有岗位一键发起招聘需求。使用岗位 JD 填充需求，HR 只需补少量必填项。"""
    try:
        pid = uuid.UUID(position_id)
    except ValueError:
        return not_found("岗位不存在")
    row = await db.execute(select(Position).where(Position.id == pid))
    pos = row.scalar_one_or_none()
    if pos is None:
        return not_found("岗位不存在")

    # 检查是否已有未发布的招聘需求
    existing = await db.execute(
        select(RecruitmentRequest).where(
            RecruitmentRequest.position_id == pid,
            RecruitmentRequest.status.in_(["draft", "ai_generated", "hr_confirmed", "dept_confirmed"]),
        )
    )
    if existing.scalar_one_or_none():
        return fail(409, "该岗位已有进行中的招聘需求，请勿重复发起")

    # 用岗位 JD 数据填充招聘需求
    req = RecruitmentRequest(
        position_name=pos.name,
        headcount=1,
        department_id=None,
        salary_range=pos.salary_range or "",
        work_experience=pos.experience_requirement or "",
        education_requirement=pos.education_requirement or "",
        job_description=pos.jd_content or "",
        job_responsibilities=pos.jd_responsibilities or "",
        job_requirements=pos.jd_requirements or "",
        bonus_items=pos.jd_preferred or "",
        core_tasks="",
        required_skills=[],
        preferred_skills=[],
        deliverable_req="",
        probation_goal="",
        elimination_criteria="",
        interviewer_ids=[],
        direct_manager_id=None,
        direct_manager_name="",
        submitter_id=current.id,
        status="draft",
        position_id=pos.id,
    )
    db.add(req)
    await db.flush()
    await db.refresh(req)
    await write_audit(db, actor=current.username, action=f"从岗位「{pos.name}」发起招聘需求", section="recruitment")
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


# ═══════════════════════════════════════════════
# AI 产物接口
# ═══════════════════════════════════════════════

def _serialize_artifact(a: PositionAIArtifact) -> dict:
    return {
        "id": str(a.id),
        "recruitmentRequestId": str(a.recruitment_request_id),
        "positionId": str(a.position_id) if a.position_id else None,
        "artifactType": a.artifact_type,
        "content": a.content,
        "version": a.version,
        "createdAt": iso_utc(a.created_at),
        "updatedAt": iso_utc(a.updated_at),
    }


@router.get("/recruitment-requests/{req_id}/ai-artifacts")
async def list_ai_artifacts(
    req_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取指定招聘需求的所有 AI 产物。"""
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    rows = await db.execute(
        select(PositionAIArtifact).where(
            PositionAIArtifact.recruitment_request_id == rid
        ).order_by(PositionAIArtifact.artifact_type)
    )
    artifacts = rows.scalars().all()
    return ok([_serialize_artifact(a) for a in artifacts])


@router.get("/recruitment-requests/{req_id}/ai-artifacts/{artifact_type}")
async def get_ai_artifact(
    req_id: str,
    artifact_type: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取指定招聘需求的某一类 AI 产物。"""
    if artifact_type not in ARTIFACT_TYPES:
        return fail(400, f"无效的产物类型: {artifact_type}，可选: {', '.join(ARTIFACT_TYPES)}")
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(
        select(PositionAIArtifact).where(
            PositionAIArtifact.recruitment_request_id == rid,
            PositionAIArtifact.artifact_type == artifact_type,
        )
    )
    artifact = row.scalar_one_or_none()
    if artifact is None:
        return not_found("未找到该类型的 AI 产物")
    return ok(_serialize_artifact(artifact))


# ── 按岗位 ID 获取 AI 产物（供面试/试用期等下游模块） ────────

@router.get("/positions/{position_id}/ai-artifacts")
async def list_position_ai_artifacts(
    position_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取指定岗位的所有 AI 产物（岗位发布后使用）。"""
    try:
        pid = uuid.UUID(position_id)
    except ValueError:
        return not_found("岗位不存在")
    rows = await db.execute(
        select(PositionAIArtifact).where(
            PositionAIArtifact.position_id == pid
        ).order_by(PositionAIArtifact.artifact_type)
    )
    artifacts = rows.scalars().all()
    return ok([_serialize_artifact(a) for a in artifacts])


@router.get("/positions/{position_id}/ai-artifacts/{artifact_type}")
async def get_position_ai_artifact(
    position_id: str,
    artifact_type: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取指定岗位的某一类 AI 产物。"""
    if artifact_type not in ARTIFACT_TYPES:
        return fail(400, f"无效的产物类型: {artifact_type}，可选: {', '.join(ARTIFACT_TYPES)}")
    try:
        pid = uuid.UUID(position_id)
    except ValueError:
        return not_found("岗位不存在")
    row = await db.execute(
        select(PositionAIArtifact).where(
            PositionAIArtifact.position_id == pid,
            PositionAIArtifact.artifact_type == artifact_type,
        )
    )
    artifact = row.scalar_one_or_none()
    if artifact is None:
        return not_found("未找到该类型的 AI 产物")
    return ok(_serialize_artifact(artifact))


# ── 岗位说明书 PDF 下载 ────────────────────────────────────────


@router.get("/recruitment-requests/{req_id}/job-description/pdf")
async def download_job_description_pdf(
    req_id: str,
    current: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """下载岗位说明书 PDF 文档。"""
    try:
        rid = uuid.UUID(req_id)
    except ValueError:
        return not_found("招聘需求不存在")
    row = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.id == rid))
    req = row.scalar_one_or_none()
    if req is None:
        return not_found("招聘需求不存在")
    if not req.ai_draft:
        return fail(400, "请先生成 AI 内容")

    # 支持两种数据结构：
    # 1. 新格式: ai_draft.jobDescription = { basicInfo: {...}, mission: "...", ... }
    # 2. 旧格式: ai_draft = { basicInfo: {...}, mission: "...", ... } (扁平结构)
    job_desc = req.ai_draft.get("jobDescription")

    if isinstance(job_desc, dict):
        # 新格式，jobDescription 是结构化对象
        pass
    elif isinstance(req.ai_draft.get("basicInfo"), dict):
        # 旧格式，ai_draft 本身就是扁平结构
        job_desc = req.ai_draft
    else:
        job_desc = None

    if not job_desc:
        return fail(400, "岗位说明书内容不存在，请点击「重新生成」生成新格式内容")

    # 如果是纯文本格式（非常旧的数据），转换为结构化格式
    if isinstance(job_desc, str):
        job_desc = {
            "basicInfo": {
                "positionName": req.position_name,
                "department": req.department_id or "—",
                "headcount": req.headcount,
                "reportTo": req.direct_manager_name or "—",
                "salaryRange": req.salary_range or "—",
                "workLocation": "—",
                "employmentType": "正式员工",
                "level": "—",
            },
            "mission": job_desc,
            "responsibilities": (req.job_responsibilities or "—").split("\n") if req.job_responsibilities else ["—"],
            "qualifications": {
                "required": (req.job_requirements or "—").split("\n") if req.job_requirements else ["—"],
                "preferred": (req.bonus_items or "—").split("\n") if req.bonus_items else ["—"],
            },
            "permissions": ["—"],
            "collaborations": {"internal": ["—"], "external": ["—"]},
            "workEnvironment": {
                "officeType": "办公室",
                "workingHours": "标准工作时间",
                "overtime": "视项目需要",
                "travel": "视岗位需要",
            },
            "kpi": req.probation_goal.split("\n") if req.probation_goal else ["—"],
            "careerPath": "—",
        }

    from app.services.job_description_pdf import generate_pdf_bytes
    from urllib.parse import quote

    pdf_bytes = generate_pdf_bytes(job_desc)
    filename = f"岗位说明书_{req.position_name}.pdf"
    # 对文件名进行 URL 编码以支持中文
    encoded_filename = quote(filename)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )
