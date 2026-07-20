"""Genie AI Agent — API endpoints and SSE streaming.

The system prompt lives in ``prompt.py`` (rules + agent configs).
The tool executor lives in ``tool_executor.py`` (LLM tool → API bridge).
This module keeps only the HTTP layer: routes, session management, and
the SSE streaming endpoint.
"""

import json
import re
import uuid
import asyncio
from datetime import datetime, date
from typing import Optional, List, AsyncGenerator, Any
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.database import get_db, async_session_factory
from app.models.agent_session import AgentProject, AgentSession, AgentMessage, AgentMaterial, AgentTask
from app.models.recruitment import Candidate, Position
from app.agent.tools import create_langchain_tools, create_langchain_tools_from_defs
from app.agent.intent_classifier import classify_intent, get_tool_defs_for_intent
from app.agent.supervisor_graph import classify_complexity_sync, ExecutionPlan, PlanStep
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from app.config import get_settings
from app.middleware.trace import get_trace_id
from app.infrastructure import minio_storage
from app.api.recruitment.resumes import extract_text_from_file, parse_resume_with_llm
from app.api.ai.prompt import AGENT_CONFIGS, build_system_prompt
from app.api.ai.tool_executor import sse_event, execute_tool_call
from app.services.ai import get_llm_client, create_langchain_llm
import os
import tempfile

router = APIRouter(tags=["AI Agent"])
settings = get_settings()

# ── Agent OS singletons (memory / skills / quality) ───────────
# These are the genuinely useful pieces merged in from the former v2/v3
# "Agent OS" stack. Everything else (query_loop, streaming_executor,
# context engine, cache builder, orchestration, hooks, retry) was dead
# code and has been removed — this v1 endpoint is now the ONLY agent path.
from app.agent_os.memory.manager import MemoryManager
from app.agent_os.memory.retriever import MemoryRetriever
from app.agent_os.skills.registry import SkillRegistry
from app.agent_os.quality.guard import QualityGuard

_memory_manager = MemoryManager(memory_dir=settings.memory_dir)
_memory_retriever = MemoryRetriever(_memory_manager)
_skill_registry = SkillRegistry(skills_dir=settings.skills_dir)
_quality_guard = QualityGuard()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK ≈ 1 token/char, ASCII ≈ 0.25 token/char."""
    if not text:
        return 0
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk + (len(text) - cjk) // 4 + 1


def _estimate_tokens_msgs(messages: list) -> int:
    """Estimate total tokens across a list of messages."""
    return sum(_estimate_tokens(str(getattr(m, "content", "") or "")) for m in messages)


def _compact_messages(messages: list, max_tokens: int = 6000) -> tuple[list, int]:
    """Token-budget truncation: drop oldest messages when over budget.

    Returns (compacted_messages, n_dropped).
    Keeps at least the final message (current user turn). Simpler and
    safer than the old CompactionPipeline, which was coupled to LoopState
    and silently corrupted ToolMessage metadata.
    """
    if not messages:
        return messages, 0
    total = _estimate_tokens_msgs(messages)
    if total <= max_tokens:
        return messages, 0
    result = list(messages)
    dropped = 0
    # Drop from the front (oldest) but never drop the last message.
    while len(result) > 1 and total > max_tokens:
        removed = result.pop(0)
        total -= _estimate_tokens(str(getattr(removed, "content", "") or ""))
        dropped += 1
    return result, dropped


async def _verify_executor(tool_name: str, params: dict) -> str:
    """Executor wrapper for QualityGuard read-back verification.

    Matches the ``async (tool_name, params) -> str`` signature that
    WriteVerifier expects, using a fresh short-lived DB session.
    """
    from app.agent.tools import _execute_tool_sync
    try:
        return await _execute_tool_sync(tool_name, **params)
    except Exception as e:
        return f"验证查询失败: {e}"


def iso_utc(dt: Optional[datetime]) -> str:
    """Serialize naive UTC datetime with trailing Z so browsers parse correctly.

    ``datetime.utcnow()`` / DB columns are naive UTC. ``isoformat()`` alone
    yields ``2026-07-19T06:00:00`` which JS treats as *local* time — in CST
    (UTC+8) a just-created session then shows as 「8 小时前」.
    """
    if not dt:
        return ""
    text = dt.isoformat()
    if text.endswith("Z") or text.endswith("+00:00"):
        return text
    # Already has an offset like +08:00
    if len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        return text
    return text + "Z"


def serialize_session(s: AgentSession) -> dict:
    return {
        "id": str(s.id),
        "title": s.title or "新对话",
        "agentId": s.agent_id,
        "projectId": str(s.project_id) if s.project_id else None,
        "createdAt": iso_utc(s.created_at),
        "updatedAt": iso_utc(s.updated_at),
    }


def serialize_project(p: AgentProject, session_count: int = 0) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "sessionCount": session_count,
        "createdAt": iso_utc(p.created_at),
        "updatedAt": iso_utc(p.updated_at),
    }


# ═══════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════

async def _build_global_overview(db: AsyncSession) -> dict:
    """Workspace overview for all projects (global recruitment stats)."""
    job_hunting = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "job_hunting")
    )).scalar() or 0
    interview_count = (await db.execute(
        select(func.count()).select_from(Candidate).where(
            Candidate.status.in_(["passed", "first_interview", "second_interview"])
        )
    )).scalar() or 0
    passed_count = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "passed")
    )).scalar() or 0
    position_count = (await db.execute(select(func.count()).select_from(Position))).scalar() or 0

    task_result = await db.execute(
        select(AgentTask).where(AgentTask.status == "running").limit(1)
    )
    active_task = task_result.scalar_one_or_none()

    return {
        "projectId": None,
        "projectName": None,
        "agents": [
            {
                "id": aid, "name": cfg["name"], "description": cfg["description"],
                "status": "就绪" if aid != "genie" else "全能就绪",
                "tone": "active",
                "icon": cfg["icon"], "iconBg": cfg["iconBg"], "iconColor": cfg["iconColor"],
            }
            for aid, cfg in AGENT_CONFIGS.items()
        ],
        "workflowSteps": [
            {"key": "recruit", "label": "筛选", "count": int(job_hunting), "active": True, "badgeTone": "green"},
            {"key": "interview", "label": "面试", "count": int(interview_count), "active": interview_count > 0, "badgeTone": "blue"},
            {"key": "training", "label": "试用", "count": 0, "active": False, "badgeTone": "gray"},
            {"key": "performance", "label": "绩效", "count": 0, "active": False, "badgeTone": "gray"},
        ],
        "stats": [
            {"key": "resumes", "label": "待处理简历", "value": int(job_hunting), "hint": "求职中", "hintTone": "up"},
            {"key": "interviews", "label": "待面试", "value": int(interview_count), "hint": "流程中", "hintTone": "default"},
            {"key": "offers", "label": "待发offer", "value": int(passed_count), "hint": "已通过", "hintTone": "default"},
            {"key": "positions", "label": "在招岗位", "value": int(position_count), "hint": "持续招聘", "hintTone": "up"},
        ],
        "suggestions": [
            {"id": "s1", "priority": "P1", "title": "处理高匹配候选人", "description": "筛选并推进高匹配简历", "actionLabel": "查看详情"},
            {"id": "s2", "priority": "P2", "title": "安排面试", "description": "为已通过候选人安排面试", "actionLabel": "立即安排"},
        ],
        "teamDynamics": [],
        "teamOutput": {"completedTasks": 0, "savedHours": "0h"},
        "activeTask": {
            "title": active_task.title, "description": active_task.description or "",
            "progress": active_task.progress or 0, "elapsed": "刚刚开始",
        } if active_task else None,
        "collaborationSteps": [],
        "phaseResults": [],
    }


async def _build_project_overview(db: AsyncSession, project_id: str) -> dict | None:
    """Workspace overview scoped to one project folder."""
    result = await db.execute(select(AgentProject).where(AgentProject.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        return None

    session_rows = await db.execute(
        select(AgentSession.id).where(AgentSession.project_id == project_id)
    )
    session_ids = [row[0] for row in session_rows.all()]

    session_count = len(session_ids)
    message_count = 0
    material_count = 0
    resume_count = 0
    completed_tasks = 0
    active_task = None

    if session_ids:
        message_count = (await db.execute(
            select(func.count()).select_from(AgentMessage).where(
                AgentMessage.session_id.in_(session_ids)
            )
        )).scalar() or 0
        material_count = (await db.execute(
            select(func.count()).select_from(AgentMaterial).where(
                AgentMaterial.session_id.in_(session_ids)
            )
        )).scalar() or 0
        resume_count = (await db.execute(
            select(func.count()).select_from(AgentMaterial).where(
                AgentMaterial.session_id.in_(session_ids),
                AgentMaterial.type == "resume",
            )
        )).scalar() or 0
        completed_tasks = (await db.execute(
            select(func.count()).select_from(AgentTask).where(
                AgentTask.session_id.in_(session_ids),
                AgentTask.status == "completed",
            )
        )).scalar() or 0
        task_result = await db.execute(
            select(AgentTask)
            .where(AgentTask.session_id.in_(session_ids), AgentTask.status == "running")
            .limit(1)
        )
        active_task = task_result.scalar_one_or_none()

    base = await _build_global_overview(db)
    base["projectId"] = str(project.id)
    base["projectName"] = project.name
    base["workflowSteps"] = [
        {"key": "recruit", "label": "对话", "count": session_count, "active": True, "badgeTone": "green"},
        {"key": "interview", "label": "消息", "count": int(message_count), "active": message_count > 0, "badgeTone": "blue"},
        {"key": "training", "label": "资料", "count": int(material_count), "active": material_count > 0, "badgeTone": "gray"},
        {"key": "performance", "label": "简历", "count": int(resume_count), "active": resume_count > 0, "badgeTone": "gray"},
    ]
    base["stats"] = [
        {"key": "sessions", "label": "项目对话", "value": session_count, "hint": project.name, "hintTone": "default"},
        {"key": "messages", "label": "消息轮次", "value": int(message_count), "hint": "累计往返", "hintTone": "default"},
        {"key": "materials", "label": "上传资料", "value": int(material_count), "hint": "含简历/JD", "hintTone": "up"},
        {"key": "tasks", "label": "已完成任务", "value": int(completed_tasks), "hint": "本项目内", "hintTone": "default"},
    ]
    base["suggestions"] = [
        {
            "id": "p1",
            "priority": "P1",
            "title": f"在「{project.name}」继续推进",
            "description": f"当前 {session_count} 个对话、{resume_count} 份简历",
            "actionLabel": "新建对话",
        },
    ]
    base["teamOutput"] = {"completedTasks": int(completed_tasks), "savedHours": f"{int(message_count * 0.1)}h"}
    base["activeTask"] = {
        "title": active_task.title,
        "description": active_task.description or "",
        "progress": active_task.progress or 0,
        "elapsed": "进行中",
    } if active_task else None
    return base


@router.get("/ai-agent/overview")
async def get_ai_agent_overview(
    projectId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Workspace overview, optionally scoped to a project folder."""
    if projectId:
        data = await _build_project_overview(db, projectId)
        if data is None:
            return {"code": 404, "message": "项目不存在", "data": None}
    else:
        data = await _build_global_overview(db)

    return {
        "code": 0,
        "message": "ok",
        "data": data,
    }


@router.get("/ai-agent/welcome-prompts")
async def get_welcome_prompts(db: AsyncSession = Depends(get_db)):
    """欢迎页快捷入口推荐（只读缓存，对用户无感）。

    推荐由后台定时任务默默刷新（见 app lifespan 中的 welcome_prompts job），
    本接口绝不在请求路径上调用 LLM。缓存为空时返回内置默认推荐。
    """
    from app.services.ai.welcome_prompt_recommender import get_cached_welcome_prompts

    prompts = await get_cached_welcome_prompts(db)
    return {
        "code": 0,
        "message": "ok",
        "data": prompts,
    }


@router.get("/ai-agent/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentSession)
        .order_by(desc(AgentSession.updated_at))
        .limit(50)
    )
    sessions = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [serialize_session(s) for s in sessions],
    }


@router.get("/ai-agent/projects")
async def list_projects(
    db: AsyncSession = Depends(get_db),
):
    """List agent conversation projects with session counts."""
    from sqlalchemy import func

    count_subq = (
        select(
            AgentSession.project_id.label("project_id"),
            func.count(AgentSession.id).label("session_count"),
        )
        .where(AgentSession.project_id.isnot(None))
        .group_by(AgentSession.project_id)
        .subquery()
    )
    result = await db.execute(
        select(AgentProject, count_subq.c.session_count)
        .outerjoin(count_subq, AgentProject.id == count_subq.c.project_id)
        .order_by(desc(AgentProject.updated_at))
    )
    rows = result.all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            serialize_project(project, int(session_count or 0))
            for project, session_count in rows
        ],
    }


class CreateProjectRequest(BaseModel):
    name: str


@router.post("/ai-agent/projects")
async def create_project(
    body: CreateProjectRequest,
    db: AsyncSession = Depends(get_db),
):
    name = (body.name or "").strip()
    if not name:
        return {"code": 400, "message": "项目名称不能为空", "data": None}
    if len(name) > 64:
        return {"code": 400, "message": "项目名称过长（最多 64 字）", "data": None}

    project = AgentProject(name=name)
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return {
        "code": 0,
        "message": "ok",
        "data": serialize_project(project, 0),
    }


class RenameProjectRequest(BaseModel):
    name: str


@router.patch("/ai-agent/projects/{project_id}")
async def rename_project(
    project_id: str,
    body: RenameProjectRequest,
    db: AsyncSession = Depends(get_db),
):
    name = (body.name or "").strip()
    if not name:
        return {"code": 400, "message": "项目名称不能为空", "data": None}
    if len(name) > 64:
        return {"code": 400, "message": "项目名称过长（最多 64 字）", "data": None}

    result = await db.execute(
        select(AgentProject).where(AgentProject.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        return {"code": 404, "message": "项目不存在", "data": None}

    project.name = name
    project.updated_at = datetime.utcnow()
    await db.flush()
    return {
        "code": 0,
        "message": "ok",
        "data": serialize_project(project),
    }


@router.delete("/ai-agent/projects/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentProject).where(AgentProject.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        return {"code": 404, "message": "项目不存在", "data": None}

    # Sessions become ungrouped; FK ondelete=SET NULL also handles this.
    await db.delete(project)
    return {"code": 0, "message": "ok", "data": None}


class CreateSessionRequest(BaseModel):
    projectId: Optional[str] = None


@router.post("/ai-agent/sessions")
async def create_session(
    body: Optional[CreateSessionRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """Create a new empty chat session."""
    project_id = None
    if body and body.projectId:
        proj_result = await db.execute(
            select(AgentProject).where(AgentProject.id == body.projectId)
        )
        project = proj_result.scalar_one_or_none()
        if not project:
            return {"code": 404, "message": "项目不存在", "data": None}
        project_id = project.id

    session = AgentSession(
        title="新对话",
        agent_id="genie",
        project_id=project_id,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)

    if project_id:
        proj = await db.get(AgentProject, project_id)
        if proj:
            proj.updated_at = datetime.utcnow()

    return {
        "code": 0,
        "message": "ok",
        "data": serialize_session(session),
    }


@router.delete("/ai-agent/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentSession).where(
            AgentSession.id == session_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"code": 404, "message": "对话不存在", "data": None}

    # Clean up MinIO objects for the session's materials before cascade delete
    mat_result = await db.execute(
        select(AgentMaterial).where(AgentMaterial.session_id == session_id)
    )
    for mat in mat_result.scalars().all():
        if mat.file_path and not (os.path.isabs(mat.file_path)):
            await asyncio.to_thread(minio_storage.delete_object, mat.file_path)

    await db.delete(session)
    return {"code": 0, "message": "ok", "data": None}


class PatchSessionRequest(BaseModel):
    title: Optional[str] = None
    projectId: Optional[str] = None


@router.patch("/ai-agent/sessions/{session_id}")
async def patch_session(
    session_id: str,
    body: PatchSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    """更新对话名称或所属项目。"""
    result = await db.execute(
        select(AgentSession).where(AgentSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"code": 404, "message": "对话不存在", "data": None}

    if body.title is not None:
        new_title = body.title.strip()
        if not new_title:
            return {"code": 400, "message": "对话名称不能为空", "data": None}
        if len(new_title) > 100:
            return {"code": 400, "message": "对话名称过长（最多 100 字）", "data": None}
        session.title = new_title

    if body.projectId is not None:
        if body.projectId == "":
            session.project_id = None
        else:
            proj_result = await db.execute(
                select(AgentProject).where(AgentProject.id == body.projectId)
            )
            project = proj_result.scalar_one_or_none()
            if not project:
                return {"code": 404, "message": "项目不存在", "data": None}
            session.project_id = project.id
            project.updated_at = datetime.utcnow()

    session.updated_at = datetime.utcnow()
    await db.flush()
    return {
        "code": 0,
        "message": "ok",
        "data": serialize_session(session),
    }


@router.get("/ai-agent/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentMessage)
        .where(AgentMessage.session_id == session_id)
        .order_by(AgentMessage.created_at)
    )
    messages = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(m.id),
                "sessionId": str(m.session_id),
                "role": m.role,
                "content": m.content,
                "createdAt": iso_utc(m.created_at),
            }
            for m in messages
        ],
    }


@router.post("/ai-agent/materials")
async def upload_material(
    type: str = Form(...),
    file: UploadFile = File(None),
    knowledgeId: str = Form(None),
    knowledgeName: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a material (resume/document/knowledge) for the current session.

    Resumes (type="resume" or "file") are auto-detected: if the content parses
    as a real resume the file goes through the same full ingestion pipeline as
    the dedicated POST /api/resumes/upload endpoint — MinIO storage, position
    matching, Candidate creation, parsing, scoring, RAG ingest.  Non-resume
    files are stored as session attachments with a plain AI analysis preview.
    """
    # Get or create session
    session_result = await db.execute(
        select(AgentSession).order_by(desc(AgentSession.updated_at)).limit(1)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        session = AgentSession(title="新对话", agent_id="recruit")
        db.add(session)
        await db.flush()

    # ── Knowledge-base picker (no file) ─────────────────────
    if not file:
        material_name = knowledgeName or "知识库素材"
        material = AgentMaterial(
            session_id=session.id, name=material_name,
            type=type, knowledge_id=knowledgeId, file_path="",
        )
        db.add(material)
        await db.flush()
        await db.refresh(material)
        return {
            "code": 0, "message": "ok",
            "data": {
                "id": str(material.id), "name": material.name, "type": material.type,
                "uploadedAt": material.uploaded_at.isoformat() if material.uploaded_at else "",
                "analysis": None,
            },
        }

    # ── File upload: detect whether this is a real resume ───
    filename = file.filename or "material"
    file_ext = os.path.splitext(filename)[1].lower() or ".pdf"
    content = await file.read()

    # Save to a temp file on disk so extract_text_from_file can read it.
    # (extract_text_from_file handles both local paths and MinIO keys;
    #  we give it a local temp so we can decide the final MinIO prefix
    #  after detection without uploading twice.)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tf:
        tf.write(content)
        tmp_path = tf.name

    try:
        text, extract_error = extract_text_from_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    is_resume = False
    parsed = {}
    if text and text.strip():
        try:
            parsed, _ = await parse_resume_with_llm(text)
            name = (parsed.get("name") or "").strip()
            education = (parsed.get("education") or "").strip()
            skills = parsed.get("skills") or []
            # Heuristic: must have a plausible Chinese name (≥2 chars,
            # no pure ASCII) AND at least one of education/skills/experience
            # with a real value.
            has_real_name = (
                name and name != "未知"
                and len(re.sub(r"[a-zA-Z\s]", "", name)) >= 2
            )
            has_real_edu = education and education not in ("未知", "其他")
            has_real_skills = bool(skills) and any(
                s and s != "未知" for s in skills
            )
            exp = (parsed.get("experience") or "").strip()
            has_real_exp = exp and exp not in ("未知", "应届", "在校中")
            if has_real_name and (has_real_edu or has_real_skills or has_real_exp):
                is_resume = True
        except Exception:
            pass

    if is_resume and type in ("resume", "file"):
        # ── Full DB ingestion (same as dedicated endpoint) ──

        # 1) Store to MinIO with proper resumes/ prefix
        object_key = f"resumes/{uuid.uuid4()}{file_ext}"
        try:
            await asyncio.to_thread(
                minio_storage.upload_bytes, object_key, content, "application/octet-stream"
            )
        except Exception as e:
            return {"code": 500, "message": f"简历存储失败: {e}", "data": None}

        # 2) AI position matching
        from app.models.recruitment import Candidate, Position
        from app.services.recruitment.position_matcher import match_position_for_resume
        from app.services.system.system_settings import get_system_setting

        resolved_position_id = ""
        position_name = ""
        match_reason = ""
        position_source = "user"
        position: Optional[Any] = None

        resolved_position_id, position_name, match_reason = (
            await match_position_for_resume(db, text)
        )
        if resolved_position_id:
            position_source = "agent"
            pos_result = await db.execute(
                select(Position).where(Position.id == resolved_position_id)
            )
            position = pos_result.scalar_one_or_none()

        if not position:
            # Fallback to system default position
            fallback_id = str(await get_system_setting(db, "defaultPositionId", "") or "")
            if fallback_id:
                pos_result = await db.execute(
                    select(Position).where(Position.id == fallback_id)
                )
                position = pos_result.scalar_one_or_none()
                if position:
                    resolved_position_id = fallback_id
                    position_source = "default"
                    position_name = position.name
                    match_reason = match_reason or "AI 匹配未命中，已回退到系统默认岗位"

        if not position:
            # Can't match any position — clean up and fail
            try:
                await asyncio.to_thread(minio_storage.delete_object, object_key)
            except Exception:
                pass
            return {
                "code": 400,
                "message": (
                    f"AI 未能判断该简历的应聘岗位：{match_reason or '无匹配'}。"
                    "请手动选择岗位后重新上传。"
                ),
                "data": None,
            }

        # 3) Create Candidate record
        candidate = Candidate(
            name=parsed.get("name") or filename,
            position_id=resolved_position_id,
            status="job_hunting",
            resume_file=object_key,
            upload_time=date.today(),
        )
        db.add(candidate)
        await db.flush()

        # 4) Full parse + scoring
        from app.api.recruitment.resumes import load_candidate, run_resume_parse
        from app.services.resume_scoring import score_all
        candidate = await load_candidate(db, candidate.id)
        if candidate:
            parse_msg = await run_resume_parse(candidate, position_name, db)
            # Score (may already be done by run_resume_parse; score_all is
            # idempotent — calls the same sub-agents but won't overwrite if
            # dimensions already populated)
            try:
                dims = await score_all(text, parsed, position_name)
                if dims and candidate.ai_analysis:
                    candidate.ai_analysis.dimensions = dims
            except Exception:
                pass

        # 5) RAG knowledge-base ingest
        try:
            from app.services.recruitment.resume_kb import ingest_resume_to_kb
            await ingest_resume_to_kb(
                db, content=content, file_name=filename,
                source_object_key=object_key, candidate_id=str(candidate.id),
            )
        except Exception:
            pass

        # 6) Notifications
        try:
            from app.services.system.notification import notify_if
            await notify_if(
                db, "notifyNewResume", "new_resume",
                f"新简历入库：{filename}（岗位：{position_name}）",
                {"candidateId": str(candidate.id)},
            )
        except Exception:
            pass

        await db.flush()

        # 7) Reload with relationships for response
        candidate = await load_candidate(db, candidate.id)
        resp_data = {}
        if candidate:
            resp_data = {
                "id": str(candidate.id),
                "name": candidate.name or filename,
                "position": position_name,
                "positionId": resolved_position_id,
                "score": candidate.score or 0,
                "status": candidate.status or "job_hunting",
                "skills": [s.skill for s in (candidate.skills or [])],
            }

        # 8) Create AgentMaterial referencing the ingested candidate
        material = AgentMaterial(
            session_id=session.id, name=filename, type="resume",
            file_path=object_key,
        )
        db.add(material)
        await db.flush()
        await db.refresh(material)

        return {
            "code": 0,
            "message": "简历已入库",
            "data": {
                "id": str(material.id),
                "name": material.name,
                "type": "resume",
                "uploadedAt": material.uploaded_at.isoformat() if material.uploaded_at else "",
                "ingested": True,
                **resp_data,
            },
        }

    # ── Not a resume (or type=knowledge) → old attachment flow ──
    object_key = f"agent/{uuid.uuid4()}{file_ext}"
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return {"code": 500, "message": f"资料存储失败: {e}", "data": None}

    material = AgentMaterial(
        session_id=session.id, name=filename, type=type,
        knowledge_id=knowledgeId, file_path=object_key,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    # AI analysis preview
    analysis = None
    if type in ("resume", "file") and text and text.strip():
        try:
            analysis_data = parsed.get("analysis", {}) if parsed else {}
            from app.services.resume_scoring import score_all
            dimensions = await score_all(text, parsed or {}, "")
            analysis = {
                "overallScore": analysis_data.get("overallScore"),
                "summary": analysis_data.get("summary", ""),
                "keywords": analysis_data.get("keywords", []),
                "dimensions": dimensions,
                "highlights": analysis_data.get("highlights", []),
                "risks": analysis_data.get("risks", []),
                "recommendation": analysis_data.get("recommendation", ""),
                "positionMatch": analysis_data.get("positionMatch", ""),
                "experienceInsight": analysis_data.get("experienceInsight", ""),
            }
            analysis["candidateName"] = parsed.get("name", "")
            analysis["skills"] = parsed.get("skills", [])
        except Exception:
            pass

    return {
        "code": 0, "message": "ok",
        "data": {
            "id": str(material.id), "name": material.name, "type": material.type,
            "knowledgeId": str(material.knowledge_id) if material.knowledge_id else None,
            "uploadedAt": material.uploaded_at.isoformat() if material.uploaded_at else "",
            "ingested": False,
            "analysis": analysis,
        },
    }


# ── Plan Generation Helper ────────────────────────────────

PLAN_SYSTEM_PROMPT = """你是一个任务规划器。将用户的请求分解为有序的执行步骤。
只返回一个 JSON 对象（不要 markdown 代码块，不要解释）：

{
  "title": "简短的任务标题（≤15字）",
  "steps": [
    {
      "index": 1,
      "title": "步骤名称（≤10字）",
      "description": "这个步骤要做什么（一句话）",
      "expected_tools": ["工具名"],
      "expected_outcome": "成功标准"
    }
  ]
}

规则：
- 每个步骤只调用 1-2 个工具
- 步骤总数 1-5 个，越少越好
- 步骤顺序要合理（先查询再操作）
- 简单查询只返回 1 个步骤
- expected_tools 使用英文工具名"""


async def _generate_plan(
    lc_messages: list,
    system_prompt: str,
    client: AsyncOpenAI,
) -> Optional[ExecutionPlan]:
    """Generate an execution plan for complex user requests.

    Uses a focused LLM call. Falls back to None on any error so the agent
    can still proceed without a plan.
    """
    try:
        # Get user message content
        user_content = ""
        for msg in reversed(lc_messages):
            if hasattr(msg, "content") and not isinstance(msg, SystemMessage):
                user_content = str(msg.content)[:500]
                break

        if not user_content:
            return None

        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()

        # Use shared JSON extraction (handles markdown fences + trailing commas)
        from app.utils.json_utils import extract_json_from_text
        clean = extract_json_from_text(raw)
        plan_data = json.loads(clean)

        if "steps" not in plan_data or not plan_data["steps"]:
            return None

        steps = [
            PlanStep(
                index=s.get("index", i + 1),
                title=s.get("title", f"步骤{i+1}"),
                description=s.get("description", ""),
                expected_tools=s.get("expected_tools", []),
                expected_outcome=s.get("expected_outcome", ""),
            )
            for i, s in enumerate(plan_data["steps"])
        ]

        return ExecutionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            title=plan_data.get("title", "执行计划"),
            steps=steps,
            total_estimated_tools=sum(len(s.expected_tools) for s in steps),
        )
    except Exception:
        return None


@router.post("/ai-agent/suggestions/{suggestion_id}/trigger")
async def trigger_suggestion(
    suggestion_id: str,
    db: AsyncSession = Depends(get_db),
):
    return {"code": 0, "message": "ok", "data": None}


@router.post("/ai-agent/chat")
async def agent_chat(
    request: Request,
    body: dict,
):
    """Main SSE streaming chat endpoint.

    This endpoint deliberately does NOT use ``Depends(get_db)``. Holding a DB
    session for the whole streaming duration leaks connections: if the client
    disconnects while the generator is blocked inside a long LLM call or a
    sync tool call, the generator cannot be cancelled and the session is
    never returned to the pool. After a few such leaks the PostgreSQL pool is
    exhausted and every other API call hangs ("前端点几次就收不到请求").

    Instead, all DB work is done in short-lived sessions (``async with
    async_session_factory()``) that are committed and closed immediately, so
    no connection is held while the SSE stream is open.
    """
    message = body.get("message", "")
    session_id = body.get("sessionId")
    agent_id = body.get("agentId", "genie")
    mentioned_agent_ids = body.get("mentionedAgentIds", [])
    material_ids = body.get("materialIds", [])

    # ── Fetch attached materials and build compact context ────────
    # Ingested resumes (file_path starts with "resumes/") are already in the
    # candidates table — give the AI a compact reference with the candidate ID
    # so it can use get_resume / update_resume instead of reprocessing raw text.
    material_context = ""
    if material_ids:
        try:
            async with async_session_factory() as db:
                mat_result = await db.execute(
                    select(AgentMaterial).where(AgentMaterial.id.in_(material_ids))
                )
                attached_materials = mat_result.scalars().all()
            if attached_materials:
                from app.models.recruitment import Candidate
                lines = ["\n\n--- 附件资料 ---"]
                for mat in attached_materials:
                    fp = (mat.file_path or "")
                    # ── Ingested resume → compact candidate reference ──
                    if mat.type in ("resume", "file") and fp.startswith("resumes/"):
                        lines.append(f"\n[已入库简历] {mat.name}")
                        try:
                            async with async_session_factory() as db2:
                                cand_result = await db2.execute(
                                    select(Candidate).where(
                                        Candidate.resume_file == fp
                                    ).limit(1)
                                )
                                c = cand_result.scalar_one_or_none()
                            if c:
                                skills_str = ", ".join(
                                    s.skill for s in (c.skills or [])
                                )[:80] or "无"
                                lines.append(
                                    f"候选人「{c.name}」(ID: {c.id}) | "
                                    f"岗位ID: {c.position_id} | 匹配分: {c.score or 0} | "
                                    f"状态: {c.status} | 技能: {skills_str}\n"
                                    f"（已入库，请用 get_resume / update_resume 操作，勿当作文本重复解析）"
                                )
                            else:
                                lines.append("（已入库，但未找到候选人记录）")
                        except Exception:
                            lines.append("（已入库，查找候选人信息失败）")
                    # ── Knowledge picker ──
                    elif mat.type == "knowledge":
                        lines.append(f"\n[知识库素材] {mat.name}")
                        if mat.knowledge_id:
                            lines.append(f"（knowledgeId: {mat.knowledge_id}）")
                    # ── Other file → extract text (clipped) ──
                    elif fp:
                        lines.append(f"\n[{mat.type}] {mat.name}")
                        try:
                            file_text, _ = extract_text_from_file(fp)
                            if file_text.strip():
                                truncated = file_text[:2000] + ("..." if len(file_text) > 2000 else "")
                                lines.append(f"内容:\n{truncated}")
                        except Exception:
                            pass
                material_context = "\n".join(lines)
        except Exception:
            pass

    # ── Get or create session, save user message, fetch history ──
    is_new_session = False
    session_title = "新对话"
    history_rows: list[tuple[str, str]] = []

    async with async_session_factory() as db:
        session = None
        if session_id:
            try:
                result = await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                session = result.scalar_one_or_none()
            except Exception:
                session = None

        if not session:
            is_new_session = True
            session = AgentSession(
                title="新对话",
                agent_id="genie",
            )
            db.add(session)
            await db.flush()
            session_id = str(session.id)

        # Save user message
        user_msg = AgentMessage(
            session_id=session.id,
            role="user",
            content=message,
        )
        db.add(user_msg)
        await db.flush()

        # Update session title
        if session.title in ("新对话", None, ""):
            session.title = message[:50] if message else "新对话"
        session_title = session.title or "新对话"

        # Fetch conversation history (last 20)
        history_result = await db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session.id)
            .order_by(AgentMessage.created_at)
            .limit(20)
        )
        history = history_result.scalars().all()
        for h in history[:-1]:  # exclude the just-saved user message
            history_rows.append((h.role, h.content or "", h.tool_blocks or []))

        await db.commit()
        session_obj_id = session.id

    # ── Build LangChain message history from plain data ──
    # Previous-turn tool calls are re-attached so the model can reference IDs
    # returned earlier (e.g. the position UUID from get_position → update_position).
    # Each reconstructed AIMessage with tool_calls MUST be immediately followed
    # by a matching ToolMessage per tool_call_id, or the OpenAI API rejects the
    # request — so we only reconstruct blocks that actually have a result.
    from langchain_core.messages import ToolMessage
    lc_messages = []
    for role, content, tool_blocks in history_rows:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            resolved_blocks = [
                tb for tb in (tool_blocks or [])
                if isinstance(tb, dict) and tb.get("id") and tb.get("result")
            ]
            if resolved_blocks:
                # 1 tool_call per AIMessage, each immediately answered by its
                # ToolMessage — guarantees a valid, well-paired message list.
                first = True
                for tb in resolved_blocks:
                    lc_messages.append(AIMessage(
                        content=content if first else "",
                        tool_calls=[{
                            "id": tb["id"],
                            "name": tb.get("name", ""),
                            "args": tb.get("params") or tb.get("args") or {},
                            "type": "tool_call",
                        }],
                    ))
                    lc_messages.append(ToolMessage(
                        content=str(tb["result"])[:1000],
                        tool_call_id=tb["id"],
                        name=tb.get("name", ""),
                    ))
                    first = False
            elif content:
                lc_messages.append(AIMessage(content=content))
    lc_messages.append(HumanMessage(content=message + material_context))

    async def event_stream() -> AsyncGenerator[str, None]:
        result = AgentResult()
        task_id = None

        try:
            # Send trace ID so the frontend can correlate logs
            trace_id = get_trace_id()
            if trace_id:
                yield sse_event("meta", {"traceId": trace_id})

            # Send initial thinking
            thinking_text = f"收到任务，正在作为{AGENT_CONFIGS.get(agent_id, {}).get('name', 'AI Agent')}分析您的指令..."
            yield sse_event("thinking", {"text": thinking_text, "append": False})

            # Sub-agent handoffs
            for mid in mentioned_agent_ids:
                agent_info = AGENT_CONFIGS.get(mid, {})
                yield sse_event("agent_handoff", {
                    "from": AGENT_CONFIGS.get(agent_id, {}).get("name", "主Agent"),
                    "to": agent_info.get("name", mid),
                    "reason": f"需要{agent_info.get('description', '协作处理')}",
                })

            # Create task for tracking (short-lived session)
            async with async_session_factory() as db:
                task = AgentTask(
                    session_id=session_obj_id,
                    title=message[:50] if message else "处理中",
                    description=message,
                    progress=0,
                    status="running",
                    started_at=datetime.utcnow(),
                )
                db.add(task)
                await db.flush()
                task_id = task.id
                await db.commit()

            # ── Intent Pre-Classification ──
            # Classify user intent to restrict available tools (hard constraint).
            # Uses keyword matching — fast, deterministic, no API call.
            intent = classify_intent(message)

            # Build LangGraph agent with intent-gated tools
            if intent == "general":
                langchain_tools = create_langchain_tools(agent_id)
            else:
                restricted_defs = get_tool_defs_for_intent(intent)
                langchain_tools = create_langchain_tools_from_defs(restricted_defs)

            system_prompt = build_system_prompt(agent_id)
            if intent != "general":
                INTENT_LABELS = {
                    "position_query": "岗位JD查询", "candidate_query": "候选人查询",
                    "candidate_action": "候选人操作", "interview": "面试管理",
                    "probation": "试用期考核", "performance": "绩效管理",
                    "knowledge": "知识库", "dashboard": "数据看板", "settings": "系统设置",
                }
                hint = INTENT_LABELS.get(intent, intent)
                system_prompt += (
                    f"\n\n[Intent Gate] 当前意图: {hint}。"
                    f"本轮仅可使用与该意图匹配的工具，禁止调用无关工具。"
                )

            # ── Memory retrieval + Skill matching (Agent OS merge) ──
            memories = []
            matched_skills = []
            try:
                memories = await _memory_retriever.retrieve(
                    query=message, llm_client=get_llm_client(), limit=5
                )
            except Exception:
                memories = []
            try:
                matched_skills = await _skill_registry.match(
                    message, llm_client=get_llm_client(), limit=3
                )
            except Exception:
                matched_skills = []

            if memories:
                yield sse_event("memory_loaded", {
                    "count": len(memories),
                    "memories": [
                        {"name": m.name, "description": m.description,
                         "type": m.memory_type, "scope": m.scope}
                        for m in memories
                    ],
                })
                mem_lines = ["## 相关记忆（供参考，不要照搬）\n"]
                for m in memories:
                    mem_lines.append(f"- **{m.description}**: {m.content[:300]}")
                system_prompt += "\n\n" + "\n".join(mem_lines)

            if matched_skills:
                system_prompt = _skill_registry.inject_skills(system_prompt, matched_skills)

            # ── Plan Generation (complex queries only) ──
            complexity = classify_complexity_sync(message)
            if complexity == "complex":
                plan = await _generate_plan(lc_messages, system_prompt, get_llm_client())
                if plan and plan.steps:
                    yield sse_event("plan_proposal", plan.to_sse_dict())
                    # Inject plan into system prompt for worker agent
                    steps_text = "\n".join(
                        f"  {s.index}. {s.title}: {s.description}"
                        for s in plan.steps
                    )
                    system_prompt += (
                        f"\n\n[执行计划] {plan.title}\n{steps_text}\n"
                        f"按步骤顺序执行。每完成一步，检查结果后再进行下一步。"
                    )

            # ── Context compaction: keep history within token budget ──
            compacted, dropped = _compact_messages(lc_messages, max_tokens=6000)
            if dropped > 0:
                yield sse_event("context_compressed", {
                    "layer": 1,
                    "tokenEstimate": _estimate_tokens_msgs(compacted),
                    "message": f"上下文较长，已省略最早的 {dropped} 条历史消息",
                })

            graph = build_agent_graph(langchain_tools, system_prompt)

            # Stream agent execution
            async for sse_str in stream_agent_response(graph, compacted, result):
                if await request.is_disconnected():
                    break
                yield sse_str

            # ── Quality Guard: verify writes actually took effect ──
            try:
                if result.tool_blocks:
                    guard_result = await _quality_guard.guard(
                        user_message=message,
                        agent_response=result.full_content,
                        tool_calls=result.tool_blocks,
                        tool_results=[tb.get("result", "") for tb in result.tool_blocks],
                        tool_executor=_verify_executor,
                    )
                    if not guard_result.passed:
                        yield sse_event("verification", {
                            "verified": False,
                            "issues": guard_result.issues[:5],
                        })
                    else:
                        yield sse_event("verification", {"verified": True, "issues": []})
            except Exception:
                pass

            # Save assistant message + update task (short-lived session)
            async with async_session_factory() as db:
                assistant_msg = AgentMessage(
                    session_id=session_obj_id,
                    role="assistant",
                    content=result.full_content,
                    thinking=None,
                    tool_blocks=result.tool_blocks if result.tool_blocks else None,
                )
                db.add(assistant_msg)
                await db.flush()

                # AI-generated session title
                if is_new_session or session_title == "新对话":
                    try:
                        title_prompt = (
                            f"根据以下对话内容，生成一个简短的标题（10个字以内，不要引号）：\n"
                            f"用户：{message[:200]}\nAI：{result.full_content[:200]}"
                        )
                        title_resp = await get_llm_client().chat.completions.create(
                            model=settings.deepseek_model,
                            messages=[{"role": "user", "content": title_prompt}],
                            temperature=0.7,
                            max_tokens=32,
                        )
                        new_title = title_resp.choices[0].message.content.strip().strip('"').strip("'")
                        if new_title and len(new_title) > 1:
                            sess_result = await db.execute(
                                select(AgentSession).where(AgentSession.id == session_obj_id)
                            )
                            sess = sess_result.scalar_one_or_none()
                            if sess:
                                sess.title = new_title[:50]
                                await db.flush()
                    except Exception:
                        pass

                # Update task status
                if task_id is not None:
                    task_result = await db.execute(select(AgentTask).where(AgentTask.id == task_id))
                    task_obj = task_result.scalar_one_or_none()
                    if task_obj:
                        task_obj.status = "done"
                        task_obj.progress = 100
                        task_obj.finished_at = datetime.utcnow()

                await db.commit()

            # ── Fire-and-forget: capture session learnings into memory ──
            # Runs in the background so it never blocks the response. Degrades
            # to a no-op if the LLM client or extraction fails.
            async def _auto_capture():
                try:
                    captured = await _memory_manager.auto_capture(
                        session_messages=lc_messages + [AIMessage(content=result.full_content)],
                        session_id=str(session_obj_id),
                        llm_client=get_llm_client(),
                    )
                except Exception:
                    captured = []
            asyncio.create_task(_auto_capture())

            yield sse_event("phase_result", {
                "id": f"phase_{uuid.uuid4().hex[:6]}",
                "tone": "success",
                "title": "任务完成",
                "description": message[:80] + ("..." if len(message) > 80 else ""),
            })

            yield sse_event("done", {})

        except asyncio.CancelledError:
            # Client disconnected — mark the task cancelled, then re-raise.
            # Shield the DB session from cancellation so the connection always
            # goes back to the pool instead of being GC'd.
            try:
                db2 = async_session_factory()
                try:
                    if task_id is not None:
                        task_result = await asyncio.shield(
                            db2.execute(select(AgentTask).where(AgentTask.id == task_id))
                        )
                        task_obj = task_result.scalar_one_or_none()
                        if task_obj:
                            task_obj.status = "cancelled"
                            task_obj.finished_at = datetime.utcnow()
                    await asyncio.shield(db2.commit())
                except BaseException:
                    await asyncio.shield(db2.rollback())
                    raise
                finally:
                    await asyncio.shield(db2.close())
            except Exception:
                pass
            raise
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Plan confirmation (formerly /agent-os/confirm-plan) ───────
# Moved here so the frontend PlanConfirmCard keeps working without any
# change, while we remove the entire agent_chat_v2 / Agent OS dead stack.
# The plan_coordinator module is intentionally NOT imported — we do not
# need the full coordinator state machine; the endpoint only needs to
# acknowledge the user's decision (approve/reject/modify) and the current
# LangGraph turn will simply continue without pausing on plan approval.

@router.post("/agent-os/confirm-plan")
async def confirm_plan(body: dict):
    """Acknowledge a plan proposal from the frontend.

    The plan was already injected into the system prompt before the LangGraph
    run, so the agent proceeds regardless. This endpoint simply returns success
    so PlanConfirmCard can dismiss its UI state without an error.
    """
    plan_id = (body.get("plan_id") or "").strip()
    action  = (body.get("action")  or "approve").strip()
    if not plan_id:
        return {"code": 400, "message": "plan_id 是必填字段", "data": None}
    return {
        "code": 0,
        "message": f"计划 {plan_id} 已{action}",
        "data": {"plan_id": plan_id, "action": action},
    }
