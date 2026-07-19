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
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.database import get_db, async_session_factory
from app.models.agent_session import AgentSession, AgentMessage, AgentMaterial, AgentTask
from app.agent.tools import create_langchain_tools, create_langchain_tools_from_defs
from app.agent.intent_classifier import classify_intent, get_tool_defs_for_intent
from app.agent.supervisor_graph import classify_complexity_sync, ExecutionPlan, PlanStep
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from app.config import get_settings
from app.infrastructure import minio_storage
from app.api.recruitment.resumes import extract_text_from_file, parse_resume_with_llm
from app.api.ai.prompt import AGENT_CONFIGS, build_system_prompt
from app.api.ai.tool_executor import sse_event, execute_tool_call
import os

router = APIRouter(tags=["AI Agent"])
settings = get_settings()

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
)


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
        "createdAt": iso_utc(s.created_at),
        "updatedAt": iso_utc(s.updated_at),
    }


# ═══════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/ai-agent/overview")
async def get_ai_agent_overview(db: AsyncSession = Depends(get_db)):
    """Workspace overview."""
    # Active task
    task_result = await db.execute(
        select(AgentTask).where(AgentTask.status == "running").limit(1)
    )
    active_task = task_result.scalar_one_or_none()

    return {
        "code": 0,
        "message": "ok",
        "data": {
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
                {"key": "recruit", "label": "筛选", "count": 0, "active": True, "badgeTone": "green"},
                {"key": "interview", "label": "面试", "count": 0, "active": False, "badgeTone": "blue"},
                {"key": "training", "label": "试用", "count": 0, "active": False, "badgeTone": "gray"},
                {"key": "performance", "label": "绩效", "count": 0, "active": False, "badgeTone": "gray"},
            ],
            "stats": [
                {"key": "resumes", "label": "待处理简历", "value": 0, "hint": "今日新增", "hintTone": "up"},
                {"key": "interviews", "label": "待面试", "value": 0, "hint": "本周安排", "hintTone": "default"},
                {"key": "offers", "label": "待发offer", "value": 0, "hint": "审批中", "hintTone": "default"},
                {"key": "hours", "label": "AI节省工时", "value": "0h", "hint": "本月累计", "hintTone": "up"},
            ],
            "suggestions": [
                {"id": "s1", "priority": "P1", "title": "处理高匹配候选人", "description": "有3位候选人匹配度超过90分", "actionLabel": "查看详情"},
                {"id": "s2", "priority": "P2", "title": "安排面试", "description": "5位候选人等待一面安排", "actionLabel": "立即安排"},
            ],
            "teamDynamics": [],
            "teamOutput": {"completedTasks": 0, "savedHours": "0h"},
            "activeTask": {
                "title": active_task.title, "description": active_task.description or "",
                "progress": active_task.progress or 0, "elapsed": "刚刚开始",
            } if active_task else None,
            "collaborationSteps": [],
            "phaseResults": [],
        },
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


@router.post("/ai-agent/sessions")
async def create_session(
    db: AsyncSession = Depends(get_db),
):
    """Create a new empty chat session."""
    session = AgentSession(
        title="新对话",
        agent_id="genie",
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
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


class RenameSessionRequest(BaseModel):
    title: str


@router.patch("/ai-agent/sessions/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    """重命名对话（支持用户自定义对话名称）。"""
    new_title = (body.title or "").strip()
    if not new_title:
        return {"code": 400, "message": "对话名称不能为空", "data": None}
    if len(new_title) > 100:
        return {"code": 400, "message": "对话名称过长（最多 100 字）", "data": None}

    result = await db.execute(
        select(AgentSession).where(AgentSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"code": 404, "message": "对话不存在", "data": None}

    session.title = new_title
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
    # Get or create session
    session_result = await db.execute(
        select(AgentSession).order_by(desc(AgentSession.updated_at)).limit(1)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        session = AgentSession(title="新对话", agent_id="recruit")
        db.add(session)
        await db.flush()

    material_name = ""
    material_path = ""
    if file:
        file_ext = os.path.splitext(file.filename or "material")[1] or ".pdf"
        object_key = f"agent/{uuid.uuid4()}{file_ext}"
        content = await file.read()
        try:
            await asyncio.to_thread(
                minio_storage.upload_bytes, object_key, content, "application/octet-stream"
            )
        except Exception as e:
            return {"code": 500, "message": f"资料存储失败: {e}", "data": None}
        material_path = object_key
        material_name = file.filename or "material"
    elif knowledgeName:
        material_name = knowledgeName

    material = AgentMaterial(
        session_id=session.id,
        name=material_name,
        type=type,
        knowledge_id=knowledgeId,
        file_path=material_path,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    # If uploading a resume, trigger AI analysis
    analysis = None
    if type in ("resume", "file") and material_path:
        try:
            text, extract_error = extract_text_from_file(material_path)
            if text.strip():
                parsed, _ = await parse_resume_with_llm(text)
                analysis_data = parsed.get("analysis", {}) if parsed else {}
                # 维度评分由 5 个独立子 Agent 计算（教育背景走规则，其余走 LLM）。
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
                # Also extract basic candidate info
                analysis["candidateName"] = parsed.get("name", "")
                analysis["skills"] = parsed.get("skills", [])
        except Exception as e:
            print(f"AI Agent resume analysis error: {e}")

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(material.id),
            "name": material.name,
            "type": material.type,
            "knowledgeId": str(material.knowledge_id) if material.knowledge_id else None,
            "uploadedAt": material.uploaded_at.isoformat() if material.uploaded_at else "",
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

        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        plan_data = json.loads(raw)

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

    # ── Fetch attached materials and build context ──
    material_context = ""
    if material_ids:
        try:
            async with async_session_factory() as db:
                mat_result = await db.execute(
                    select(AgentMaterial).where(AgentMaterial.id.in_(material_ids))
                )
                attached_materials = mat_result.scalars().all()
            if attached_materials:
                lines = ["\n\n--- 附件资料 ---"]
                for mat in attached_materials:
                    lines.append(f"\n[{mat.type}] {mat.name}")
                    if mat.file_path:
                        try:
                            file_text, _ = extract_text_from_file(mat.file_path)
                            if file_text.strip():
                                truncated = file_text[:3000] + ("..." if len(file_text) > 3000 else "")
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
            history_rows.append((h.role, h.content or ""))

        await db.commit()
        session_obj_id = session.id

    # ── Build LangChain message history from plain data ──
    lc_messages = []
    for role, content in history_rows:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    lc_messages.append(HumanMessage(content=message + material_context))

    async def event_stream() -> AsyncGenerator[str, None]:
        result = AgentResult()
        task_id = None

        try:
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

            # ── Plan Generation (complex queries only) ──
            complexity = classify_complexity_sync(message)
            if complexity == "complex":
                plan = await _generate_plan(lc_messages, system_prompt, llm_client)
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

            graph = build_agent_graph(langchain_tools, system_prompt)

            # Stream agent execution
            async for sse_str in stream_agent_response(graph, lc_messages, result):
                if await request.is_disconnected():
                    break
                yield sse_str

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
                        title_resp = await llm_client.chat.completions.create(
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
