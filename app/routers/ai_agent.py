import json
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
from app.models.agent import AgentSession, AgentMessage, AgentMaterial, AgentTask
from app.agent.tools import create_langchain_tools
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from app.config import get_settings
from app.routers.resumes import extract_text_from_file, parse_resume_with_llm
import os

router = APIRouter(tags=["AI Agent"])
settings = get_settings()

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
)

os.makedirs(settings.upload_dir, exist_ok=True)

# Agent configurations
AGENT_CONFIGS = {
    "recruit": {
        "name": "招聘 Agent",
        "description": "负责简历解析、人才筛选、岗位匹配",
        "icon": "search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "interview": {
        "name": "面试 Agent",
        "description": "AI 出题、面试分析、录用建议",
        "icon": "interview",
        "iconBg": "#fce4ec",
        "iconColor": "#e91e63",
    },
    "training": {
        "name": "培训 Agent",
        "description": "入职培养、试用期跟踪",
        "icon": "training",
        "iconBg": "#e8f5e9",
        "iconColor": "#4caf50",
    },
    "performance": {
        "name": "绩效 Agent",
        "description": "绩效分析、员工成长建议",
        "icon": "performance",
        "iconBg": "#fff3e0",
        "iconColor": "#ff9800",
    },
}


# ── SSE Helpers ─────────────────────────────────────────

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def execute_tool_call(tool_name: str, params: dict, db: AsyncSession) -> str:
    """Execute a tool and return a human-readable result summary."""
    try:
        if tool_name == "list_resumes":
            from app.routers.resumes import list_resumes as fn
            pos_id = params.get("positionId", "all")
            statuses = params.get("statuses", "")
            keyword = params.get("keyword", "")
            limit = params.get("limit", 10)
            result = await fn(positionId=pos_id, statuses=statuses, keyword=keyword, page=1, pageSize=limit, db=db)
            data = result["data"]
            if isinstance(data, dict) and "list" in data:
                candidates = data["list"]
                return f"找到 {data['total']} 位候选人。前 {len(candidates)} 位"
            return f"查询结果：{json.dumps(data, ensure_ascii=False)[:500]}"

        elif tool_name == "get_resume":
            from app.routers.resumes import get_resume as fn
            result = await fn(resume_id=params["id"], db=db)
            data = result.get("data")
            if data:
                return f"候选人「{data['name']}」- {data['position']}，匹配度 {data['score']} 分，状态 {data['status']}。技能：{', '.join(data.get('skills', [])[:8])}。{data.get('aiAnalysis', {}).get('summary', '')}"
            return "候选人不存在"

        elif tool_name == "list_positions":
            from app.routers.positions import list_positions as fn
            result = await fn(db=db)
            data = result["data"]
            return f"共有 {len(data)} 个岗位：{'，'.join(p['name'] for p in data)}"
            return f"共 {len(data)} 个岗位"

        elif tool_name == "update_resume":
            from app.routers.resumes import update_resume as fn
            fields = params.get("fields", {})
            if "status" in params:
                fields["status"] = params["status"]
            result = await fn(resume_id=params["id"], body=fields, db=db)
            if result["code"] == 0:
                return f"已成功更新候选人 {params['id']} 的信息"
            return f"更新失败：{result['message']}"

        elif tool_name == "get_questions":
            from app.routers.interview import get_questions as fn
            result = await fn(candidateId=params["candidateId"], round=params["round"], db=db)
            questions = result.get("data", [])
            return f"共 {len(questions)} 道题"

        elif tool_name == "generate_questions":
            from app.routers.interview import regenerate_questions as fn
            result = await fn(body=params, db=db)
            questions = result.get("data", [])
            return f"已生成 {len(questions)} 道新题目"

        elif tool_name == "get_evaluation":
            from app.routers.interview import get_evaluation as fn
            result = await fn(candidate_id=params["candidateId"], round=params.get("round", "first"), db=db)
            scores = result.get("data", [])
            scored = [s for s in scores if s.get("hrScore") is not None]
            return f"面试评分：{len(scored)}/{len(scores)} 题已评分"

        elif tool_name == "ai_score_question":
            from app.routers.interview import ai_score_question as fn
            result = await fn(question_id=params["questionId"], body={"answer": params.get("answer", "")}, db=db)
            data = result.get("data", {})
            return f"AI评分：{data.get('score', 0)} 分"

        elif tool_name == "get_leaderboard":
            from app.routers.interview import get_leaderboard as fn
            result = await fn(category=params["category"], db=db)
            data = result.get("data", [])
            return f"排行榜共 {len(data)} 人"

        elif tool_name == "list_probation":
            from app.routers.probation import list_probation as fn
            result = await fn(department=params.get("department", "all"), status=params.get("status", "all"), db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"试用期员工：共 {data.get('total', 0)} 人"
            return "查询完成"

        elif tool_name == "list_performance":
            from app.routers.performance import list_performance as fn
            result = await fn(quarter=params["quarter"], db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"绩效数据：共 {data.get('total', 0)} 条记录"
            return "查询完成"

        elif tool_name == "rag_search":
            try:
                from app.services.rag.search_service import search as rag_search_fn
                top_k = params.get("topK", 5)
                results = await rag_search_fn(
                    query=params["query"],
                    kb_ids=None,
                    top_k=top_k,
                )
                if results:
                    summaries = []
                    for r in results[:3]:
                        summaries.append(
                            f"[{r['kb_name']}] {r['file_name']}: {r['content'][:300]}"
                        )
                    return f"检索到 {len(results)} 条相关知识:\n" + "\n---\n".join(summaries)
                return "未检索到相关知识"
            except Exception:
                return "知识库检索暂不可用"

        elif tool_name == "list_knowledge":
            from app.routers.knowledge import list_knowledge as fn
            result = await fn(categoryKey=params.get("categoryKey", "all"), keyword=params.get("keyword", ""), db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"知识库：共 {data.get('total', 0)} 条素材"
            return "查询完成"

        elif tool_name == "get_operations_dashboard":
            from app.routers.dashboard import get_operations as fn
            result = await fn(db=db)
            data = result.get("data", {})
            summary = data.get("summary", {})
            return f"运营概览：{summary.get('title', '')} - {summary.get('text', '')}"

        elif tool_name == "get_settings":
            from app.routers.settings import get_settings as fn
            result = await fn(db=db)
            data = result.get("data", {})
            return f"系统设置：公司「{data.get('companyName', '')}」，AI解析={'开启' if data.get('aiResumeAnalysis') else '关闭'}"

        else:
            return f"工具 {tool_name} 执行完成"

    except Exception as e:
        return f"工具执行错误: {str(e)}"


# ── Agent System Prompt ─────────────────────────────────

def build_system_prompt(agent_id: str = "recruit") -> str:
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["recruit"])
    return f"""你是 Genie 智能招聘系统的 {agent_info['name']}，{agent_info['description']}。

你是招聘总管，可以协调招聘、面试、培训、绩效四个子Agent协同工作。

工作规则：
1. 你必须使用工具来获取系统中的实际数据，不要编造数据
2. 当用户要求执行操作时，先确认理解意图，再调用合适的工具
3. 操作完成后，用简洁的中文汇报结果
4. 如果用户的问题涉及多个步骤，请按顺序执行并汇报进度
5. 你可以将复杂任务拆解给子Agent处理：使用 @interview、@training、@performance 标记
6. 对于知识类问题，优先使用 rag_search 检索知识库
7. 回复使用 Markdown 格式，包含清晰的标题和列表

可用的子Agent：
- @interview：负责面试出题、面试分析、录用建议
- @training：负责入职培养、试用期跟踪
- @performance：负责绩效分析、员工成长建议"""


# ── API Endpoints ───────────────────────────────────────

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
                    "status": "就绪", "tone": "active",
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
        "data": [
            {
                "id": str(s.id),
                "title": s.title or "新对话",
                "agentId": s.agent_id,
                "createdAt": s.created_at.isoformat() if s.created_at else "",
                "updatedAt": s.updated_at.isoformat() if s.updated_at else "",
            }
            for s in sessions
        ],
    }


@router.post("/ai-agent/sessions")
async def create_session(
    db: AsyncSession = Depends(get_db),
):
    """Create a new empty chat session."""
    session = AgentSession(
        title="新对话",
        agent_id="recruit",
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(session.id),
            "title": session.title,
            "agentId": session.agent_id,
            "createdAt": session.created_at.isoformat() if session.created_at else "",
            "updatedAt": session.updated_at.isoformat() if session.updated_at else "",
        },
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

    await db.delete(session)
    return {"code": 0, "message": "ok", "data": None}


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
                "createdAt": m.created_at.isoformat() if m.created_at else "",
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
        saved_name = f"{uuid.uuid4()}{file_ext}"
        material_path = os.path.join(settings.upload_dir, saved_name)
        with open(material_path, "wb") as f:
            content = await file.read()
            f.write(content)
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
    if type in ("resume", "file") and material_path and os.path.exists(material_path):
        try:
            text, extract_error = extract_text_from_file(material_path)
            if text.strip():
                parsed, _ = await parse_resume_with_llm(text)
                analysis_data = parsed.get("analysis", {}) if parsed else {}
                analysis = {
                    "overallScore": analysis_data.get("overallScore"),
                    "summary": analysis_data.get("summary", ""),
                    "keywords": analysis_data.get("keywords", []),
                    "dimensions": analysis_data.get("dimensions", []),
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
    agent_id = body.get("agentId", "recruit")
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
                    if mat.file_path and os.path.exists(mat.file_path):
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
                agent_id=agent_id,
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

            # Build LangGraph agent
            langchain_tools = create_langchain_tools(agent_id)
            system_prompt = build_system_prompt(agent_id)
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
            # Client disconnected — mark the task cancelled (short-lived session),
            # then re-raise so Starlette closes the stream and the socket is released.
            try:
                async with async_session_factory() as db:
                    if task_id is not None:
                        task_result = await db.execute(select(AgentTask).where(AgentTask.id == task_id))
                        task_obj = task_result.scalar_one_or_none()
                        if task_obj:
                            task_obj.status = "cancelled"
                            task_obj.finished_at = datetime.utcnow()
                    await db.commit()
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
