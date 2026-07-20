"""
Genie AI Agent v3 — Claude Code architecture chat endpoint.

Complete rewrite using:
- Async generator state machine (query_loop replaces LangGraph ReAct)
- StreamingToolExecutor (tools run DURING model streaming)
- 5-stage compaction pipeline with circuit breaker
- Deny-First permission engine
- Cache-aware prompt building
- Extended hook system (27 events, 6 types)
- Smart retry with model fallback
- Transcript-first persistence (user message saved BEFORE API call)
"""

from __future__ import annotations

import json
import re
import uuid
import asyncio
import platform
from datetime import datetime
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.database import get_db, async_session_factory
from app.models.agent_session import AgentSession, AgentMessage, AgentTask
from app.agent.tools import create_langchain_tools, create_langchain_tools_from_defs
from app.agent.intent_classifier import classify_intent, get_tool_defs_for_intent
from app.agent.supervisor_graph import classify_complexity_sync, ExecutionPlan, PlanStep
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from app.agent.tool_result import ToolResult as LegacyToolResult
from app.agent_os.agent_os import AgentOS
from app.agent_os.output.schemas import ToolResult as StructuredToolResult
from app.agent_os.hooks.events import HookEvent
from app.agent_os.loop.state import QueryContext, LoopState, ContinueReason, ExitReason
from app.agent_os.loop.query_loop import query_loop
from app.agent_os.permissions.engine import PermissionMode
from app.config import get_settings
from app.api.ai.prompt import AGENT_CONFIGS, build_system_prompt
from app.api.ai.tool_executor import sse_event, execute_tool_call

router = APIRouter(tags=["AI Agent v3"])
settings = get_settings()

# ── Global Agent OS instance (v3) ───────────────────────────
agent_os = AgentOS(
    memory_dir="memory/",
    max_context_tokens=getattr(settings, "max_context_tokens", 8000),
)

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=120.0,  # increased for streaming
    max_retries=0,   # we handle retries ourselves
)


def iso_utc(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    text = dt.isoformat()
    if text.endswith("Z") or text.endswith("+00:00"):
        return text
    if len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        return text
    return text + "Z"


# ═══════════════════════════════════════════════════════════════
#  Plan Generation (shared with v2)
# ═══════════════════════════════════════════════════════════════

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

async def _generate_plan(lc_messages: list, client: AsyncOpenAI) -> Optional[ExecutionPlan]:
    """Generate an execution plan for complex user requests."""
    try:
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


# ═══════════════════════════════════════════════════════════════
#  Memory & Skills & Hooks Management Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/agent-os/memories")
async def list_memories():
    """List all persistent memories."""
    memories = agent_os.memory_manager.load_all()
    return {
        "code": 0, "message": "ok",
        "data": [
            {
                "name": m.name, "description": m.description,
                "type": m.memory_type, "scope": m.scope,
                "created": m.created, "updated": m.updated,
            }
            for m in memories
        ],
    }

@router.post("/agent-os/memories")
async def create_or_update_memory(body: dict):
    """Create or update a memory."""
    name = (body.get("name") or "").strip()
    content = (body.get("content") or "").strip()
    description = (body.get("description") or "").strip()
    if not name:
        return {"code": 400, "message": "name 是必填字段", "data": None}
    mem = await agent_os.memory_manager.remember(
        name=name, content=content, description=description,
        metadata=body.get("metadata", {}),
    )
    return {
        "code": 0, "message": "ok",
        "data": {"name": mem.name, "description": mem.description,
                 "type": mem.memory_type, "scope": mem.scope},
    }

@router.delete("/agent-os/memories/{name}")
async def delete_memory(name: str):
    """Delete a memory by name."""
    existed = agent_os.memory_manager.forget(name)
    if not existed:
        return {"code": 404, "message": f"记忆 '{name}' 不存在", "data": None}
    return {"code": 0, "message": "ok", "data": None}

@router.get("/agent-os/skills")
async def list_skills():
    """List all available skills."""
    skills = agent_os.skill_registry.list_all()
    return {"code": 0, "message": "ok", "data": [s.to_api_dict() for s in skills]}

@router.post("/agent-os/skills/reload")
async def reload_skills():
    """Hot-reload skills from disk."""
    agent_os.skill_registry.reload()
    skills = agent_os.skill_registry.list_all()
    return {"code": 0, "message": f"已重新加载 {len(skills)} 个技能", "data": [s.name for s in skills]}

@router.get("/agent-os/hooks")
async def list_hooks():
    """List all registered hooks and available events."""
    return {
        "code": 0, "message": "ok",
        "data": {
            "hooks": agent_os.hook_manager.list_hooks(),
            "events": agent_os.hook_manager.list_events(),
        },
    }

@router.get("/agent-os/permissions")
async def list_permission_rules():
    """List all permission rules."""
    return {
        "code": 0, "message": "ok",
        "data": {"rules": agent_os.permission_engine.list_rules()},
    }

@router.get("/agent-os/context-stats")
async def get_context_stats():
    """Return context compression stats."""
    stats = agent_os.compaction_pipeline.get_stats()
    return {"code": 0, "message": "ok", "data": stats}

@router.get("/agent-os/cache-stats")
async def get_cache_stats():
    """Return prompt cache statistics."""
    return {
        "code": 0, "message": "ok",
        "data": agent_os.prompt_builder.get_cache_stats(),
    }

@router.post("/agent-os/confirm-plan")
async def confirm_plan(body: dict):
    """Confirm, reject, or modify a proposed execution plan."""
    plan_id = (body.get("plan_id") or "").strip()
    action = (body.get("action") or "approve").strip()
    modifications = (body.get("modifications") or "").strip()
    if not plan_id:
        return {"code": 400, "message": "plan_id 是必填字段", "data": None}
    ok = agent_os.plan_coordinator.confirm(plan_id, action, modifications)
    if not ok:
        return {"code": 404, "message": f"计划 '{plan_id}' 不存在或已过期", "data": None}
    return {"code": 0, "message": f"计划 {plan_id} 已{action}",
            "data": {"plan_id": plan_id, "action": action}}


# ═══════════════════════════════════════════════════════════════
#  Main SSE Streaming Chat Endpoint (v3 Architecture)
# ═══════════════════════════════════════════════════════════════

@router.post("/agent-os/chat")
async def agent_os_chat(request: Request, body: dict):
    """Agent OS v3 SSE streaming chat endpoint.

    Claude Code architecture:
    - Async generator state machine (query_loop)
    - Stream tool execution (tools run DURING model streaming)
    - 5-stage compaction before every API call
    - Deny-first permission checks
    - Transcript-first persistence
    - Post-turn memory extraction (fire-and-forget)
    """
    message = body.get("message", "")
    session_id = body.get("sessionId")
    agent_id = body.get("agentId", "genie")
    mentioned_agent_ids = body.get("mentionedAgentIds", [])
    material_ids = body.get("materialIds", [])
    permission_mode = body.get("permissionMode", "default")

    # ── Load memories ─────────────────────────────────────
    memories = await agent_os.load_memories(query=message, llm_client=llm_client, limit=5)

    # ── Fetch attached materials ──────────────────────────
    material_context = ""
    if material_ids:
        try:
            from app.models.agent_session import AgentMaterial
            async with async_session_factory() as db:
                mat_result = await db.execute(
                    select(AgentMaterial).where(AgentMaterial.id.in_(material_ids))
                )
                attached = mat_result.scalars().all()
            if attached:
                lines = ["\n\n--- 附件资料 ---"]
                for mat in attached:
                    lines.append(f"\n[{mat.type}] {mat.name}")
                    if mat.file_path:
                        try:
                            from app.api.recruitment.resumes import extract_text_from_file
                            file_text, _ = extract_text_from_file(mat.file_path)
                            if file_text.strip():
                                truncated = file_text[:3000] + ("..." if len(file_text) > 3000 else "")
                                lines.append(f"内容:\n{truncated}")
                        except Exception:
                            pass
                material_context = "\n".join(lines)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════
    #  Transcript-First: Save user message BEFORE API call
    # ═══════════════════════════════════════════════════════
    is_new_session = False
    session_title = "新对话"
    session_obj_id = None
    history_rows: list[tuple[str, str]] = []

    async with async_session_factory() as db:
        session = None
        if session_id:
            try:
                result = await db.execute(
                    select(AgentSession).where(AgentSession.id == session_id)
                )
                session = result.scalar_one_or_none()
            except Exception:
                session = None

        if not session:
            is_new_session = True
            session = AgentSession(title="新对话", agent_id="genie")
            db.add(session)
            await db.flush()
            session_id = str(session.id)

        # 🔑 SAVE USER MESSAGE BEFORE API CALL (transcript-first)
        user_msg = AgentMessage(
            session_id=session.id,
            role="user",
            content=message,
        )
        db.add(user_msg)
        await db.flush()

        if session.title in ("新对话", None, ""):
            session.title = message[:50] if message else "新对话"
        session_title = session.title or "新对话"

        # Load history
        history_result = await db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session.id)
            .order_by(AgentMessage.created_at)
            .limit(30)
        )
        history = history_result.scalars().all()
        for h in history[:-1]:
            history_rows.append((h.role, h.content or ""))

        await db.commit()
        session_obj_id = session.id

    # ── Build LangChain messages ────────────────────────
    lc_messages = []
    for role, content in history_rows:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    lc_messages.append(HumanMessage(content=message + material_context))

    # ── Build session context ───────────────────────────
    session_context_parts = []
    pos_ids = re.findall(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", message)
    if pos_ids:
        session_context_parts.append(f"用户消息中引用了 ID: {', '.join(pos_ids[:5])}")

    # ── Detect position/candidate mentions ──────────────
    async def event_stream() -> AsyncGenerator[str, None]:
        task_id = None
        assistant_full_response = ""
        assistant_tool_blocks: list[dict] = []

        try:
            # ── Thinking notification ──
            agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["genie"])
            thinking_text = f"收到任务，正在作为{agent_info.get('name', 'AI Agent')}分析..."

            if memories:
                mem_names = ", ".join(m.description for m in memories[:3])
                thinking_text += f"\n已加载相关记忆：{mem_names}"
                yield sse_event("memory_loaded", {
                    "count": len(memories),
                    "memories": [{"name": m.name, "description": m.description} for m in memories[:5]],
                })

            yield sse_event("thinking", {"text": thinking_text, "append": False})

            # ── Intent classification + tool gating ─────
            intent = classify_intent(message)
            if intent == "general":
                langchain_tools = create_langchain_tools(agent_id)
                tool_defs = [{
                    "name": t.name, "description": t.description,
                    "parameters": t.args_schema.schema() if hasattr(t, "args_schema") else {},
                } for t in langchain_tools]
            else:
                restricted_defs = get_tool_defs_for_intent(intent)
                langchain_tools = create_langchain_tools_from_defs(restricted_defs)
                tool_defs = restricted_defs

            INTENT_LABELS = {
                "position_query": "岗位JD查询", "candidate_query": "候选人查询",
                "candidate_action": "候选人操作", "interview": "面试管理",
                "probation": "试用期考核", "performance": "绩效管理",
                "knowledge": "知识库", "dashboard": "数据看板", "settings": "系统设置",
            }
            runtime_directives = ""
            if intent != "general":
                hint = INTENT_LABELS.get(intent, intent)
                runtime_directives = f"[Intent Gate] 当前意图: {hint}。本轮仅可使用与该意图匹配的工具。"

            # ── Skill matching ──────────────────────────
            matched_skills = await agent_os.match_skills(message, llm_client, limit=3)
            if matched_skills:
                skill_names = ", ".join(s.name for s in matched_skills)
                yield sse_event("thinking", {
                    "text": f"匹配到相关技能：{skill_names}",
                    "append": True,
                })

            # ── Register tools for deferred loading ─────
            valid_tool_defs = []
            for td in tool_defs:
                if isinstance(td, dict) and "name" in td:
                    valid_tool_defs.append(td)
            agent_os.register_tools(valid_tool_defs)

            # ── Build layered system prompt ─────────────
            base_prompt = build_system_prompt(agent_id)
            skill_catalog = agent_os.skill_registry.get_catalog()
            base_prompt += f"\n\n{skill_catalog}"

            full_prompt, static_prompt, dynamic_prompt = agent_os.build_prompt(
                base_prompt=base_prompt,
                agent_id=agent_id,
                agent_config=AGENT_CONFIGS.get(agent_id, {}),
                memories=memories,
                session_context="\n".join(session_context_parts) if session_context_parts else "",
                runtime_directives=runtime_directives,
                environment={
                    "cwd": "/app",
                    "os": platform.system(),
                    "platform": platform.platform(),
                },
            )

            # ── Fire UserPromptSubmit hook ───────────────
            try:
                await agent_os.fire_hooks(
                    HookEvent.USER_PROMPT_SUBMIT,
                    {"session_id": str(session_obj_id), "message": message},
                )
            except Exception:
                pass

            # ── Inject matched skills into prompt ───────
            if matched_skills:
                full_prompt = agent_os.inject_skills_into_prompt(full_prompt, matched_skills)

            # ── Plan generation (complex queries) ───────
            complexity = classify_complexity_sync(message)
            if complexity == "complex":
                plan = await _generate_plan(lc_messages, llm_client)
                if plan and plan.steps:
                    yield sse_event("plan_proposal", plan.to_sse_dict())
                    steps_text = "\n".join(
                        f"  {s.index}. {s.title}: {s.description}"
                        for s in plan.steps
                    )
                    dynamic_prompt += (
                        f"\n\n[执行计划] {plan.title}\n{steps_text}\n"
                        f"按步骤顺序执行。每完成一步，检查结果后再进行下一步。"
                    )

            # ── Task tracking ────────────────────────────
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
                await db.commit()
                task_id = task.id

            # ═══════════════════════════════════════════════
            #  PHASE: Run Agent Loop (v3 Architecture)
            # ═══════════════════════════════════════════════

            # Build QueryContext
            query_ctx = QueryContext(
                session_id=str(session_obj_id),
                agent_id=agent_id,
                permission_mode=permission_mode,
                max_turns=getattr(settings, "max_agent_turns", 30),
                max_output_tokens=8192,
                max_output_tokens_escalated=64000,
                enable_streaming_tools=True,
                enable_auto_compact=True,
            )

            # Build tool schemas
            tool_schemas = agent_os.get_tool_schemas()

            # 🔑 Run the new agent loop
            for event_type in ["content", "tool_call", "tool_result",
                               "thinking", "compaction", "error", "done"]:
                pass  # Just declaring expected types

            async for event in agent_os.run_loop(
                messages=lc_messages,
                system_prompt_static=static_prompt,
                system_prompt_dynamic=dynamic_prompt,
                tools=tool_schemas,
                context=query_ctx,
                llm_client=llm_client,
                tool_executor=execute_tool_call,
                db_factory=async_session_factory,
            ):
                if await request.is_disconnected():
                    break

                event_type = event.get("type", "")
                event_data = event.get("data", {})

                if event_type == "content":
                    assistant_full_response += event_data.get("delta", "")
                    yield sse_event("content", event_data)

                elif event_type == "tool_call":
                    assistant_tool_blocks.append(event_data)
                    yield sse_event("tool_call", event_data)

                elif event_type == "tool_result":
                    yield sse_event("tool_result", event_data)

                elif event_type == "thinking":
                    yield sse_event("thinking", event_data)

                elif event_type == "compaction":
                    yield sse_event("context_compressed", {
                        "layer": event_data.get("layer", 0),
                        "tokenEstimate": event_data.get("compressed_tokens", 0),
                        "message": event_data.get("message", "上下文已压缩"),
                    })

                elif event_type == "error":
                    yield sse_event("error", event_data)

                elif event_type == "done":
                    exit_reason = event_data.get("exit_reason", "completed")
                    if exit_reason != "completed":
                        yield sse_event("verification", {
                            "verified": exit_reason == "completed",
                            "issues": [f"Agent 异常退出: {exit_reason}"],
                        })

            # ═══════════════════════════════════════════════
            #  Post-loop: Persist + Auto-capture + Hooks
            # ═══════════════════════════════════════════════

            # ── Persist assistant response ──
            async with async_session_factory() as db:
                assistant_msg = AgentMessage(
                    session_id=session_obj_id,
                    role="assistant",
                    content=assistant_full_response,
                    thinking=None,
                    tool_blocks=assistant_tool_blocks if assistant_tool_blocks else None,
                )
                db.add(assistant_msg)
                await db.flush()

                # AI-generated session title
                if is_new_session or session_title == "新对话":
                    try:
                        title_prompt = (
                            f"根据以下对话内容，生成一个简短的标题（10个字以内，不要引号）：\n"
                            f"用户：{message[:200]}\nAI：{assistant_full_response[:200]}"
                        )
                        title_resp = await llm_client.chat.completions.create(
                            model=settings.deepseek_model,
                            messages=[{"role": "user", "content": title_prompt}],
                            temperature=0.7, max_tokens=32,
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

                # Mark task done
                if task_id is not None:
                    task_result = await db.execute(
                        select(AgentTask).where(AgentTask.id == task_id)
                    )
                    task_obj = task_result.scalar_one_or_none()
                    if task_obj:
                        task_obj.status = "done"
                        task_obj.progress = 100
                        task_obj.finished_at = datetime.utcnow()

                await db.commit()

            # ── Fire AGENT_RESPONSE hook ──
            try:
                await agent_os.fire_hooks(
                    HookEvent.AGENT_RESPONSE,
                    {
                        "session_id": str(session_obj_id),
                        "user_message": message,
                        "response": assistant_full_response,
                        "tool_calls": [b.get("name", "") for b in assistant_tool_blocks],
                    },
                )
            except Exception:
                pass

            # ── Fire-and-forget: auto-capture learnings ──
            async def _auto_capture():
                try:
                    new_memories = await agent_os.capture_session_learnings(
                        messages=lc_messages + [AIMessage(content=assistant_full_response)],
                        session_id=str(session_obj_id),
                        llm_client=llm_client,
                    )
                    # Note: this runs in background, results are not yielded
                    # (Claude Code pattern: fire-and-forget, not blocking the response)
                except Exception:
                    pass

            asyncio.create_task(_auto_capture())

            yield sse_event("phase_result", {
                "id": f"phase_{uuid.uuid4().hex[:6]}",
                "tone": "success",
                "title": "任务完成",
                "description": message[:80] + ("..." if len(message) > 80 else ""),
            })

            yield sse_event("done", {})

        except asyncio.CancelledError:
            # Cleanup on cancel
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
            yield sse_event("error", {"message": str(e)[:500]})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Tool executor wrapper for QualityGuard ────────────────

async def _execute_with_session(tool_name: str, params: dict) -> str:
    """Execute a tool with a fresh DB session (for verification)."""
    from app.api.ai.tool_executor import execute_tool_call as _exec
    db = async_session_factory()
    try:
        result = await _exec(tool_name, params, db)
        await db.commit()
        return result
    except Exception as e:
        await db.rollback()
        return f"工具执行错误: {e}"
    finally:
        await db.close()
