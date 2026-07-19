"""
Genie AI Agent v2 — Agent OS-powered chat endpoint.

Upgrades the original agent_chat.py with:
- MemoryManager: persistent memory across sessions (load + auto-capture)
- ContextEngine: smart compression before every LLM call
- SystemPromptBuilder: layered prompt with deferred tool loading
- QualityGuard: post-execution verification + reflection

The original v1 endpoint is preserved at agent_chat.py for backward
compatibility.  Switch via ``ENABLE_AGENT_OS=true`` or use the /v2 suffix.
"""

from __future__ import annotations

import json
import re
import uuid
import asyncio
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
from app.agent_os.orchestration.workflow import WorkflowDefinition
from app.config import get_settings
from app.api.ai.prompt import AGENT_CONFIGS, build_system_prompt
from app.api.ai.tool_executor import sse_event, execute_tool_call

router = APIRouter(tags=["AI Agent v2"])
settings = get_settings()

# ── Global Agent OS instance ──────────────────────────────
agent_os = AgentOS(
    memory_dir="memory/",
    max_context_tokens=getattr(settings, "max_context_tokens", 8000),
)

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
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


# ═══════════════════════════════════════════════════════════
#  Plan Generation (moved from v1, shared)
# ═══════════════════════════════════════════════════════════

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
    client: AsyncOpenAI,
) -> Optional[ExecutionPlan]:
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


# ═══════════════════════════════════════════════════════════
#  AgentOS Memory Management Endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/agent-os/memories")
async def list_memories():
    """List all persistent memories."""
    memories = agent_os.memory_manager.load_all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "name": m.name,
                "description": m.description,
                "type": m.memory_type,
                "scope": m.scope,
                "created": m.created,
                "updated": m.updated,
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
        name=name,
        content=content,
        description=description,
        metadata=body.get("metadata", {}),
    )
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "name": mem.name,
            "description": mem.description,
            "type": mem.memory_type,
            "scope": mem.scope,
        },
    }


@router.delete("/agent-os/memories/{name}")
async def delete_memory(name: str):
    """Delete a memory by name."""
    existed = agent_os.memory_manager.forget(name)
    if not existed:
        return {"code": 404, "message": f"记忆 '{name}' 不存在", "data": None}
    return {"code": 0, "message": "ok", "data": None}


@router.get("/agent-os/context-stats")
async def get_context_stats():
    """Return recent context compression stats (for the frontend usage bar)."""
    stats = agent_os.context_engine._stats[-10:]  # last 10 compressions
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "totalMessages": s.total_messages,
                "originalTokens": s.original_tokens,
                "compressedTokens": s.compressed_tokens,
                "savingsPercent": round(s.savings_percent, 1),
                "layerUsed": s.layer_used,
            }
            for s in stats
        ],
    }


# ═══════════════════════════════════════════════════════════
#  Agent OS Skills Endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/agent-os/skills")
async def list_skills():
    """List all available skills (catalog)."""
    skills = agent_os.skill_registry.list_all()
    return {
        "code": 0,
        "message": "ok",
        "data": [s.to_api_dict() for s in skills],
    }


@router.post("/agent-os/skills/{name}/trigger")
async def trigger_skill(name: str):
    """Manually trigger a skill by name (returns its full content)."""
    skill = agent_os.skill_registry.get(name)
    if not skill:
        return {"code": 404, "message": f"技能 '{name}' 不存在", "data": None}
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "name": skill.name,
            "description": skill.description,
            "triggers": skill.triggers,
            "content": skill.content[:2000],  # truncated for API display
        },
    }


@router.post("/agent-os/skills/reload")
async def reload_skills():
    """Hot-reload skills from disk (no restart needed)."""
    agent_os.skill_registry.reload()
    skills = agent_os.skill_registry.list_all()
    return {
        "code": 0,
        "message": f"已重新加载 {len(skills)} 个技能",
        "data": [s.name for s in skills],
    }


# ═══════════════════════════════════════════════════════════
#  Agent OS Hooks Endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/agent-os/hooks")
async def list_hooks():
    """List all registered hooks."""
    return {
        "code": 0,
        "message": "ok",
        "data": agent_os.hook_manager.list_hooks(),
    }


# ═══════════════════════════════════════════════════════════
#  Agent OS Plan Confirmation
# ═══════════════════════════════════════════════════════════

@router.post("/agent-os/confirm-plan")
async def confirm_plan(body: dict):
    """Confirm, reject, or modify a proposed execution plan.

    Body: {"plan_id": "...", "action": "approve" | "reject" | "modify", "modifications": "..."}
    """
    plan_id = (body.get("plan_id") or "").strip()
    action = (body.get("action") or "approve").strip()
    modifications = (body.get("modifications") or "").strip()

    if not plan_id:
        return {"code": 400, "message": "plan_id 是必填字段", "data": None}

    ok = agent_os.plan_coordinator.confirm(plan_id, action, modifications)
    if not ok:
        return {"code": 404, "message": f"计划 '{plan_id}' 不存在或已过期", "data": None}

    return {
        "code": 0,
        "message": f"计划 {plan_id} 已{action}",
        "data": {"plan_id": plan_id, "action": action},
    }


# ═══════════════════════════════════════════════════════════
#  Agent OS Workflow / Sub-Agent Endpoints
# ═══════════════════════════════════════════════════════════

@router.post("/agent-os/workflow")
async def execute_workflow(body: dict):
    """Execute a multi-agent workflow.

    Body::

        {
          "name": "批量筛选+出题",
          "stages": [
            {
              "name": "screening",
              "parallel": true,
              "tasks": [
                {"agent": "recruit", "task": "筛选 Agent 工程师候选人 top 5"},
                {"agent": "recruit", "task": "筛选前端候选人 top 5"}
              ]
            },
            {
              "name": "questions",
              "parallel": true,
              "depends_on": ["screening"],
              "tasks": [...]
            }
          ]
        }

    Each task's ``tools`` list is validated against the agent_type's
    authorized tool set to prevent privilege escalation.
    """
    from app.agent.tools import get_tools_for_agent

    # ── Validate tool authorization per task ──
    try:
        wf = WorkflowDefinition.from_json(body)
    except Exception as e:
        return {"code": 400, "message": f"工作流定义无效: {e}", "data": None}

    for stage in wf.stages:
        for task in stage.tasks:
            # Resolve allowed tools for this agent type
            allowed_defs = get_tools_for_agent(task.agent_type)
            allowed_names = {t["name"] for t in allowed_defs}
            if task.tools is not None:
                # Intersect user-specified tools with agent's authorized set
                unauthorized = [t for t in task.tools if t not in allowed_names]
                if unauthorized:
                    return {
                        "code": 403,
                        "message": (
                            f"Agent '{task.agent_type}' 无权使用工具: {', '.join(unauthorized)}。"
                            f"允许的工具: {', '.join(sorted(allowed_names)[:20])}..."
                        ),
                        "data": None,
                    }
                # Use the validated list
                task.tools = [t for t in task.tools if t in allowed_names]
            else:
                # No explicit tools → use agent's default set
                task.tools = list(allowed_names)

    result = await agent_os.workflow_engine.execute(
        wf, llm_client, execute_tool_call, async_session_factory
    )

    return {
        "code": 0 if result.success else 1,
        "message": f"完成 {result.completed_tasks}/{result.total_tasks} 个任务，{result.failed_tasks} 个失败",
        "data": {
            "workflow_id": result.workflow_id,
            "success": result.success,
            "total_tasks": result.total_tasks,
            "completed_tasks": result.completed_tasks,
            "failed_tasks": result.failed_tasks,
            "stage_results": {
                name: [
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "summary": r.summary[:200],
                        "tool_calls": r.tool_calls_made,
                    }
                    for r in results
                ]
                for name, results in result.stage_results.items()
            },
            "error": result.error,
        },
    }


# ═══════════════════════════════════════════════════════════
#  Main SSE Streaming Chat Endpoint
# ═══════════════════════════════════════════════════════════

@router.post("/agent-os/chat")
async def agent_os_chat(
    request: Request,
    body: dict,
):
    """Agent OS-powered SSE streaming chat endpoint.

    Same schema as v1, with added intelligence:
    - Relevant memories are loaded and injected into the system prompt
    - Context is compressed before LLM calls when needed
    - Tool results use structured output for better frontend rendering
    - Post-execution quality checks catch errors
    - Session learnings are auto-captured on completion
    """
    message = body.get("message", "")
    session_id = body.get("sessionId")
    agent_id = body.get("agentId", "genie")
    mentioned_agent_ids = body.get("mentionedAgentIds", [])
    material_ids = body.get("materialIds", [])

    # ── Phase 0: Load memories ─────────────────────────
    memories = await agent_os.load_memories(
        query=message,
        llm_client=llm_client,
        limit=5,
    )

    # ── Fetch attached materials ────────────────────────
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

    # ── Get or create session, save user message ───────
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

        history_result = await db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session.id)
            .order_by(AgentMessage.created_at)
            .limit(20)
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

    # ── Build session context string ────────────────────
    session_context_parts = []
    # Detect if the conversation mentions a position or candidate
    pos_ids = re.findall(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", message)
    if pos_ids:
        session_context_parts.append(f"用户消息中引用了 ID: {', '.join(pos_ids[:5])}")

    async def event_stream() -> AsyncGenerator[str, None]:
        result = AgentResult()
        task_id = None
        tool_call_log: list[dict] = []
        tool_result_log: list[str] = []

        try:
            # ── Phase 1: Thinking + Memory notification ──
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

            # ── Phase 2: Intent classification + tool gating ──
            intent = classify_intent(message)
            if intent == "general":
                langchain_tools = create_langchain_tools(agent_id)
                tool_defs = create_langchain_tools(agent_id)  # full defs for catalog
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

            # ── Phase 2.5: Skill matching ──
            matched_skills = await agent_os.match_skills(message, llm_client, limit=3)
            if matched_skills:
                skill_names = ", ".join(s.name for s in matched_skills)
                yield sse_event("thinking", {
                    "text": f"匹配到相关技能：{skill_names}",
                    "append": True,
                })

            # ── Phase 3: Agent OS Prompt Building ──
            base_prompt = build_system_prompt(agent_id)
            # Inject skill catalog into the base prompt
            skill_catalog = agent_os.skill_registry.get_catalog()
            base_prompt += f"\n\n{skill_catalog}"
            full_prompt, static_prompt, dynamic_prompt = agent_os.build_prompt(
                base_prompt=base_prompt,
                agent_id=agent_id,
                agent_config=AGENT_CONFIGS.get(agent_id, {}),
                tool_defs=[
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.args_schema.schema() if hasattr(t, "args_schema") else {},
                    }
                    for t in langchain_tools
                ] if hasattr(langchain_tools[0], "name") else tool_defs,
                memories=memories,
                session_context="\n".join(session_context_parts) if session_context_parts else "",
                runtime_directives=runtime_directives,
            )

            # ── Phase 4: Context compression ──
            ctx = await agent_os.prepare_context(
                messages=lc_messages,
                system_prompt=static_prompt,
                system_prompt_dynamic=dynamic_prompt,
            )
            if ctx.compression_applied:
                yield sse_event("context_compressed", {
                    "layer": ctx.compression_layer,
                    "tokenEstimate": ctx.token_estimate,
                    "message": f"上下文已压缩（Layer {ctx.compression_layer}），当前预估 {ctx.token_estimate} tokens",
                })

            # ── Inject matched skills into final prompt ──
            if matched_skills:
                full_prompt = agent_os.inject_skills_into_prompt(full_prompt, matched_skills)

            # ── Phase 5: Plan generation (complex queries) ──
            complexity = classify_complexity_sync(message)
            if complexity == "complex":
                plan = await _generate_plan(lc_messages, llm_client)
                if plan and plan.steps:
                    yield sse_event("plan_proposal", plan.to_sse_dict())
                    steps_text = "\n".join(
                        f"  {s.index}. {s.title}: {s.description}"
                        for s in plan.steps
                    )
                    full_prompt += (
                        f"\n\n[执行计划] {plan.title}\n{steps_text}\n"
                        f"按步骤顺序执行。每完成一步，检查结果后再进行下一步。"
                    )

            # ── Phase 6: Task tracking ──
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

            # ── Phase 7: Agent execution ──
            # Build graph with structured tool output adapter
            from app.agent_os.output.adapter import ToolResultAdapter
            adapter = ToolResultAdapter()
            graph = build_agent_graph(langchain_tools, full_prompt, tool_result_adapter=adapter)

            # Inject context-prepared messages
            async for sse_str in stream_agent_response(graph, ctx.messages, result):
                if await request.is_disconnected():
                    break
                yield sse_str

            # ── Phase 8: Quality guard ──
            if result.full_content and getattr(settings, "enable_verification", True):
                guard_result = await agent_os.verify_quality(
                    user_message=message,
                    agent_response=result.full_content,
                    tool_calls=result.tool_blocks,
                    tool_results=[b.get("result", "") for b in result.tool_blocks],
                    tool_executor=_execute_with_session,
                )
                if guard_result.issues:
                    yield sse_event("verification", {
                        "verified": guard_result.passed,
                        "issues": guard_result.issues,
                    })

                if guard_result.reflection_output and getattr(settings, "enable_reflection", True):
                    yield sse_event("reflection", {
                        "reason": guard_result.reflection_output.reason,
                    })
                    # Run one reflection iteration: re-invoke agent with reflection prompt
                    reflection_msg = HumanMessage(
                        content=guard_result.reflection_output.reflection_prompt
                    )
                    reflection_messages = ctx.messages + [reflection_msg]
                    reflection_result = AgentResult()
                    async for sse_str in stream_agent_response(
                        graph, reflection_messages, reflection_result
                    ):
                        if await request.is_disconnected():
                            break
                        yield sse_str
                    if reflection_result.full_content:
                        result.full_content = reflection_result.full_content

            # ── Phase 9: Persist + auto-capture ──
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

                # AI-generated session title (only for new sessions)
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

            # ── Fire hooks: agent response complete ──
            try:
                hook_ctx = {
                    "session_id": str(session_obj_id),
                    "user_message": message,
                    "response": result.full_content,
                    "tool_calls": [b.get("name", "") for b in result.tool_blocks],
                }
                hook_results = await agent_os.fire_hooks(
                    HookEvent.AGENT_RESPONSE, hook_ctx
                )
                for hr in hook_results:
                    if hr.get("action") == "warn":
                        yield sse_event("verification", {
                            "verified": True,
                            "issues": [hr.get("message", "Hook 警告")],
                        })
            except Exception:
                pass

            # ── Phase 10: Auto-capture learnings ──
            try:
                new_memories = await agent_os.capture_session_learnings(
                    messages=lc_messages + [
                        AIMessage(content=result.full_content)
                    ],
                    session_id=str(session_obj_id),
                    llm_client=llm_client,
                )
                if new_memories:
                    yield sse_event("memory_created", {
                        "count": len(new_memories),
                        "memories": [m.description for m in new_memories],
                    })
            except Exception:
                pass  # Auto-capture is best-effort

            yield sse_event("phase_result", {
                "id": f"phase_{uuid.uuid4().hex[:6]}",
                "tone": "success",
                "title": "任务完成",
                "description": message[:80] + ("..." if len(message) > 80 else ""),
            })

            yield sse_event("done", {})

        except asyncio.CancelledError:
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
