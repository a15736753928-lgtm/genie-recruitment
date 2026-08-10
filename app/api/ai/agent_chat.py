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
import hashlib
from datetime import datetime, date
from typing import Optional, List, AsyncGenerator, Any
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.database import get_db, async_session_factory
from app.utils.clock import iso_utc
from app.models.agent_session import AgentProject, AgentSession, AgentMessage, AgentMaterial, AgentTask
from app.models.recruitment import Candidate, Position
from app.core.security import get_current_user, CurrentUser
from app.core.permissions import WILDCARD_PERMISSION
from app.agent.tools import create_langchain_tools
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from langgraph.errors import GraphRecursionError
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

# ── Write-verification guard (read-after-write only) ───────────
from app.agent_os.quality.guard import QualityGuard
from app.utils.responses import ok, fail, not_found

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


_ENTITY_EXTRACT_RE = re.compile(
    r'「([^「」]{2,8})」|'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    re.I,
)


def _extract_digest(units: list[list]) -> str:
    """从被压缩丢弃的消息里提取高频实体（中文名 + UUID），供后续轮参考。

    长对话截断会丢掉早期关键事实（候选人名/ID、岗位名）。这里用规则提取
    「」内的 2~8 字片段和 UUID，拼成一行"早期对话要点"注入 system prompt，
    让模型在后续轮仍知道早期提过哪些实体（真正的 LLM 滚动摘要是后续演进项，
    规则版零延迟、不额外烧 token）。
    """
    found: list[str] = []
    seen: set[str] = set()
    for unit in units:
        for m in unit:
            text = str(getattr(m, "content", "") or "")
            for mt in _ENTITY_EXTRACT_RE.finditer(text):
                item = (mt.group(1) or mt.group(2) or "").strip()
                if not item or len(item) > 8:
                    continue
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(item)
                if len(found) >= 8:
                    break
            if len(found) >= 8:
                break
    if not found:
        return ""
    return (
        "（早期对话已压缩，其中提及过：" + "、".join(found) +
        "——如后续需要引用这些实体，请以工具查询返回的真实数据为准）"
    )


def _compact_messages(messages: list, max_tokens: int = 6000) -> tuple[list, int, str]:
    """Token-budget truncation: drop oldest messages when over budget.

    Returns (compacted_messages, n_dropped, digest).
    Keeps at least the final message (current user turn). Simpler and
    safer than the old CompactionPipeline, which was coupled to LoopState
    and silently corrupted ToolMessage metadata.

    Messages are dropped in atomic units, not one at a time: an
    AIMessage carrying ``tool_calls`` is grouped with the ToolMessage(s)
    that immediately follow it (history reconstruction upstream always
    emits them as an adjacent pair — one tool_call per AIMessage). If we
    popped from the front message-by-message, a cut could land between
    such a pair and leave an orphaned ToolMessage with no preceding
    tool_calls — which OpenAI-compatible APIs reject outright (400).

    ``digest``：被丢弃消息里提取的实体要点（空串表示本次未发生截断）。
    """
    if not messages:
        return messages, 0, ""
    total = _estimate_tokens_msgs(messages)
    if total <= max_tokens:
        return messages, 0, ""

    units: list[list] = []
    i, n = 0, len(messages)
    while i < n:
        msg = messages[i]
        unit = [msg]
        i += 1
        if getattr(msg, "tool_calls", None):
            while i < n and getattr(messages[i], "type", None) == "tool":
                unit.append(messages[i])
                i += 1
        units.append(unit)

    dropped_units: list[list] = []
    dropped = 0
    # Drop whole units from the front (oldest) but never drop the last unit.
    while len(units) > 1 and total > max_tokens:
        removed_unit = units.pop(0)
        dropped_units.append(removed_unit)
        for m in removed_unit:
            total -= _estimate_tokens(str(getattr(m, "content", "") or ""))
        dropped += len(removed_unit)

    digest = _extract_digest(dropped_units)
    result = [m for unit in units for m in unit]
    return result, dropped, digest


async def _persist_assistant_turn(
    *,
    session_obj_id,
    task_id,
    result: "AgentResult",
    message: str,
    is_new_session: bool,
    session_title: str,
    task_status: str,
) -> None:
    """Save the assistant message + close out the task, on every exit path.

    Previously this only ran on the success path — GraphRecursionError and
    generic-exception branches skipped it entirely, so a task that errored
    out stayed ``status="running"`` forever (the overview panel would show
    a phantom "in progress" task that never finishes) and any partial reply
    already streamed to the user was lost from history on refresh.
    """
    # A block can still be "running" here if the client disconnected
    # mid-tool-call (the success path's `if await request.is_disconnected():
    # break` stops consuming events without waiting for on_tool_end) or the
    # stream errored out between on_tool_start and on_tool_end. Left as-is,
    # that tool card would render a permanent spinner every time this
    # message is reloaded from history — resolve it to a neutral terminal
    # state instead.
    persisted_tool_blocks = [
        {k: v for k, v in block.items() if not k.startswith("_")}
        if block.get("status") != "running"
        else {
            **{k: v for k, v in block.items() if not k.startswith("_")},
            "status": "done",
            "success": False,
            "summary": "连接中断，执行结果未知",
        }
        for block in result.tool_blocks
    ] if result.tool_blocks else None
    async with async_session_factory() as db:
        assistant_msg = AgentMessage(
            session_id=session_obj_id,
            role="assistant",
            content=result.full_content,
            thinking=None,
            tool_blocks=persisted_tool_blocks,
        )
        db.add(assistant_msg)
        await db.flush()

        # AI-generated session title (only worth it if the turn actually
        # produced content — skip on empty/failed turns).
        if (is_new_session or session_title == "新对话") and result.full_content:
            try:
                title_prompt = (
                    f"根据以下对话内容，生成一个简短的标题（10个字以内，不要引号）：\n"
                    f"用户：{message[:200]}\nAI：{result.full_content[:200]}"
                )
                title_resp = await get_llm_client().chat.completions.create(
                    model=settings.deepseek_model,
                    extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
                    messages=[{"role": "user", "content": title_prompt}],
                    temperature=0.7,
                    max_tokens=512,
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
                task_obj.status = task_status
                task_obj.progress = 100 if task_status == "done" else task_obj.progress
                task_obj.finished_at = datetime.utcnow()

        await db.commit()


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


def _build_session_graph(agent_id: str, current: CurrentUser):
    """重建会话的工具集 + 系统提示词 + 编译图。

    chat / resume / pending 三个端点共用。resume 时历史从 checkpoint 恢复，
    这里只保证图结构与挂起时一致（同 agent、同角色 → 同工具集与能力清单）。
    """
    langchain_tools = create_langchain_tools(agent_id, current)
    visible_tool_names = {
        getattr(t, "name", "") for t in langchain_tools if getattr(t, "name", "")
    }
    system_prompt = build_system_prompt(
        agent_id,
        visible_tool_names=visible_tool_names,
        role_codes=current.roles,
    )
    graph = build_agent_graph(langchain_tools, system_prompt)
    return graph, langchain_tools, system_prompt


async def _persist_pending_confirm(
    *,
    session_obj_id,
    task_id,
    result: "AgentResult",
    interrupt_payload: dict,
) -> None:
    """挂起时保存 assistant 消息（含 confirming 块）+ 任务保持 running。

    与 _persist_assistant_turn 不同：任务不设 finished_at、不标 done，
    前端据此知道本任务尚未完结、确认卡保持可交互。resume 端点成功后会
    把这条消息的 confirming 块就地更新为 resolved。
    """
    tool_blocks = []
    for block in result.tool_blocks:
        cleaned = {k: v for k, v in block.items() if not k.startswith("_")}
        if cleaned.get("confirmation"):
            cleaned["status"] = "confirming"
        tool_blocks.append(cleaned)
    async with async_session_factory() as db:
        msg = AgentMessage(
            session_id=session_obj_id,
            role="assistant",
            content=result.full_content,
            thinking=None,
            tool_blocks=tool_blocks or None,
            handoffs={
                "type": "confirm",
                "kind": interrupt_payload.get("kind"),
                "question": interrupt_payload.get("question"),
                "options": interrupt_payload.get("options", []),
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        db.add(msg)
        await db.commit()


async def _resolve_pending_message(session_id) -> None:
    """resume 成功后把该会话最近一条挂起消息的 confirming 块更新为 resolved。

    只处理最近一条 ``handoffs.type=="confirm"`` 的 assistant 消息——旧的已
    resolved 或 expired 的跳过。
    """
    async with async_session_factory() as db:
        result = await db.execute(
            select(AgentMessage)
            .where(
                AgentMessage.session_id == session_id,
                AgentMessage.role == "assistant",
            )
            .order_by(desc(AgentMessage.created_at))
            .limit(10)
        )
        for msg in result.scalars().all():
            handoffs = msg.handoffs if isinstance(msg.handoffs, dict) else {}
            if handoffs.get("type") == "confirm":
                blocks = msg.tool_blocks or []
                changed = False
                for b in blocks:
                    if isinstance(b, dict) and b.get("status") == "confirming":
                        b["status"] = "resolved"
                        changed = True
                if changed:
                    # 坑：`msg.tool_blocks = blocks` 赋回的是同一 list 引用，普通
                    # JSON 列 set 时用 == 比较新旧值（同一对象恒等）→ 不标记 dirty，
                    # UPDATE 不含 tool_blocks，confirming→resolved 从未落库，刷新后
                    # 前端回放仍是 confirming 确认卡仍可点。必须 flag_modified 强制
                    # 标记该列（就地改过的 dict 会在 flush 时被重新序列化）。
                    flag_modified(msg, "tool_blocks")
                    msg.handoffs = {
                        **handoffs,
                        "resolved_at": datetime.utcnow().isoformat(),
                    }
                    await db.commit()
                return


async def _run_quality_guard(
    *,
    result: "AgentResult",
    compacted: list,
    langchain_tools: list,
    system_prompt: str,
    request: Request,
    allow_retry: bool = True,
) -> AsyncGenerator[str, None]:
    """QualityGuard 读后写验证 + 单次修正轮（chat 与 resume 共用）。

    - allow_retry=True 且有 compacted 历史：验证失败喂回模型做一次修正。
    - 否则（resume 无重建的历史 / 明确关闭）：只做单次验证并如实报告。
    任何异常视为通过，不阻断主流程。
    """
    try:
        if not result.tool_blocks:
            return
        guard_result = await _quality_guard.guard(
            tool_calls=result.tool_blocks,
            tool_executor=_verify_executor,
        )
        if not guard_result.passed:
            issues_text = "\n".join(f"- {i}" for i in guard_result.issues[:5])
            if allow_retry and compacted:
                # Feed the verification failures back to the model and
                # let it try to fix them (single retry).
                correction_msgs = compacted + [
                    AIMessage(content=result.full_content or ""),
                    HumanMessage(content=(
                        "[系统校验] 以下写操作未能确认生效。请先检查原因："
                        "若操作其实已生效（可能只是回读时机太早），不要重复执行；"
                        "若确实失败，重新调用对应写工具并确认成功：\n"
                        f"{issues_text}"
                    )),
                ]
                retry_result = AgentResult()
                retry_graph = build_agent_graph(langchain_tools, system_prompt)
                # QA 修正轮用独立 thread：与主线程隔离，避免污染挂起/恢复状态。
                async for sse_str in stream_agent_response(
                    retry_graph, correction_msgs, retry_result,
                    thread_id=f"qa_{uuid.uuid4().hex[:12]}",
                ):
                    if await request.is_disconnected():
                        break
                    yield sse_str
                if retry_result.full_content:
                    result.full_content = retry_result.full_content
                # 最终验证必须覆盖「第一次 + 修正轮」的写操作：
                #   - 修正轮重调过的工具名 → 只验证修正轮那次（旧的失败调用若已
                #     被后续成功调用覆盖，不应拖累整体结果）；
                #   - 修正轮未重调的工具 → 仍验证第一次那次（真实生效则通过、
                #     仍失败则如实报告）。
                retry_blocks = retry_result.tool_blocks or []
                first_blocks = list(result.tool_blocks)
                retry_names = {b.get("name") for b in retry_blocks if b.get("name")}
                unretried = [
                    b for b in first_blocks if b.get("name") not in retry_names
                ]
                if retry_blocks:
                    result.tool_blocks.extend(retry_blocks)
                final_blocks = unretried + retry_blocks
                final_check = await _quality_guard.guard(
                    tool_calls=final_blocks,
                    tool_executor=_verify_executor,
                )
                yield sse_event("verification", {
                    "verified": final_check.passed,
                    "issues": final_check.issues[:5],
                })
            else:
                # resume 路径：没有重建的历史上下文，无法安全做修正轮——
                # 如实报告未通过的验证，让用户/模型下一轮处理。
                yield sse_event("verification", {
                    "verified": False,
                    "issues": guard_result.issues[:5],
                })
        else:
            yield sse_event("verification", {"verified": True, "issues": []})
    except Exception:
        pass


def _can_see_all(current: CurrentUser) -> bool:
    """admin 凭 system:manage 通配可见全部对话（运维/审计）。其他人只见自己。"""
    return WILDCARD_PERMISSION in current.permissions


# ── Agent ID 推导 ──────────────────────────────────────────
# 一账号一角色，agent_id = role_code。equity_committee 回退到 hr。
ROLE_TO_AGENT: dict[str, str] = {
    "hr": "hr", "ceo": "ceo", "manager": "manager",
    "interviewer": "interviewer", "mentor": "mentor",
    "project_lead": "project_lead", "employee": "employee",
    "admin": "admin", "equity_committee": "equity_committee",
}


def _agent_id_for_user(current: CurrentUser) -> str:
    """从当前用户角色推导 agent_id。取第一个匹配的角色 code。"""
    for role in current.roles:
        if role in ROLE_TO_AGENT:
            return ROLE_TO_AGENT[role]
    return "employee"


async def _get_owned_session(
    db: AsyncSession, session_id: str, current: CurrentUser
) -> AgentSession | None:
    """取会话并校验归属。不属本人且非 admin → None（端点据此返 404，不返 403 防探测）。"""
    try:
        sid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(AgentSession).where(AgentSession.id == sid))
    session = result.scalar_one_or_none()
    if session is None:
        return None
    if session.owner_id is not None and session.owner_id != current.id and not _can_see_all(current):
        return None
    return session


async def _get_owned_project(
    db: AsyncSession, project_id: str, current: CurrentUser
) -> AgentProject | None:
    """取会话文件夹并校验归属。不属本人且非 admin → None。"""
    try:
        pid = uuid.UUID(project_id)
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(AgentProject).where(AgentProject.id == pid))
    project = result.scalar_one_or_none()
    if project is None:
        return None
    if project.owner_id is not None and project.owner_id != current.id and not _can_see_all(current):
        return None
    return project


def serialize_session(s: AgentSession) -> dict:
    return {
        "id": str(s.id),
        "title": s.title or "新对话",
        "agentId": s.agent_id,
        "projectId": str(s.project_id) if s.project_id else None,
        "ownerId": str(s.owner_id) if s.owner_id else None,
        "createdAt": iso_utc(s.created_at),
        "updatedAt": iso_utc(s.updated_at),
    }


def serialize_project(p: AgentProject, session_count: int = 0) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "sessionCount": session_count,
        "ownerId": str(p.owner_id) if p.owner_id else None,
        "createdAt": iso_utc(p.created_at),
        "updatedAt": iso_utc(p.updated_at),
    }


# ═══════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════

async def _build_global_overview(db: AsyncSession, current: CurrentUser) -> dict:
    """Workspace overview for all projects (global recruitment stats).

    业务统计数字（待筛简历等）保持全局；active_task 收窄到 current 可见会话范围，
    避免把别人的进行中任务显示在工作台。
    """
    # 新词表：求职中 = job_hunting（含原 new/parsed/pending_screen/pending_materials）；
    # 面试中 = round1 + round2（invited 已删除，筛选通过直接一面）。
    job_hunting = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "job_hunting")
    )).scalar() or 0
    interview_count = (await db.execute(
        select(func.count()).select_from(Candidate).where(
            Candidate.status.in_(["round1", "round2"])
        )
    )).scalar() or 0
    # 「待发 offer」必须统计 pending_offer；此前统计的是 invited（刚邀约面试），
    # 首页那个数字一直是错的。
    pending_offer = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "pending_offer")
    )).scalar() or 0
    position_count = (await db.execute(select(func.count()).select_from(Position))).scalar() or 0

    task_q = select(AgentTask).where(AgentTask.status == "running")
    if not _can_see_all(current):
        task_q = task_q.join(
            AgentSession, AgentSession.id == AgentTask.session_id
        ).where(AgentSession.owner_id == current.id)
    task_result = await db.execute(task_q.limit(1))
    active_task = task_result.scalar_one_or_none()

    # 一账号一角色：返回当前用户角色的 agent 配置
    agent_id = _agent_id_for_user(current)
    agent_cfg = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["employee"])

    return {
        "projectId": None,
        "projectName": None,
        "agents": [
            {
                "id": agent_id,
                "name": agent_cfg["name"],
                "description": agent_cfg["description"],
                "status": "就绪",
                "tone": "active",
                "icon": agent_cfg["icon"],
                "iconBg": agent_cfg["iconBg"],
                "iconColor": agent_cfg["iconColor"],
            },
        ],
        "workflowSteps": [
            {"key": "recruit", "label": "筛选", "count": int(job_hunting), "active": True, "badgeTone": "green"},
            {"key": "interview", "label": "面试", "count": int(interview_count), "active": interview_count > 0, "badgeTone": "blue"},
            {"key": "training", "label": "试用", "count": 0, "active": False, "badgeTone": "gray"},
            {"key": "performance", "label": "绩效", "count": 0, "active": False, "badgeTone": "gray"},
        ],
        "stats": [
            {"key": "resumes", "label": "求职中", "value": int(job_hunting), "hint": "待筛选", "hintTone": "up"},
            {"key": "interviews", "label": "待面试", "value": int(interview_count), "hint": "流程中", "hintTone": "default"},
            {"key": "offers", "label": "待发offer", "value": int(pending_offer), "hint": "二面已通过", "hintTone": "default"},
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


async def _build_project_overview(db: AsyncSession, project_id: str, current: CurrentUser) -> dict | None:
    """Workspace overview scoped to one project folder."""
    result = await db.execute(select(AgentProject).where(AgentProject.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        return None
    # 项目归属校验——非本人且非 admin 视为不存在（同 session 一律 404 防探测）
    if project.owner_id is not None and project.owner_id != current.id and not _can_see_all(current):
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
                # chat 端点写入的完成态是 "done"（历史上曾误写 "completed" 导致统计恒 0）
                AgentTask.status == "done",
            )
        )).scalar() or 0
        task_result = await db.execute(
            select(AgentTask)
            .where(AgentTask.session_id.in_(session_ids), AgentTask.status == "running")
            .limit(1)
        )
        active_task = task_result.scalar_one_or_none()

    base = await _build_global_overview(db, current)
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
    current: CurrentUser = Depends(get_current_user),
):
    """Workspace overview, optionally scoped to a project folder."""
    if projectId:
        data = await _build_project_overview(db, projectId, current)
        if data is None:
            return not_found("项目不存在")
    else:
        data = await _build_global_overview(db, current)

    return ok(data)


@router.get("/ai-agent/welcome-prompts")
async def get_welcome_prompts(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """欢迎页快捷入口推荐（只读缓存，对用户无感）。

    推荐由后台定时任务默默刷新（见 app lifespan 中的 welcome_prompts job），
    本接口绝不在请求路径上调用 LLM。缓存为空时返回内置默认推荐。
    """
    from app.services.ai.welcome_prompt_recommender import get_cached_welcome_prompts

    prompts = await get_cached_welcome_prompts(db)
    return ok(prompts)


@router.get("/ai-agent/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    q = (
        select(AgentSession)
        .order_by(desc(AgentSession.updated_at))
        .limit(50)
    )
    if not _can_see_all(current):
        q = q.where(AgentSession.owner_id == current.id)
    result = await db.execute(q)
    sessions = result.scalars().all()
    return ok([serialize_session(s) for s in sessions])


@router.get("/ai-agent/projects")
async def list_projects(
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """List agent conversation projects with session counts."""
    from sqlalchemy import func

    see_all = _can_see_all(current)
    count_subq = (
        select(
            AgentSession.project_id.label("project_id"),
            func.count(AgentSession.id).label("session_count"),
        )
        .where(AgentSession.project_id.isnot(None))
    )
    if not see_all:
        count_subq = count_subq.where(AgentSession.owner_id == current.id)
    count_subq = count_subq.group_by(AgentSession.project_id).subquery()

    q = (
        select(AgentProject, count_subq.c.session_count)
        .outerjoin(count_subq, AgentProject.id == count_subq.c.project_id)
        .order_by(desc(AgentProject.updated_at))
    )
    if not see_all:
        q = q.where(AgentProject.owner_id == current.id)
    result = await db.execute(q)
    rows = result.all()
    return ok([
          serialize_project(project, int(session_count or 0))
          for project, session_count in rows
      ])


class CreateProjectRequest(BaseModel):
    name: str


@router.post("/ai-agent/projects")
async def create_project(
    body: CreateProjectRequest,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    name = (body.name or "").strip()
    if not name:
        return fail(400, "项目名称不能为空")
    if len(name) > 64:
        return fail(400, "项目名称过长（最多 64 字）")

    project = AgentProject(name=name, owner_id=current.id)
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return ok(serialize_project(project, 0))


class RenameProjectRequest(BaseModel):
    name: str


@router.patch("/ai-agent/projects/{project_id}")
async def rename_project(
    project_id: str,
    body: RenameProjectRequest,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    name = (body.name or "").strip()
    if not name:
        return fail(400, "项目名称不能为空")
    if len(name) > 64:
        return fail(400, "项目名称过长（最多 64 字）")

    project = await _get_owned_project(db, project_id, current)
    if not project:
        return not_found("项目不存在")

    project.name = name
    project.updated_at = datetime.utcnow()
    await db.flush()
    return ok(serialize_project(project))


@router.delete("/ai-agent/projects/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    project = await _get_owned_project(db, project_id, current)
    if not project:
        return not_found("项目不存在")

    # Sessions become ungrouped; FK ondelete=SET NULL also handles this.
    await db.delete(project)
    return ok()


class CreateSessionRequest(BaseModel):
    projectId: Optional[str] = None
    agentId: Optional[str] = None


@router.post("/ai-agent/sessions")
async def create_session(
    body: Optional[CreateSessionRequest] = None,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """Create a new empty chat session."""
    project_id = None
    if body and body.projectId:
        project = await _get_owned_project(db, body.projectId, current)
        if not project:
            return not_found("项目不存在")
        project_id = project.id

    session = AgentSession(
        title="新对话",
        agent_id=_agent_id_for_user(current),
        project_id=project_id,
        owner_id=current.id,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)

    if project_id:
        proj = await db.get(AgentProject, project_id)
        if proj:
            proj.updated_at = datetime.utcnow()

    return ok(serialize_session(session))


@router.delete("/ai-agent/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    session = await _get_owned_session(db, session_id, current)
    if not session:
        return not_found("对话不存在")

    # Clean up MinIO objects for the session's materials before cascade delete
    mat_result = await db.execute(
        select(AgentMaterial).where(AgentMaterial.session_id == session.id)
    )
    for mat in mat_result.scalars().all():
        if mat.file_path and not (os.path.isabs(mat.file_path)):
            await asyncio.to_thread(minio_storage.delete_object, mat.file_path)

    await db.delete(session)
    return ok()


class PatchSessionRequest(BaseModel):
    title: Optional[str] = None
    projectId: Optional[str] = None


@router.patch("/ai-agent/sessions/{session_id}")
async def patch_session(
    session_id: str,
    body: PatchSessionRequest,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """更新对话名称或所属项目。"""
    session = await _get_owned_session(db, session_id, current)
    if not session:
        return not_found("对话不存在")

    if body.title is not None:
        new_title = body.title.strip()
        if not new_title:
            return fail(400, "对话名称不能为空")
        if len(new_title) > 100:
            return fail(400, "对话名称过长（最多 100 字）")
        session.title = new_title

    if body.projectId is not None:
        if body.projectId == "":
            session.project_id = None
        else:
            # 移入的文件夹必须也属本人/admin，否则对方能把对话塞进别人文件夹
            project = await _get_owned_project(db, body.projectId, current)
            if not project:
                return not_found("项目不存在")
            session.project_id = project.id
            project.updated_at = datetime.utcnow()

    session.updated_at = datetime.utcnow()
    await db.flush()
    return ok(serialize_session(session))


@router.get("/ai-agent/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    session = await _get_owned_session(db, session_id, current)
    if not session:
        return not_found("对话不存在")

    result = await db.execute(
        select(AgentMessage)
        .where(AgentMessage.session_id == session.id)
        .order_by(AgentMessage.created_at)
    )
    messages = result.scalars().all()
    return ok([
          {
              "id": str(m.id),
              "sessionId": str(m.session_id),
              "role": m.role,
              "content": m.content,
              "meta": m.handoffs if m.role == "user" and isinstance(m.handoffs, dict) else None,
              "toolBlocks": m.tool_blocks if isinstance(m.tool_blocks, list) else None,
              "createdAt": iso_utc(m.created_at),
          }
          for m in messages
      ])


@router.get("/ai-agent/sessions/{session_id}/materials")
async def get_session_materials(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """会话已上传素材列表（页面刷新后恢复右栏文件区）。"""
    session = await _get_owned_session(db, session_id, current)
    if not session:
        return not_found("对话不存在")

    result = await db.execute(
        select(AgentMaterial)
        .where(AgentMaterial.session_id == session.id)
        .order_by(AgentMaterial.uploaded_at)
    )
    materials = result.scalars().all()
    return ok([
          {
              "id": str(m.id),
              "name": m.name,
              "type": m.type,
              "ingested": bool(m.file_path and m.file_path.startswith("resumes/")),
              "uploadedAt": iso_utc(m.uploaded_at),
          }
          for m in materials
      ])


@router.get("/ai-agent/sessions/{session_id}/tasks")
async def get_session_tasks(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """会话任务列表（右栏工作状态：最近任务与进行中任务）。"""
    session = await _get_owned_session(db, session_id, current)
    if not session:
        return not_found("对话不存在")

    result = await db.execute(
        select(AgentTask)
        .where(AgentTask.session_id == session.id)
        .order_by(desc(AgentTask.started_at))
        .limit(20)
    )
    tasks = result.scalars().all()
    return ok([
          {
              "id": str(t.id),
              "title": t.title or "",
              "description": t.description or "",
              "progress": t.progress or 0,
              "status": t.status or "",
              "startedAt": iso_utc(t.started_at),
              "finishedAt": iso_utc(t.finished_at),
          }
          for t in tasks
      ])


@router.post("/ai-agent/sessions/{session_id}/messages")
async def append_session_message(
    session_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """Append a lightweight user/assistant notice (e.g. file-ingest progress) to a session."""
    role = (body.get("role") or "assistant").strip()
    content = (body.get("content") or "").strip()
    if role not in ("user", "assistant") or not content:
        return fail(400, "请提供 role 与 content")

    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        return fail(400, "sessionId 无效")

    session = await db.get(AgentSession, sid)
    if session:
        # 已存在 → 校验归属（不属本人且非 admin 视为不存在）
        if session.owner_id is not None and session.owner_id != current.id and not _can_see_all(current):
            return not_found("对话不存在")
    else:
        session = AgentSession(id=sid, title="新对话", agent_id=_agent_id_for_user(current), owner_id=current.id)
        db.add(session)
        await db.flush()

    msg = AgentMessage(session_id=sid, role=role, content=content)
    db.add(msg)
    session.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(msg)

    return ok({
          "id": str(msg.id),
          "sessionId": str(msg.session_id),
          "role": msg.role,
          "content": msg.content,
          "meta": None,
          "createdAt": iso_utc(msg.created_at),
      })


@router.post("/ai-agent/materials/from-candidates")
async def materials_from_candidates(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """Attach already-ingested candidates as AgentMaterial for the active session."""
    candidate_ids = body.get("candidateIds") or []
    if not candidate_ids:
        return fail(400, "请提供 candidateIds")

    # 取"最新会话"必须收窄到本人，否则会把材料挂到别人的会话上
    session_q = select(AgentSession).order_by(desc(AgentSession.updated_at)).limit(1)
    if not _can_see_all(current):
        session_q = session_q.where(AgentSession.owner_id == current.id)
    session_result = await db.execute(session_q)
    session = session_result.scalar_one_or_none()
    if not session:
        session = AgentSession(title="新对话", agent_id=_agent_id_for_user(current), owner_id=current.id)
        db.add(session)
        await db.flush()

    from app.api.recruitment.resumes import CANDIDATE_LOAD_OPTIONS

    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id.in_(candidate_ids))
    )
    by_id = {str(c.id): c for c in result.scalars().all()}

    out: list[dict] = []
    for cid in candidate_ids:
        candidate = by_id.get(str(cid))
        if not candidate:
            continue
        filename = (
            os.path.basename(candidate.resume_file)
            if candidate.resume_file
            else (candidate.name or "resume")
        )
        material = AgentMaterial(
            session_id=session.id,
            name=filename,
            type="resume",
            file_path=candidate.resume_file or "",
        )
        db.add(material)
        await db.flush()
        await db.refresh(material)
        out.append({
            "id": str(material.id),
            "name": material.name,
            "type": "resume",
            "uploadedAt": iso_utc(material.uploaded_at),
            "ingested": True,
            "candidateId": str(candidate.id),
            "position": candidate.position.name if candidate.position else "",
            "positionId": str(candidate.position_id) if candidate.position_id else "",
            "score": candidate.score or 0,
            "status": candidate.status or "job_hunting",
            "skills": [s.skill for s in (candidate.skills or [])],
        })

    return ok(out)


@router.post("/ai-agent/materials/ingest-batch")
async def ingest_materials_batch(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """Ingest previously staged session attachments (agent/* keys) into resume DB.

    并发度 3：每个待入库文件走独立 DB session 并行处理，互不阻塞。
    已在库中的文件（resumes/ 前缀）和非法类型走快速路径，串行完成。
    """
    material_ids = body.get("materialIds") or []
    if not material_ids:
        return fail(400, "请提供 materialIds")

    from app.api.recruitment.resume_upload import _upload_one_resume, _load_candidate, launch_resume_scoring, launch_score_task
    import os

    # 材料经 session 间接归属——非 admin 仅能取到本人会话下的材料，别人的 material_id 静默跳过
    mat_q = select(AgentMaterial).where(AgentMaterial.id.in_(material_ids))
    if not _can_see_all(current):
        mat_q = mat_q.join(
            AgentSession, AgentSession.id == AgentMaterial.session_id
        ).where(AgentSession.owner_id == current.id)
    result = await db.execute(mat_q)
    by_id = {str(m.id): m for m in result.scalars().all()}

    # ── Phase 1: classify — fast-path vs need-processing ──────────
    # mid → pre-built item for fast-path cases (already ingested / wrong type /
    # download failure); these get placed directly in the output stream.
    fast_items: dict[str, dict] = {}
    # Items that need full _upload_one_resume: (mid, name, file_ext, content, fp)
    to_process: list[tuple] = []

    for mid in material_ids:
        mat = by_id.get(str(mid))
        if not mat:
            continue

        fp = (mat.file_path or "").strip()
        if not fp:
            continue

        # ── Already ingested (resumes/ prefix) ──
        if fp.startswith("resumes/"):
            try:
                cand_result = await db.execute(
                    select(Candidate).where(Candidate.resume_file == fp).limit(1)
                )
                candidate = cand_result.scalar_one_or_none()
            except Exception:
                candidate = None
            if candidate:
                fast_items[mid] = {
                    "id": str(mat.id), "name": mat.name, "type": mat.type,
                    "uploadedAt": iso_utc(mat.uploaded_at),
                    "ingested": True,
                    "candidateId": str(candidate.id),
                    "position": candidate.position.name if candidate.position else "",
                    "positionId": str(candidate.position_id) if candidate.position_id else "",
                    "score": candidate.score or 0,
                    "status": candidate.status or "job_hunting",
                    "skills": [s.skill for s in (candidate.skills or [])],
                }
                continue

        # ── Wrong type ──
        if mat.type not in ("resume", "file", "jd", "material"):
            fast_items[mid] = {
                "id": str(mat.id), "name": mat.name, "type": mat.type,
                "uploadedAt": iso_utc(mat.uploaded_at),
                "ingested": False,
            }
            continue

        # ── Download from MinIO (must be serial — filesystem I/O) ──
        try:
            tmp_path = await asyncio.to_thread(minio_storage.download_to_temp, fp)
            try:
                with open(tmp_path, "rb") as f:
                    content = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            fast_items[mid] = {
                "id": str(mat.id), "name": mat.name, "type": mat.type,
                "uploadedAt": iso_utc(mat.uploaded_at),
                "ingested": False,
                "analysis": {"summary": f"读取文件失败: {e}"},
            }
            continue

        file_ext = os.path.splitext(mat.name)[1].lower() or os.path.splitext(fp)[1].lower() or ".pdf"
        to_process.append((mid, mat.name, file_ext, content, fp))

    # ── Phase 2: parallel ingest (Semaphore=3, own DB session per task) ──
    sem = asyncio.Semaphore(3)

    async def _process_one(mid, name, file_ext, content, fp):
        """Process ONE material in its own DB session + transaction."""
        async with sem:
            async with async_session_factory() as task_db:
                try:
                    upload_result = await _upload_one_resume(
                        task_db,
                        original_name=name,
                        content=content,
                        file_ext=file_ext,
                    )
                    status = upload_result.get("status")

                    # ── Build item & collect side-effect info ──
                    if status == "success":
                        data = upload_result.get("data") or {}
                        candidate_id = data.get("id")
                        candidate = await _load_candidate(task_db, candidate_id) if candidate_id else None
                        item: dict = {
                            "id": str(mid), "name": name, "type": "resume",
                            "ingested": True,
                        }
                        if candidate:
                            item.update({
                                "candidateId": str(candidate.id),
                                "position": candidate.position.name if candidate.position else "",
                                "positionId": str(candidate.position_id) if candidate.position_id else "",
                                "score": candidate.score or 0,
                                "status": candidate.status or "job_hunting",
                                "skills": [s.skill for s in (candidate.skills or [])],
                            })
                        await task_db.commit()
                        # 后台评分在 commit 之后启动，否则独立 session 读不到候选人
                        launch_resume_scoring(upload_result)
                        return {"mid": mid, "item": item,
                                "resume_file": candidate.resume_file if candidate else None,
                                "original_fp": fp}

                    elif status == "duplicate":
                        existing_id = upload_result.get("existingCandidateId")
                        candidate = await _load_candidate(task_db, existing_id) if existing_id else None
                        item = {
                            "id": str(mid), "name": name, "type": "resume",
                            "ingested": True,
                        }
                        if candidate:
                            item.update({
                                "candidateId": str(candidate.id),
                                "position": candidate.position.name if candidate.position else "",
                                "positionId": str(candidate.position_id) if candidate.position_id else "",
                                "score": candidate.score or 0,
                                "status": candidate.status or "job_hunting",
                                "skills": [s.skill for s in (candidate.skills or [])],
                            })
                        await task_db.commit()
                        return {"mid": mid, "item": item,
                                "resume_file": candidate.resume_file if candidate else None,
                                "original_fp": fp}

                    else:
                        return {"mid": mid, "item": {
                            "id": str(mid), "name": name, "type": "resume",
                            "ingested": False,
                            "analysis": {"summary": upload_result.get("message") or "入库失败"},
                        }}

                except Exception as exc:
                    return {"mid": mid, "item": {
                        "id": str(mid), "name": name, "type": "resume",
                        "ingested": False,
                        "analysis": {"summary": f"入库异常: {exc}"},
                    }}

    # Run all processable items in parallel; keep order via result map
    if to_process:
        results = await asyncio.gather(*[
            _process_one(mid, name, file_ext, content, fp)
            for mid, name, file_ext, content, fp in to_process
        ])
        result_by_mid: dict[str, dict] = {r["mid"]: r for r in results}
    else:
        result_by_mid = {}

    # ── Phase 3: assemble output (preserve material_ids order) ─────
    out: list[dict] = []
    for mid in material_ids:
        if mid in fast_items:
            out.append(fast_items[mid])
            continue

        r = result_by_mid.get(mid)
        if not r:
            continue

        # Update AgentMaterial in the request session to reflect new state
        mat = by_id.get(mid)
        if mat:
            resume_file = r.get("resume_file")
            original_fp = r.get("original_fp")
            if resume_file:
                mat.file_path = resume_file
                mat.type = "resume"
            if original_fp and original_fp.startswith("agent/"):
                try:
                    await asyncio.to_thread(minio_storage.delete_object, original_fp)
                except Exception:
                    pass

        out.append(r["item"])

    await db.commit()
    return ok(out)


@router.post("/ai-agent/materials")
async def upload_material(
    type: str = Form(...),
    file: UploadFile = File(None),
    knowledgeId: str = Form(None),
    knowledgeName: str = Form(None),
    autoIngest: str = Form("true"),
    sessionId: str = Form(None),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    """Upload a material (resume/document/knowledge) for the current session.

    Resumes (type="resume" or "file") are auto-detected: if the content parses
    as a real resume the file goes through the same full ingestion pipeline as
    the dedicated POST /api/resumes/upload endpoint — MinIO storage, position
    matching, Candidate creation, parsing, scoring, RAG ingest.  Set
    ``autoIngest=false`` to store as session attachment only (analyze later).
    Non-resume files are stored as session attachments with a plain AI analysis preview.
    """
    should_ingest = autoIngest.strip().lower() not in ("false", "0", "no")
    # Get or create session — prefer the explicit sessionId from the caller
    # (falling back to "latest session" keeps老前端兼容，但有并发竞态)
    session = None
    if sessionId:
        try:
            session = await db.get(AgentSession, uuid.UUID(sessionId))
        except ValueError:
            session = None
        # 显式 sessionId 必须属本人/admin，否则视为不存在
        if session and session.owner_id is not None and session.owner_id != current.id and not _can_see_all(current):
            return not_found("对话不存在")
    if not session:
        session_q = select(AgentSession).order_by(desc(AgentSession.updated_at)).limit(1)
        if not _can_see_all(current):
            session_q = session_q.where(AgentSession.owner_id == current.id)
        session_result = await db.execute(session_q)
        session = session_result.scalar_one_or_none()
    if not session:
        session = AgentSession(title="新对话", agent_id=_agent_id_for_user(current), owner_id=current.id)
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
        return ok({
              "id": str(material.id), "name": material.name, "type": material.type,
              "uploadedAt": iso_utc(material.uploaded_at),
              "analysis": None,
          })

    # ── File upload: detect whether this is a real resume ───
    filename = file.filename or "material"
    file_ext = os.path.splitext(filename)[1].lower() or ".pdf"
    content = await file.read()

    # ── 全局去重：同内容（SHA256）已入库 → 复用已有素材，不重复入库 ──
    content_hash = hashlib.sha256(content).hexdigest() if content else ""
    if content_hash:
        dup_result = await db.execute(
            select(AgentMaterial).where(AgentMaterial.content_hash == content_hash).limit(1)
        )
        dup = dup_result.scalar_one_or_none()
        if dup:
            # 复用已有素材：不写 MinIO、不建 Candidate、不落新记录
            return ok({
                "id": str(dup.id), "name": dup.name, "type": dup.type,
                "knowledgeId": str(dup.knowledge_id) if dup.knowledge_id else None,
                "uploadedAt": iso_utc(dup.uploaded_at),
                "ingested": bool(dup.file_path and dup.file_path.startswith("resumes/")),
                "analysis": None,
                "duplicate": True,
            }, message="该文件已存在，已复用已有素材，不重复入库")

    # Save to a temp file on disk so extract_text_from_file can read it.
    # (extract_text_from_file handles both local paths and MinIO keys;
    #  we give it a local temp so we can decide the final MinIO prefix
    #  after detection without uploading twice.)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tf:
        tf.write(content)
        tmp_path = tf.name

    try:
        text, extract_error = await extract_text_from_file(tmp_path)
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

    if is_resume and type in ("resume", "file") and should_ingest:
        # ── Full DB ingestion (same as dedicated endpoint) ──

        # 1) Store to MinIO with proper resumes/ prefix
        object_key = f"resumes/{uuid.uuid4()}{file_ext}"
        try:
            await asyncio.to_thread(
                minio_storage.upload_bytes, object_key, content, "application/octet-stream"
            )
        except Exception as e:
            return fail(500, f"简历存储失败: {e}")

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
            return fail(400, f"AI 未能判断该简历的应聘岗位：{match_reason or '无匹配'}。"
                    "请手动选择岗位后重新上传。")

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
        candidate = await load_candidate(db, candidate.id)
        score_task = None
        if candidate:
            _, score_task = await run_resume_parse(candidate, position_name, db)

        # 5) RAG knowledge-base ingest 已断开（2026-08-01）——知识库/RAG 模块停用
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
                "candidateId": str(candidate.id),
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
            file_path=object_key, content_hash=content_hash,
        )
        db.add(material)
        await db.flush()
        await db.refresh(material)

        # 后台评分在 commit 之后启动（独立 session 需读到已提交的候选人）
        await db.commit()
        launch_score_task(score_task)

        return ok({
              "id": str(material.id),
              "name": material.name,
              "type": "resume",
              "uploadedAt": iso_utc(material.uploaded_at),
              "ingested": True,
              **resp_data,
          }, message="简历已入库")

    # ── Not a resume (or type=knowledge) → old attachment flow ──
    object_key = f"agent/{uuid.uuid4()}{file_ext}"
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return fail(500, f"资料存储失败: {e}")

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

    return ok({
          "id": str(material.id), "name": material.name, "type": material.type,
          "knowledgeId": str(material.knowledge_id) if material.knowledge_id else None,
          "uploadedAt": iso_utc(material.uploaded_at),
          "ingested": False,
          "analysis": analysis,
      })


# ── Plan Generation Helper (removed) ──────────────────────
# Multi-step execution is handled natively by the ReAct loop; the separate
# plan-generation LLM call and complexity classifier were removed during the
# over-engineering cleanup.


@router.post("/ai-agent/suggestions/{suggestion_id}/trigger")
async def trigger_suggestion(
    suggestion_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
):
    return ok()


@router.post("/ai-agent/chat")
async def agent_chat(
    request: Request,
    body: dict,
    current: CurrentUser = Depends(get_current_user),
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
    display_message = (body.get("displayMessage") or message or "").strip()
    message_meta = body.get("messageMeta")
    finalize_materials = body.get("finalizeMaterials")
    session_id = body.get("sessionId")
    # agent_id 不由前端指定（不可信）。已有会话从 session.agent_id 读取；
    # 新建会话由当前角色推导（_agent_id_for_user）。这里先按角色兜底，
    # session 确定后若 session.agent_id 有值则覆盖。
    agent_id = _agent_id_for_user(current)
    material_ids = body.get("materialIds", [])
    ingestion_completed = bool(body.get("ingestionCompleted"))
    user_message_persisted = bool(body.get("userMessagePersisted"))

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
                    id_line = f"materialId: {mat.id}" + (f", fileKey: {fp}" if fp else "")
                    # ── Ingested resume → compact candidate reference ──
                    if mat.type in ("resume", "file") and fp.startswith("resumes/"):
                        lines.append(f"\n[已入库简历] {mat.name} ({id_line})")
                        try:
                            from sqlalchemy.orm import selectinload

                            async with async_session_factory() as db2:
                                cand_result = await db2.execute(
                                    select(Candidate)
                                    .options(selectinload(Candidate.skills))
                                    .where(Candidate.resume_file == fp)
                                    .limit(1)
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
                                lines.append(
                                    "（文件路径为 resumes/ 但未找到候选人，可能已被清空删除，"
                                    "不可当作已入库，勿重复 upload_resume）"
                                )
                        except Exception:
                            lines.append("（已入库，查找候选人信息失败）")
                    # ── Knowledge picker ──
                    elif mat.type == "knowledge":
                        lines.append(f"\n[知识库素材] {mat.name} ({id_line})")
                        if mat.knowledge_id:
                            lines.append(f"（knowledgeId: {mat.knowledge_id}）")
                    # ── Other file → extract text (clipped) ──
                    elif fp:
                        lines.append(f"\n[{mat.type}] {mat.name} ({id_line})")
                        try:
                            file_text, _ = await extract_text_from_file(fp)
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
            # 显式 sessionId 必须属本人/admin；属他人 → 404 防探测
            if session and session.owner_id is not None and session.owner_id != current.id and not _can_see_all(current):
                return JSONResponse(
                    status_code=404,
                    content={"code": 404, "message": "对话不存在", "data": None},
                )

        if not session:
            is_new_session = True
            client_sid = None
            if session_id:
                try:
                    client_sid = uuid.UUID(session_id)
                except ValueError:
                    client_sid = None
            session = AgentSession(
                id=client_sid or uuid.uuid4(),
                title="新对话",
                agent_id=_agent_id_for_user(current),
                owner_id=current.id,
            )
            db.add(session)
            await db.flush()
            session_id = str(session.id)

        # agent_id 以数据库为准：旧会话保留其 agent_id，新会话已按角色推导
        if session.agent_id:
            agent_id = session.agent_id

        # Save user message (display text for UI; model prompt may differ)
        if not user_message_persisted:
            user_msg = AgentMessage(
                session_id=session.id,
                role="user",
                content=display_message,
                handoffs=message_meta if isinstance(message_meta, dict) else None,
            )
            db.add(user_msg)
            await db.flush()

        if finalize_materials and isinstance(finalize_materials, list):
            pending_result = await db.execute(
                select(AgentMessage)
                .where(
                    AgentMessage.session_id == session.id,
                    AgentMessage.role == "user",
                )
                .order_by(desc(AgentMessage.created_at))
                .limit(8)
            )
            for prior in pending_result.scalars().all():
                meta = prior.handoffs if isinstance(prior.handoffs, dict) else {}
                if meta.get("pendingAttachments"):
                    prior.handoffs = {
                        **meta,
                        "pendingAttachments": None,
                        "materials": finalize_materials,
                    }
                    break

        # Update session title（取首行并清洗，避免换行/首尾空白撑破标题 UI）
        if session.title in ("新对话", None, ""):
            first_line = (display_message or "").strip().split("\n")[0].strip()
            session.title = first_line[:50] if first_line else "新对话"
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
                        content=str(tb["result"])[:3000],
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
                yield sse_event("meta", {"traceId": trace_id, "sessionId": str(session_id)})

            # Send initial thinking
            thinking_text = "正在思考…"
            yield sse_event("thinking", {"text": thinking_text, "append": False})

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

            # ── Tools: the model always sees the full tool set ──
            # No intent gating. The ReAct loop picks the right tool and can
            # self-correct across turns; tool descriptions carry the
            # disambiguation that keyword gating used to enforce.
            langchain_tools = create_langchain_tools(agent_id, current)

            _INGEST_UPLOAD_TOOLS = frozenset({
                "upload_resume", "batch_parse_resumes",
            })
            if ingestion_completed:
                langchain_tools = [
                    t for t in langchain_tools
                    if getattr(t, "name", None) not in _INGEST_UPLOAD_TOOLS
                ]

            # 能力清单由「实际注入模型的可见工具」反向驱动，与工具 fail-closed 过滤同源，
            # 保证 AI 宣称的能力严格等于它能调的工具。
            visible_tool_names = {
                getattr(t, "name", "") for t in langchain_tools if getattr(t, "name", "")
            }
            # 叠加当前账号角色专属段（prompts/roles/<role>.txt）——身份/职责视角/
            # 话术按角色差异化，工具可见性仍由 visible_tool_names 同源驱动。
            system_prompt = build_system_prompt(
                agent_id,
                visible_tool_names=visible_tool_names,
                role_codes=current.roles,
            )

            if ingestion_completed:
                system_prompt += (
                    "\n\n[系统] 附件已由前端自动完成批量入库，候选人记录已写入数据库。"
                    "请仅根据附件资料向用户汇总入库结果，禁止再调用 upload_resume、"
                    "batch_parse_resumes 等上传/入库工具。"
                )

            # ── Context compaction: keep history within token budget ──
            # 若发生截断，把被丢弃消息里提取的实体要点注入 system prompt，
            # 避免长对话早期确立的关键事实（候选人名/ID/岗位名）被静默丢弃。
            compacted, _dropped, history_digest = _compact_messages(lc_messages, max_tokens=6000)
            if history_digest:
                system_prompt += "\n\n[历史要点] " + history_digest

            graph = build_agent_graph(langchain_tools, system_prompt)

            # Stream agent execution
            async for sse_str in stream_agent_response(
                graph, compacted, result, thread_id=str(session_obj_id)
            ):
                if await request.is_disconnected():
                    break
                yield sse_str

            # ── 用户决策挂起：保存 confirming 块，等前端调 resume ──
            if result.interrupted:
                await _persist_pending_confirm(
                    session_obj_id=session_obj_id,
                    task_id=task_id,
                    result=result,
                    interrupt_payload=result.interrupt_payload or {},
                )
                yield sse_event("done", {})
                return

            # ── Quality Guard: verify writes actually took effect ──
            # Read-after-write confirmation. If a write can't be confirmed, we
            # inject the issue back into the agent for ONE correction pass
            # instead of just reporting it and marking the task done.
            async for sse_str in _run_quality_guard(
                result=result,
                compacted=compacted,
                langchain_tools=langchain_tools,
                system_prompt=system_prompt,
                request=request,
            ):
                if await request.is_disconnected():
                    break
                yield sse_str

            # Save assistant message + update task (short-lived session)
            await _persist_assistant_turn(
                session_obj_id=session_obj_id,
                task_id=task_id,
                result=result,
                message=message,
                is_new_session=is_new_session,
                session_title=session_title,
                task_status="done",
            )

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
        except GraphRecursionError:
            # Task needed more tool-calling rounds than the recursion budget.
            # Any prior writes already committed; give the user a readable note
            # instead of a raw stack-trace-style error.
            note = (
                "\n\n（这个任务步骤较多，我已尽力执行到当前进度。"
                "如果还没完成，请把剩下的部分再说一次，我接着做。）"
            )
            result.full_content += note
            yield sse_event("content", {"delta": note})
            try:
                await _persist_assistant_turn(
                    session_obj_id=session_obj_id,
                    task_id=task_id,
                    result=result,
                    message=message,
                    is_new_session=is_new_session,
                    session_title=session_title,
                    task_status="done",
                )
            except Exception:
                # Best-effort — the user already saw the note above even if
                # persistence fails, so don't turn this into a hard error.
                pass
            yield sse_event("done", {})
        except Exception as e:
            # Persist whatever partial reply/tool results were streamed
            # before the failure, and close out the task as "failed" instead
            # of leaving it stuck at "running" forever (previously this
            # branch skipped persistence entirely — see _persist_assistant_turn
            # docstring).
            if not result.full_content:
                result.full_content = "（抱歉，这次回复出现异常，请重试。）"
            try:
                await _persist_assistant_turn(
                    session_obj_id=session_obj_id,
                    task_id=task_id,
                    result=result,
                    message=message,
                    is_new_session=is_new_session,
                    session_title=session_title,
                    task_status="failed",
                )
            except Exception:
                pass
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


async def _confirm_expired(session_id) -> bool:
    """该会话最近一条挂起确认是否已超时（>30 分钟）或已被标过期/已处理。

    resume 端点在恢复 checkpoint 前校验，与 app.py 的定时清理口径一致；
    幂等返回 True 时调用方清 checkpoint 并回 410。
    """
    from datetime import timedelta
    async with async_session_factory() as db:
        result = await db.execute(
            select(AgentMessage)
            .where(
                AgentMessage.session_id == session_id,
                AgentMessage.role == "assistant",
            )
            .order_by(desc(AgentMessage.created_at))
            .limit(10)
        )
        for m in result.scalars().all():
            handoffs = m.handoffs if isinstance(m.handoffs, dict) else {}
            if handoffs.get("type") != "confirm":
                continue
            if handoffs.get("expired_at") or handoffs.get("resolved_at"):
                return True
            raw = handoffs.get("created_at") or m.created_at
            try:
                created = datetime.fromisoformat(str(raw)) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                created = m.created_at
            return created is not None and created < datetime.utcnow() - timedelta(minutes=30)
    return False


async def _check_session_ownership(session_id: str, current: CurrentUser) -> AgentSession | None:
    """校验会话存在且属本人/admin；否则返回 None（调用方回 404 防探测）。"""
    async with async_session_factory() as db:
        result = await db.execute(
            select(AgentSession).where(AgentSession.id == session_id)
        )
        session = result.scalar_one_or_none()
    if not session:
        return None
    if session.owner_id is not None and session.owner_id != current.id and not _can_see_all(current):
        return None
    return session


@router.post("/ai-agent/sessions/{session_id}/resume")
async def agent_resume(
    request: Request,
    session_id: str,
    body: dict,
    current: CurrentUser = Depends(get_current_user),
):
    """Resume a paused agent conversation after a user decision.

    图从 checkpoint 恢复（thread_id = session_id），不传 messages——历史与
    挂起状态都在 checkpoint 里。SSE 事件格式与 /chat 一致，前端可复用同一
    解析逻辑。
    """
    choice = (body.get("choice") or "").strip()
    session = await _check_session_ownership(session_id, current)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "对话不存在", "data": None},
        )

    agent_id = session.agent_id or _agent_id_for_user(current)
    graph, langchain_tools, system_prompt = _build_session_graph(agent_id, current)

    config = {"configurable": {"thread_id": str(session.id)}}
    try:
        state = await graph.aget_state(config)
    except Exception:
        state = None
    has_pending = bool(state and state.next and state.tasks)
    if not has_pending:
        # 没有挂起的确认（可能已 resume / 已过期 / 进程重启丢内存）——幂等告知
        return JSONResponse(
            status_code=410,
            content={"code": 410, "message": "该对话没有待确认的操作", "data": None},
        )

    # 超时校验：挂起确认超过 30 分钟视为过期（与 app.py 定时清理同口径），
    # 即时清掉 checkpoint，避免用户点选后才报错。
    if await _confirm_expired(session.id):
        try:
            # CompiledStateGraph 不暴露 adelete_thread，走底层 checkpointer
            await graph.checkpointer.adelete_thread(str(session.id))
        except Exception:
            pass
        return JSONResponse(
            status_code=410,
            content={"code": 410, "message": "该确认已超过 30 分钟未操作，已过期，请重新发起。", "data": None},
        )

    async def resume_stream() -> AsyncGenerator[str, None]:
        result = AgentResult()
        task_id = None

        try:
            trace_id = get_trace_id()
            if trace_id:
                yield sse_event("meta", {"traceId": trace_id, "sessionId": str(session.id)})
            yield sse_event("thinking", {"text": "正在继续执行…", "append": False})

            # 本次 resume 也建一个任务用于追踪
            async with async_session_factory() as db:
                task = AgentTask(
                    session_id=session.id,
                    title=f"确认：{choice[:50]}" if choice else "继续执行确认操作",
                    description=choice,
                    progress=0,
                    status="running",
                    started_at=datetime.utcnow(),
                )
                db.add(task)
                await db.flush()
                task_id = task.id
                await db.commit()

            # resume_value 非空 → stream_agent_response 内部用 Command(resume=choice)
            # 从 checkpoint 恢复并继续执行（tools 不重跑、confirm_gate 拿回选择）。
            async for sse_str in stream_agent_response(
                graph, None, result,
                thread_id=str(session.id),
                resume_value=choice,
            ):
                if await request.is_disconnected():
                    break
                yield sse_str

            # resume 无重建的历史上下文，QA 只做单次验证、不做修正轮
            async for sse_str in _run_quality_guard(
                result=result,
                compacted=[],
                langchain_tools=langchain_tools,
                system_prompt=system_prompt,
                request=request,
                allow_retry=False,
            ):
                if await request.is_disconnected():
                    break
                yield sse_str

            # 保存本轮 assistant 消息 + 关闭任务；再把上一条挂起消息的
            # confirming 块就地更新为 resolved（前端不再渲染确认按钮）。
            await _persist_assistant_turn(
                session_obj_id=session.id,
                task_id=task_id,
                result=result,
                message=f"（用户确认：{choice}）",
                is_new_session=False,
                session_title=session.title or "新对话",
                task_status="done",
            )
            await _resolve_pending_message(session.id)

            yield sse_event("phase_result", {
                "id": f"phase_{uuid.uuid4().hex[:6]}",
                "tone": "success",
                "title": "任务完成",
                "description": f"已按您的选择继续：{choice[:80]}",
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
            if not result.full_content:
                result.full_content = "（抱歉，这次回复出现异常，请重试。）"
            try:
                await _persist_assistant_turn(
                    session_obj_id=session.id,
                    task_id=task_id,
                    result=result,
                    message=f"（用户确认：{choice}）",
                    is_new_session=False,
                    session_title=session.title or "新对话",
                    task_status="failed",
                )
            except Exception:
                pass
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        resume_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/ai-agent/sessions/{session_id}/pending")
async def agent_pending(
    session_id: str,
    current: CurrentUser = Depends(get_current_user),
):
    """返回会话是否有待确认的挂起操作（前端刷新恢复 / 轮询用）。

    挂起状态存在 LangGraph checkpoint（内存），进程重启会丢——查询不到时
    返回 pending:false，前端自然降级为不可恢复的普通历史消息。
    """
    session = await _check_session_ownership(session_id, current)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "对话不存在", "data": None},
        )

    agent_id = session.agent_id or _agent_id_for_user(current)
    graph, _, _ = _build_session_graph(agent_id, current)
    config = {"configurable": {"thread_id": str(session.id)}}
    try:
        state = await graph.aget_state(config)
    except Exception:
        state = None
    if not (state and state.next and state.tasks):
        return ok({"pending": False})

    intr = None
    for t in state.tasks:
        if getattr(t, "interrupts", None):
            intr = t.interrupts[0]
            break
    if intr is None:
        return ok({"pending": False})
    value = getattr(intr, "value", None) or {}
    return ok({
        "pending": True,
        "kind": value.get("kind"),
        "question": value.get("question"),
        "options": value.get("options", []),
        "interruptId": getattr(intr, "id", ""),
    })
