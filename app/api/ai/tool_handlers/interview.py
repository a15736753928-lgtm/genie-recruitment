"""Interview tool handlers."""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_result import ToolResult, DisplayHint


def _round_label(round_key: str) -> str:
    return "一面" if round_key == "first" else "二面"


def _is_uuid(value) -> bool:
    try:
        _uuid.UUID(str(value))
        return True
    except (ValueError, TypeError):
        return False


async def _resolve_candidate_id(db: AsyncSession, candidate_id: str) -> str | None:
    """LLM 偶发把 candidateId 传成中文名而非 UUID（实测传「崔剑宏」）。

    直接 `Candidate.id == 中文名` 在 Postgres UUID 列上抛 DataError → 被
    execute_tool_call 转成「工具执行失败」，确认卡/出题都走不通。这里统一
    解析为真实 UUID：合法 UUID 直接用；否则按姓名精确/模糊查。
    找不到返回 None。
    """
    if _is_uuid(candidate_id):
        return str(candidate_id)
    from app.models.recruitment import Candidate
    from sqlalchemy import select

    name = str(candidate_id or "").strip()
    if not name:
        return None
    row = (await db.execute(
        select(Candidate).where(Candidate.name == name).limit(1)
    )).scalar_one_or_none()
    if row:
        return str(row.id)
    row = (await db.execute(
        select(Candidate).where(Candidate.name.ilike(f"%{name}%")).limit(1)
    )).scalar_one_or_none()
    return str(row.id) if row else None


async def _get_questions(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import query_interview_questions, normalize_interview_round
    from app.models.recruitment import Candidate
    from sqlalchemy import select

    candidate_id = params["candidateId"]
    round_key = normalize_interview_round(params.get("round") or "first")
    round_label = _round_label(round_key)

    result = await query_interview_questions(
        db, candidate_id=candidate_id, round=round_key,
    )
    if result.get("code") != 0:
        return f"❌ 面试题查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", {}) or {}
    questions = data.get("questions", [])
    if questions:
        lines = [f"候选人 {candidate_id} 的{round_label}面试题（共 {len(questions)} 道）："]
        for q in questions:
            lines.append(
                f"  [{q.get('index', '?')}] {q.get('content', '')} | "
                f"分类:{q.get('category', '')} | 难度:{q.get('difficulty', '')} | ID:{q.get('id', '')}"
            )
        return "\n".join(lines)

    cand = (await db.execute(select(Candidate).where(Candidate.id == candidate_id))).scalar_one_or_none()
    if cand and cand.status == "job_hunting":
        return (
            f"暂无{round_label}面试题。候选人当前为「求职中」，需先邀约一面"
            f"（状态改为 round1）后再生成{round_label}面试题；或使用 generate_questions 强制出题。"
        )
    return f"暂无{round_label}面试题，请使用 generate_questions 生成"


async def _generate_questions(params: dict, db: AsyncSession) -> str | ToolResult:
    from app.api.talent.interview import (
        regenerate_questions as fn,
        normalize_interview_round,
        INTERVIEW_ELIGIBLE_STATUSES,
    )
    from app.models.recruitment import Candidate
    from sqlalchemy import select
    from app.api.ai.tool_executor import get_current_user_for_tools

    # LLM 偶发把 candidateId 传成中文名而非 UUID，统一解析为真实 id 后再走业务。
    candidate_id = await _resolve_candidate_id(db, params.get("candidateId", ""))
    if not candidate_id:
        return f"未找到候选人 {params.get('candidateId', '')}，请先调用 list_resumes 确认候选人。"
    body = {**params, "candidateId": candidate_id, "round": normalize_interview_round(params.get("round") or "first")}
    round_label = _round_label(body["round"])

    # 状态前置校验：候选人未进入面试流程时，返回「需用户决策」的确认结果，由 graph
    # confirm_gate 挂起并给用户选项。RBAC 不越界：仅当前用户有 resume:decide 才提供
    # 「邀约一面」选项；无该权限则只告知需要哪个角色、引导找对应的人。
    cand = (await db.execute(select(Candidate).where(Candidate.id == candidate_id))).scalar_one_or_none()
    if not cand:
        return f"未找到候选人 {candidate_id}，请先确认候选人 ID。"
    if cand.status not in INTERVIEW_ELIGIBLE_STATUSES:
        cu = get_current_user_for_tools()
        can_invite = cu is not None and cu.has("resume:decide")
        if can_invite:
            return ToolResult(
                success=False,
                tool_name="generate_questions",
                summary=f"候选人「{cand.name}」为求职中，需先邀约一面",
                details=(
                    f"候选人「{cand.name}」(ID: {candidate_id}) 当前为 {cand.status}（求职中），"
                    f"未进入面试流程，不能直接生成{round_label}面试题。"
                    f"系统已请求用户确认是否先邀约一面再生成面试题。"
                ),
                display_hint=DisplayHint.CONFIRM,
                confirmation={
                    "kind": "confirm_interview_invite",
                    "candidate_id": candidate_id,
                    "candidate_name": cand.name,
                    "round": round_label,
                    "question": f"候选人「{cand.name}」当前状态为「求职中」，尚未进入面试流程。",
                    "options": [
                        {
                            "label": f"邀约一面并生成{round_label}题",
                            "description": f"将候选人状态从求职中推进为一面中，并生成{round_label}面试题供您预览",
                        },
                        {
                            "label": "取消",
                            "description": "不执行任何操作，保持候选人当前状态",
                        },
                    ],
                },
            )
        return (
            f"候选人「{cand.name}」当前为求职中（job_hunting），未进入面试流程，无法直接生成{round_label}面试题。"
            f"先邀约一面（将候选人状态改为一面中）需要 resume:decide 权限（由 HR 角色拥有），"
            f"请让 HR 先为这位候选人邀约一面后再出题；或告诉我换一位已在面试中的候选人。"
        )

    result = await fn(body=body, db=db)
    data = result.get("data", {}) or {}
    questions = data.get("questions", [])
    if not questions:
        return f"未能生成{round_label}面试题，请稍后重试或检查面试出题开关。"
    lines = [f"已生成 {len(questions)} 道{round_label}面试题："]
    for q in questions:
        lines.append(
            f"  [{q.get('index', '?')}] {q.get('content', '')} | "
            f"分类:{q.get('category', '')} | 难度:{q.get('difficulty', '')} | ID:{q.get('id', '')}"
        )
    detail = "\n".join(lines)
    # 结构化返回题目数据：details 仍是文本（供 LLM 总结），data 带完整题目列表
    # （前端渲染「查看面试题」按钮 + 弹窗，无需再发消息让 AI 重查）。
    return ToolResult(
        success=True,
        tool_name="generate_questions",
        summary=f"已生成 {len(questions)} 道{round_label}面试题",
        details=detail,
        data={
            "questions": questions,
            "candidateId": candidate_id,
            "candidateName": cand.name,
            "round": round_label,
            "count": len(questions),
        },
        display_hint=DisplayHint.QUESTIONS,
    )


async def _get_evaluation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import query_interview_evaluation, normalize_interview_round

    candidate_id = params["candidateId"]
    round_key = normalize_interview_round(params.get("round") or "first")
    round_label = _round_label(round_key)

    result = await query_interview_evaluation(
        db,
        candidate_id=candidate_id,
        round=round_key,
        transcript_id=params.get("transcriptId"),
    )
    if result.get("code") != 0:
        return f"❌ 面试评定查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", {}) or {}
    scores = data.get("questions", [])
    if not scores:
        if not data.get("transcriptId"):
            return (
                f"暂无{round_label}面试评定数据。"
                f"请先在「面试评定」页面上传{round_label}录音或转写文本，"
                f"系统会从对话中自动抽取题目并 AI 评分。"
                f"（注意：面试评定与「面试出题」的预生成题目是两套独立流程。）"
            )
        return f"已找到{round_label}转写记录，但尚未抽取到面试题，请等待后台解析完成或重新上传。"
    scored = [s for s in scores if s.get("primaryScore") is not None]
    lines = [f"{round_label}面试评分（{len(scored)}/{len(scores)} 题已评分）："]
    for s in scores:
        score_str = f"{s.get('primaryScore')}分" if s.get("primaryScore") is not None else "未评分"
        answer_preview = (s.get("answer") or "")[:100]
        lines.append(
            f"  [{s.get('index', '?')}] {s.get('content', '')[:80]}... | "
            f"得分:{score_str} | 分类:{s.get('category', '')}"
        )
        if answer_preview:
            lines.append(f"      回答: {answer_preview}...")
    report = data.get("assessmentReport")
    if report:
        lines.append("\n综合评定报告已生成，可在面试评定详情页查看。")
    return "\n".join(lines)


async def _ai_score_question(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import ai_score_question as fn
    result = await fn(question_id=params["questionId"], body={"answer": params.get("answer", "")}, db=db)
    # 路由在「AI 评分开关关闭 / 手动评分模式」返回 fail(403)、题目不存在返回
    # not_found——不检查 code 会把失败包装成"AI评分：0 分"，必须显式检查。
    if result.get("code") != 0:
        return f"❌ AI评分失败：{result.get('message', '未知错误')}"
    data = result.get("data", {})
    return f"AI评分：{data.get('score', 0)} 分"


async def _get_leaderboard(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import get_leaderboard as fn
    # category 容错：模型常按工具描述传「一面/二面」，路由只认 first_result/second_result，
    # 传错会 fail(400) → 模型看到失败反复重试同一调用 → 工具循环（实测 manager 连续调 16 次）。
    category = (params.get("category") or "").strip()
    if category in ("first", "一面", "first_result"):
        category = "first_result"
    elif category in ("second", "二面", "second_result"):
        category = "second_result"
    result = await fn(category=category, db=db)
    if result.get("code") != 0:
        return f"❌ 排行榜查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", []) or []
    if not data:
        return "排行榜暂无数据"
    try:
        limit = int(params.get("limit", 20) or 20)
    except (TypeError, ValueError):
        limit = 20
    lines = [f"排行榜（{params.get('category', '')}）共 {len(data)} 人，前 {min(limit, len(data))} 名："]
    for i, r in enumerate(data[:limit], 1):
        lines.append(
            f"  #{r.get('rank', i)} [{r.get('id', '')}] {r.get('name', '?')} | "
            f"分:{r.get('score') or r.get('totalScore', '—')} | 状态:{r.get('status', '—')}"
        )
    return "\n".join(lines)


async def _save_questions(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import save_questions as fn, normalize_interview_round
    result = await fn(
        body={
            "candidateId": params["candidateId"],
            "round": normalize_interview_round(params["round"]),
            "questions": params["questions"],
        },
        db=db,
    )
    return f"已保存 {len(params['questions'])} 道面试题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"


async def _replace_question(params: dict, db: AsyncSession) -> str | ToolResult:
    from app.api.talent.interview import replace_question as fn, normalize_interview_round
    from app.agent.tool_result import ToolResult, DisplayHint
    body = {**params, "round": normalize_interview_round(params.get("round") or "first")}
    result = await fn(question_id=params["questionId"], body=body, db=db)
    data = result.get("data", {}) or {}
    qs = data.get("questions", [])
    if result.get("code") != 0:
        return f"替换失败：{result.get('message', '')}"
    lines = [f"已替换题目，当前题单共 {len(qs)} 道："]
    for q in qs:
        lines.append(
            f"  [{q.get('index', '?')}] {q.get('content', '')} | "
            f"分类:{q.get('category', '')} | 难度:{q.get('difficulty', '')}"
        )
    return ToolResult(
        success=True,
        tool_name="replace_question",
        summary=f"已替换题目，当前题单共 {len(qs)} 道",
        details="\n".join(lines),
        data=data,
        display_hint=DisplayHint.QUESTIONS,
    )


async def _save_evaluation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import save_evaluation as fn, normalize_interview_round
    result = await fn(
        candidate_id=params["candidateId"],
        body={
            "round": normalize_interview_round(params["round"]),
            "scores": params["scores"],
        },
        db=db,
    )
    return f"已保存 {len(params['scores'])} 题评分" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"


async def _submit_evaluation(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import submit_evaluation as fn, normalize_interview_round

    # 路由参数名是 round_（alias="round"）且带 Query 默认值：直调必须显式传 round_，
    # 否则拿到的是 Query 对象，`round_ == "second"` 恒为 False，二面分支永远走不到。
    round_key = normalize_interview_round(params.get("round") or "first")
    round_label = _round_label(round_key)
    result = await fn(candidate_id=params["candidateId"], round_=round_key, db=db)
    if result.get("code") != 0:
        return f"提交失败：{result.get('message', '')}"
    d = result.get("data") or {}
    final_score = d.get("finalScore")
    passed = d.get("passed")
    if passed is True:
        outcome = "已推进到二面" if round_key == "first" else "已推进到待发 Offer"
    elif passed is False:
        outcome = "候选人已被标记为未通过（rejected）"
    else:
        outcome = "候选人状态未变更"
    return (
        f"已提交候选人 {params['candidateId']} 的{round_label}评定，"
        f"加权总分 {final_score}（合格线 {d.get('passThreshold', '—')}）。{outcome}。"
    )


async def _get_rankings(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import get_rankings as fn
    result = await fn(candidateId=params["candidateId"], db=db)
    if result.get("code") != 0:
        return f"❌ 排名查询失败：{result.get('message', '未知错误')}"
    data = result.get("data", [])
    if not data:
        return "暂无同岗位排名数据"
    lines = [f"同岗位排名共 {len(data)} 人："]
    for r in data[:20]:
        lines.append(
            f"  #{r.get('rank', '?')} {r.get('name', '?')} | "
            f"匹配度 {r.get('score', 0)} 分 | 状态: {r.get('status', '?')}"
        )
    return "\n".join(lines)


def register_handlers(registry) -> None:
    registry.register("get_questions", _get_questions)
    registry.register("generate_questions", _generate_questions)
    registry.register("get_evaluation", _get_evaluation)
    registry.register("ai_score_question", _ai_score_question)
    registry.register("get_leaderboard", _get_leaderboard)
    registry.register("save_questions", _save_questions)
    registry.register("replace_question", _replace_question)
    registry.register("save_evaluation", _save_evaluation)
    registry.register("submit_evaluation", _submit_evaluation)
    registry.register("get_rankings", _get_rankings)
