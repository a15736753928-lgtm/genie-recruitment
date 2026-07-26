"""Interview tool handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


def _round_label(round_key: str) -> str:
    return "一面" if round_key == "first" else "二面"


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
    if cand and cand.status in ("new", "parsed", "pending_screen"):
        return (
            f"暂无{round_label}面试题。候选人当前为「待筛选」，需先通过初筛（状态改为 invited）"
            f"后再生成{round_label}面试题；或使用 generate_questions 强制出题。"
        )
    return f"暂无{round_label}面试题，请使用 generate_questions 生成"


async def _generate_questions(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import regenerate_questions as fn, normalize_interview_round

    body = {**params, "round": normalize_interview_round(params.get("round") or "first")}
    result = await fn(body=body, db=db)
    data = result.get("data", {}) or {}
    questions = data.get("questions", [])
    round_label = _round_label(body["round"])
    if not questions:
        return f"题目生成失败：{result.get('message', '未知错误')}"
    lines = [f"已生成 {len(questions)} 道{round_label}面试题："]
    for q in questions:
        lines.append(
            f"  [{q.get('index', '?')}] {q.get('content', '')} | "
            f"分类:{q.get('category', '')} | 难度:{q.get('difficulty', '')} | ID:{q.get('id', '')}"
        )
    return "\n".join(lines)


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
    data = result.get("data", {})
    return f"AI评分：{data.get('score', 0)} 分"


async def _get_leaderboard(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import get_leaderboard as fn
    result = await fn(category=params["category"], db=db)
    data = result.get("data", []) or []
    if not data:
        return "排行榜暂无数据"
    limit = int(params.get("limit", 20) or 20)
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


async def _replace_question(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import replace_question as fn, normalize_interview_round
    body = {**params, "round": normalize_interview_round(params.get("round") or "first")}
    result = await fn(question_id=params["questionId"], body=body, db=db)
    data = result.get("data", {}) or {}
    qs = data.get("questions", [])
    return f"已替换题目，当前题单共 {len(qs)} 道" if result["code"] == 0 else f"替换失败：{result.get('message', '')}"


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
    from app.api.talent.interview import submit_evaluation as fn
    result = await fn(candidate_id=params["candidateId"], db=db)
    return f"已提交候选人 {params['candidateId']} 的面试评定" if result["code"] == 0 else f"提交失败：{result.get('message', '')}"


async def _get_rankings(params: dict, db: AsyncSession) -> str:
    from app.api.talent.interview import get_rankings as fn
    result = await fn(candidateId=params["candidateId"], db=db)
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
