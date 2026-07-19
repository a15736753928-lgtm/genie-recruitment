import os
import json
import uuid
import asyncio
from typing import Optional, List
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case, or_
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import AsyncOpenAI
from app.database import get_db
from app.models.recruitment import Candidate, Position
from app.models.interview import InterviewQuestion, InterviewEvaluation, InterviewTranscript
from app.config import get_settings
from app.infrastructure import minio_storage
from app.services.system.system_settings import get_system_setting

router = APIRouter(tags=["面试"])
settings = get_settings()

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
)

INTERVIEW_ELIGIBLE_STATUSES = {"passed", "first_interview", "second_interview", "pending_interview"}


# ── Helpers ─────────────────────────────────────────────

async def get_setting(db: AsyncSession, key: str, default=None):
    """兼容旧调用点，统一走 system_settings 服务。"""
    return await get_system_setting(db, key, default)


async def generate_questions_with_llm(
    candidate_name: str,
    position_name: str,
    round: str,
    count: int = 8,
    resume_text: str = "",
    rag_context: str = "",
    prompt_override: str = "",
    category_override: str = "",
    difficulty_override: str = "",
) -> List[dict]:
    """Generate interview questions using DeepSeek with optional RAG context."""
    round_label = "一面（侧重基础技能和项目经验）" if round == "first" else "二面（侧重架构设计、领导力和综合素质）"

    system_prompt = f"""你是一位资深的HR面试官和技术面试专家。请为候选人「{candidate_name}」生成{round_label}的面试题目。
岗位：{position_name}
题目数量：{count}

要求：
1. 每道题包含：category（分类）、difficulty（easy/medium/hard）、content（题目内容）
2. 难度分布：30% easy, 40% medium, 30% hard
3. 题目要有针对性，结合候选人背景和岗位要求
4. 一面侧重技术基础、编码能力、项目经验
5. 二面侧重系统设计、架构能力、领导力、团队协作
6. 分类可从以下选择：技术能力、项目经验、系统设计、沟通协作、学习能力、领导力"""

    if rag_context:
        system_prompt += f"\n\n参考知识库素材：\n{rag_context}"

    if prompt_override:
        system_prompt += f"\n\n额外要求：{prompt_override}"

    if resume_text:
        system_prompt += f"\n\n候选人简历摘要：\n{resume_text[:2000]}"

    system_prompt += f"\n\n请返回纯JSON数组，每道题格式如下：\n[{{\"category\": \"分类\", \"difficulty\": \"{'hard' if difficulty_override else 'easy'}\", \"content\": \"题目内容\"}}, ...]"

    if category_override and difficulty_override:
        system_prompt = f"""请为「{candidate_name}」生成1道面试题目（{round_label}）。
岗位：{position_name}
分类：{category_override}
难度：{difficulty_override}
{('额外要求：' + prompt_override) if prompt_override else ''}

请返回纯JSON对象：{{"category": "...", "difficulty": "...", "content": "..."}}"""

    try:
        response = await llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"Question generation error: {e}")
        # Return fallback questions
        return [
            {"category": "技术能力", "difficulty": "medium", "content": f"请介绍你在{position_name}领域的技术栈和项目经验。"},
            {"category": "项目经验", "difficulty": "easy", "content": "请分享一个你最有成就感的项目经历。"},
        ]


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        from PyPDF2 import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    elif ext in ('.docx', '.doc'):
        from docx import Document
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    elif ext in ('.txt', '.md'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return ""


def _resume_text(stored: str) -> str:
    """Extract text from a stored resume reference (MinIO key or legacy path)."""
    with minio_storage.resolved_local_path(stored or "") as local:
        if not local:
            return ""
        return extract_text(local)


async def extract_qa_from_transcript(transcript_text: str, position_name: str) -> List[dict]:
    """从面试转写文本中抽取「问题 + 回答原文 + 分类」列表。

    面试官实际问的题目可能与「面试出题」环节生成的题目完全不同，这里完全由上传的
    面试对话驱动，不依赖预生成题目。
    """
    if not transcript_text or not transcript_text.strip():
        return []
    system_prompt = f"""你是一位资深的面试记录分析专家。请从下面这段面试转写文本中，抽取面试官实际提出的问题以及候选人对应的回答原文。

岗位：{position_name or '未知'}

要求：
1. 只抽取面试中真实发生的一问一答，不要臆造。
2. 每条包含：question（面试官问题原文，可适当精简但保留原意）、answer（候选人回答原文，保留关键内容）、category（分类，从「技术能力、项目经验、工程素养、团队协作、架构设计、领导力、沟通表达」中选最贴近的一个）。
3. 按面试发生顺序输出。
4. 若文本无法识别出任何问答对，返回空数组 []。

请返回纯JSON数组，格式如下：
[{{"question": "...", "answer": "...", "category": "..."}}, ...]

面试转写文本：
{transcript_text[:8000]}"""

    try:
        response = await llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        if isinstance(data, dict):
            data = [data]
        return [d for d in data if isinstance(d, dict) and d.get("question")]
    except Exception as e:
        print(f"Transcript QA extraction error: {e}")
        return []


async def get_or_generate_questions(candidate_id: str, round: str, db: AsyncSession) -> List[InterviewQuestion]:
    """Get existing questions or auto-generate them."""
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate or candidate.status not in INTERVIEW_ELIGIBLE_STATUSES:
        return []

    result = await db.execute(
        select(InterviewQuestion)
        .where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
            or_(
                InterviewQuestion.source == "pre_generated",
                InterviewQuestion.source.is_(None),
            ),
        ))
        .order_by(InterviewQuestion.index_num)
    )
    questions = result.scalars().all()
    if questions:
        return list(questions)

    # AI 出题开关
    ai_gen = await get_setting(db, "aiQuestionGeneration", True)
    if not ai_gen:
        return []

    # Auto-generate
    count = int(await get_setting(db, "defaultQuestionCount", 8) or 8)
    rag_text = ""
    if candidate.resume_file:
        try:
            rag_text = await asyncio.to_thread(_resume_text, candidate.resume_file)
        except Exception:
            pass

    q_data = await generate_questions_with_llm(
        candidate.name,
        candidate.position.name if candidate.position else "未知岗位",
        round,
        count,
        rag_text,
    )

    if isinstance(q_data, dict):
        q_data = [q_data]

    questions = []
    for i, qd in enumerate(q_data):
        q = InterviewQuestion(
            candidate_id=candidate_id,
            round=round,
            index_num=i + 1,
            content=qd.get("content", ""),
            category=qd.get("category", "综合"),
            difficulty=qd.get("difficulty", "medium"),
        )
        db.add(q)
        questions.append(q)

    await db.flush()
    return questions


# ── Endpoints ───────────────────────────────────────────

def _question_to_dict(q: InterviewQuestion) -> dict:
    return {
        "id": str(q.id),
        "index": q.index_num,
        "content": q.content,
        "category": q.category,
        "difficulty": q.difficulty,
        "round": q.round,
        "candidateId": str(q.candidate_id),
    }


def _build_question_stats(questions: List[InterviewQuestion]) -> dict:
    """按难度/分类汇总题目分布,用于前端筛选工具条展示。"""
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
    by_category: dict[str, int] = {}
    for q in questions:
        d = (q.difficulty or "medium")
        if d in by_difficulty:
            by_difficulty[d] += 1
        else:
            by_difficulty[d] = 1
        cat = q.category or "未分类"
        by_category[cat] = by_category.get(cat, 0) + 1
    return {
        "total": len(questions),
        "byDifficulty": by_difficulty,
        "byCategory": by_category,
    }


@router.get("/interview/questions")
async def get_questions(
    candidateId: str = Query(...),
    round: str = Query(...),
    category: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """获取候选人某轮的面试题,支持按分类/难度筛选,并返回全量统计。

    `data` 为筛选后的题目列表;`stats` 始终基于本轮全量题目(忽略筛选),
    供前端工具条展示「共 N 题 · 简单 a · 中等 b · 较难 c」。
    """
    all_questions = await get_or_generate_questions(candidateId, round, db)

    filtered = all_questions
    if category:
        filtered = [q for q in filtered if (q.category or "") == category]
    if difficulty:
        filtered = [q for q in filtered if (q.difficulty or "") == difficulty]

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "questions": [_question_to_dict(q) for q in filtered],
            "stats": _build_question_stats(all_questions),
            "filteredCount": len(filtered),
        },
    }


@router.put("/interview/questions")
async def save_questions(body: dict, db: AsyncSession = Depends(get_db)):
    candidate_id = body.get("candidateId")
    round = body.get("round")
    questions = body.get("questions", [])

    # Remove existing
    existing = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        ))
    )
    for q in existing.scalars().all():
        await db.delete(q)

    # Add new
    for i, qd in enumerate(questions):
        q = InterviewQuestion(
            candidate_id=candidate_id,
            round=round,
            index_num=qd.get("index", i + 1),
            content=qd.get("content", ""),
            category=qd.get("category", ""),
            difficulty=qd.get("difficulty", "medium"),
        )
        db.add(q)

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.post("/interview/questions/regenerate")
async def regenerate_questions(body: dict, db: AsyncSession = Depends(get_db)):
    candidate_id = body.get("candidateId")
    round = body.get("round")

    # Delete existing
    existing = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        ))
    )
    for q in existing.scalars().all():
        await db.delete(q)

    await db.flush()

    # Regenerate
    questions = await get_or_generate_questions(candidate_id, round, db)
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "questions": [_question_to_dict(q) for q in questions],
            "stats": _build_question_stats(questions),
            "filteredCount": len(questions),
        },
    }


@router.post("/interview/questions/{question_id}/replace")
async def replace_question(question_id: str, body: dict, db: AsyncSession = Depends(get_db)):
    candidate_id = body.get("candidateId")
    round = body.get("round")

    # Get candidate info
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    # 前端「手动添加题目」时会用 `{candidateId}-{round}-custom-{ts}` 这种非 UUID 的本地 id，
    # 这类题目从未落库；这里做一次 UUID 校验，避免 asyncpg 把非法 UUID 直接抛成 500。
    old: Optional[InterviewQuestion] = None
    try:
        parsed_qid = uuid.UUID(str(question_id))
    except (ValueError, TypeError):
        parsed_qid = None
    if parsed_qid is not None:
        old_q = await db.execute(select(InterviewQuestion).where(InterviewQuestion.id == parsed_qid))
        old = old_q.scalar_one_or_none()

    rag_text = ""
    if candidate.resume_file:
        try:
            rag_text = await asyncio.to_thread(_resume_text, candidate.resume_file)
        except Exception:
            pass

    q_data = await generate_questions_with_llm(
        candidate.name,
        candidate.position.name if candidate.position else "未知",
        round,
        1,
        rag_text,
        prompt_override=body.get("prompt", ""),
        category_override=body.get("category", ""),
        difficulty_override=body.get("difficulty", "medium"),
    )

    if isinstance(q_data, list) and q_data:
        q_data = q_data[0]

    if old is not None:
        # 替换已存在的题目：保留原 index
        old.content = q_data.get("content", old.content)
        old.category = q_data.get("category", old.category)
        old.difficulty = q_data.get("difficulty", old.difficulty)
    else:
        # 题目不存在（前端本地自定义题或已被删除）：追加一道新题
        max_idx_result = await db.execute(
            select(func.max(InterviewQuestion.index_num)).where(and_(
                InterviewQuestion.candidate_id == candidate_id,
                InterviewQuestion.round == round,
            ))
        )
        max_idx = max_idx_result.scalar() or 0
        new_q = InterviewQuestion(
            candidate_id=candidate_id,
            round=round,
            index_num=max_idx + 1,
            content=q_data.get("content", ""),
            category=q_data.get("category", "综合"),
            difficulty=q_data.get("difficulty", "medium"),
        )
        db.add(new_q)

    await db.flush()

    # Return all questions for this round
    all_qs = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        )).order_by(InterviewQuestion.index_num)
    )
    questions = all_qs.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "questions": [_question_to_dict(q) for q in questions],
            "stats": _build_question_stats(questions),
            "filteredCount": len(questions),
        },
    }


@router.post("/interview/questions/batch-delete")
async def batch_delete_questions(body: dict, db: AsyncSession = Depends(get_db)):
    """批量删除面试题,删除后对剩余题目按难度排序重新连续编号。

    body: { candidateId, round, questionIds: [uuid str, ...] }
    返回剩余题目列表与全量统计。
    """
    candidate_id = body.get("candidateId")
    round = body.get("round")
    raw_ids = body.get("questionIds") or []
    if not candidate_id or not round or not raw_ids:
        return {"code": 400, "message": "candidateId / round / questionIds 不能为空", "data": None}

    # 解析 UUID,过滤掉前端本地自定义题(非 UUID 的 id,从未落库)
    parsed_ids: list[uuid.UUID] = []
    for qid in raw_ids:
        try:
            parsed_ids.append(uuid.UUID(str(qid)))
        except (ValueError, TypeError):
            continue

    deleted_count = 0
    if parsed_ids:
        to_delete_result = await db.execute(
            select(InterviewQuestion).where(and_(
                InterviewQuestion.candidate_id == candidate_id,
                InterviewQuestion.round == round,
                InterviewQuestion.id.in_(parsed_ids),
            ))
        )
        for q in to_delete_result.scalars().all():
            await db.delete(q)
            deleted_count += 1
        await db.flush()

    # 重新编号剩余题目(按 index_num 升序保持原顺序)
    remaining_result = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        )).order_by(InterviewQuestion.index_num)
    )
    remaining = remaining_result.scalars().all()
    for i, q in enumerate(remaining, 1):
        q.index_num = i
    await db.flush()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "deletedCount": deleted_count,
            "questions": [_question_to_dict(q) for q in remaining],
            "stats": _build_question_stats(remaining),
            "filteredCount": len(remaining),
        },
    }


@router.post("/interview/questions/append")
async def append_question(body: dict, db: AsyncSession = Depends(get_db)):
    """AI 生成一道面试题并追加到本轮末尾。

    body: { candidateId, round, prompt?, category?, difficulty? }
    - prompt 可选,为空时 AI 根据候选人背景与岗位自动出题。
    - category / difficulty 可选,默认 "技术能力" / "medium"。
    返回本轮全量题目列表与统计。
    """
    candidate_id = body.get("candidateId")
    round = body.get("round")
    if not candidate_id or not round:
        return {"code": 400, "message": "candidateId / round 不能为空", "data": None}

    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    prompt = (body.get("prompt") or "").strip()
    category = (body.get("category") or "技术能力").strip() or "技术能力"
    difficulty = (body.get("difficulty") or "medium").strip() or "medium"

    rag_text = ""
    if candidate.resume_file:
        try:
            rag_text = await asyncio.to_thread(_resume_text, candidate.resume_file)
        except Exception:
            pass

    q_data = await generate_questions_with_llm(
        candidate.name,
        candidate.position.name if candidate.position else "未知岗位",
        round,
        1,
        rag_text,
        prompt_override=prompt,
        category_override=category,
        difficulty_override=difficulty,
    )

    if isinstance(q_data, list) and q_data:
        q_data = q_data[0]
    if not isinstance(q_data, dict):
        q_data = {"category": category, "difficulty": difficulty,
                  "content": f"请结合你的项目经验,谈谈在{candidate.position.name if candidate.position else '该岗位'}上的关键实践。"}

    max_idx_result = await db.execute(
        select(func.max(InterviewQuestion.index_num)).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        ))
    )
    max_idx = max_idx_result.scalar() or 0
    new_q = InterviewQuestion(
        candidate_id=candidate_id,
        round=round,
        index_num=max_idx + 1,
        content=q_data.get("content", ""),
        category=q_data.get("category", category),
        difficulty=q_data.get("difficulty", difficulty),
    )
    db.add(new_q)
    await db.flush()

    all_qs = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        )).order_by(InterviewQuestion.index_num)
    )
    questions = all_qs.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "questions": [_question_to_dict(q) for q in questions],
            "stats": _build_question_stats(questions),
            "filteredCount": len(questions),
        },
    }


@router.get("/interview/evaluation/leaderboard")
async def get_leaderboard(
    category: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation leaderboard by category (first_result | second_result)."""
    status_map = {
        "first_result": "first_interview",
        "second_result": "second_interview",
    }
    candidate_status = status_map.get(category)
    if not candidate_status:
        return {"code": 400, "message": "无效的排行榜类型", "data": []}

    status_filter = Candidate.status == candidate_status

    # Get candidates with the right status
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.evaluations),
        )
        .where(status_filter)
    )
    candidates = result.scalars().all()

    leaderboard = []
    for c in candidates:
        round = "first" if category == "first_result" else "second"

        q_result = await db.execute(
            select(func.count()).select_from(InterviewQuestion).where(and_(
                InterviewQuestion.candidate_id == c.id,
                InterviewQuestion.round == round,
            ))
        )
        question_count = q_result.scalar() or 0

        evals = [ev for ev in (c.evaluations or []) if ev.round == round]
        scored_count = sum(1 for ev in evals if ev.hr_score is not None)
        total_score = sum(ev.hr_score or 0 for ev in evals)
        audio_uploaded = any(ev.audio_uploaded for ev in evals)

        if scored_count == 0 and not audio_uploaded:
            eval_status = "pending"
        elif scored_count >= question_count and question_count > 0:
            eval_status = "completed"
        else:
            eval_status = "in_progress"

        leaderboard.append({
            "rank": 0,
            "candidateId": str(c.id),
            "name": c.name,
            "position": c.position.name if c.position else "",
            "totalScore": total_score,
            "maxScore": question_count * 100,
            "scoredCount": scored_count,
            "questionCount": question_count,
            "evalStatus": eval_status,
            "interviewRound": round,
            "audioUploaded": audio_uploaded,
        })

    status_priority = {"completed": 0, "in_progress": 1, "pending": 2}
    leaderboard.sort(key=lambda x: (
        status_priority.get(x["evalStatus"], 3),
        -x["totalScore"],
        x["name"],
    ))

    for i, item in enumerate(leaderboard):
        item["rank"] = i + 1

    return {"code": 0, "message": "ok", "data": leaderboard}


@router.get("/interview/evaluation/{candidate_id}")
async def get_evaluation(
    candidate_id: str,
    round: str = Query("first"),
    transcriptId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")

    # 面试评定页完全由上传的面试对话驱动，只展示从转写文本抽取的题目
    # （source='transcript'），与「面试出题」环节生成的题目完全独立。
    # 若指定 transcriptId，只返回该次上传记录的题目；否则取最近一次上传记录。
    if transcriptId:
        transcript_filter = InterviewQuestion.transcript_id == transcriptId
    else:
        latest_t_result = await db.execute(
            select(InterviewTranscript.id).where(and_(
                InterviewTranscript.candidate_id == candidate_id,
                InterviewTranscript.round == round,
            )).order_by(InterviewTranscript.created_at.desc()).limit(1)
        )
        latest_tid = latest_t_result.scalar_one_or_none()
        transcript_filter = InterviewQuestion.transcript_id == latest_tid if latest_tid else None

    base_filters = [
        InterviewQuestion.candidate_id == candidate_id,
        InterviewQuestion.round == round,
        InterviewQuestion.source == "transcript",
    ]
    if transcript_filter is not None:
        base_filters.append(transcript_filter)

    questions = await db.execute(
        select(InterviewQuestion).where(and_(*base_filters)).order_by(InterviewQuestion.index_num)
    )
    questions = questions.scalars().all()

    # Get evaluations for these questions
    eval_result = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.candidate_id == candidate_id,
            InterviewEvaluation.round == round,
        ))
    )
    evals = {str(e.question_id): e for e in eval_result.scalars().all()}

    data = []
    for q in questions:
        e = evals.get(str(q.id))
        # 按评分模式决定默认展示分数：ai 优先 aiScore，manual 优先 hrScore
        primary_score = None
        if e:
            if scoring_mode == "manual":
                primary_score = e.hr_score if e.hr_score is not None else e.ai_score
            else:
                primary_score = e.ai_score if e.ai_score is not None else e.hr_score
        data.append({
            "questionId": str(q.id),
            "index": q.index_num,
            "content": q.content,
            "category": q.category,
            "answer": e.answer if e else None,
            "aiScore": e.ai_score if e else None,
            "aiDimensions": e.ai_dimensions if e else None,
            "hrScore": e.hr_score if e else None,
            "primaryScore": primary_score,
            "scoringMode": scoring_mode,
            "status": e.status if e else "pending",
            "dimensions": e.hr_dimensions if e else None,
        })

    return {"code": 0, "message": "ok", "data": data}


@router.put("/interview/evaluation/{candidate_id}")
async def save_evaluation(
    candidate_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    round = body.get("round", "first")
    scores = body.get("scores", [])

    for sc in scores:
        qid = sc.get("questionId")
        e_result = await db.execute(
            select(InterviewEvaluation).where(and_(
                InterviewEvaluation.candidate_id == candidate_id,
                InterviewEvaluation.round == round,
                InterviewEvaluation.question_id == qid,
            ))
        )
        e = e_result.scalar_one_or_none()
        if not e:
            e = InterviewEvaluation(
                candidate_id=candidate_id,
                round=round,
                question_id=qid,
            )
            db.add(e)

        e.hr_score = sc.get("hrScore")
        e.hr_dimensions = sc.get("dimensions")
        e.answer = sc.get("answer")
        if sc.get("hrScore") is not None:
            e.status = "scored"

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.post("/interview/evaluation/{candidate_id}/transcript")
async def upload_transcript(
    candidate_id: str,
    file: UploadFile = File(...),
    round: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    # Save to MinIO and parse file. We keep the bytes in memory to both upload
    # to MinIO and extract text via a temp file (no re-download needed).
    file_ext = os.path.splitext(file.filename or "transcript")[1] or ".txt"
    object_key = f"interview/{uuid.uuid4()}{file_ext}"
    content = await file.read()
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return {"code": 500, "message": f"转写文件存储失败: {e}", "data": None}

    # Extract text from the in-memory bytes via a temp file
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        transcript_text = extract_text(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Save transcript —— 每次上传创建一条新的历史记录（不再覆盖）
    t = InterviewTranscript(
        candidate_id=candidate_id,
        round=round,
        content=transcript_text,
        source="upload",
        filename=file.filename or "transcript",
    )
    db.add(t)
    await db.flush()
    transcript_id = t.id

    # 面试评定完全由上传的面试对话驱动：从转写文本中抽取实际问答，落库为
    # source='transcript' 的题目（关联到本次上传的 transcript_id），与「面试出题」
    # 环节生成的题目完全独立，再对每个回答触发 AI 评分。
    ai_scoring_enabled = await get_setting(db, "aiInterviewScoring", True)
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")

    # 读取岗位名用于抽取/评分上下文
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    position_name = candidate.position.name if (candidate and candidate.position) else "未知岗位"

    qa_list = await extract_qa_from_transcript(transcript_text, position_name)
    print(f"[upload_transcript] candidate={candidate_id} round={round} transcript_id={transcript_id} "
          f"ai_scoring_enabled={ai_scoring_enabled} scoring_mode={scoring_mode} "
          f"extracted_qa_count={len(qa_list)}")

    if qa_list:
        # 先落库所有抽取出的题目（关联到本次 transcript_id），拿到 question id
        created_questions: list[tuple[InterviewQuestion, str]] = []
        for i, qa in enumerate(qa_list):
            q = InterviewQuestion(
                candidate_id=candidate_id,
                round=round,
                index_num=i + 1,
                content=qa.get("question", ""),
                category=qa.get("category", "综合"),
                difficulty="medium",
                source="transcript",
                transcript_id=transcript_id,
            )
            db.add(q)
            await db.flush()
            created_questions.append((q, qa.get("answer", "")))

        if ai_scoring_enabled and scoring_mode != "manual":
            # 并发调用评分 Agent（复制多份并行），一次处理一个问题，多个问题并发跑
            from app.agent.scoring_agent import score_answers_batch

            fallback = int(await get_setting(db, "passScoreThreshold", 75) or 75)
            batch_items = [
                {
                    "question_content": q.content or "",
                    "answer": ans,
                    "category": q.category or "",
                    "difficulty": q.difficulty or "medium",
                }
                for q, ans in created_questions
            ]
            print(f"[upload_transcript] 抽取到 {len(batch_items)} 道题，"
                  f"每题评 3 个维度(表达能力/逻辑思维/技术深度)，"
                  f"最多 5 道题并行评分")
            results = await score_answers_batch(
                batch_items,
                position_name=position_name,
                fallback_score=fallback,
                concurrency=5,
            )

            # 把并发评分结果落库到对应的 InterviewEvaluation
            for (q, ans), result in zip(created_questions, results):
                e_result = await db.execute(
                    select(InterviewEvaluation).where(
                        InterviewEvaluation.question_id == q.id
                    )
                )
                e = e_result.scalar_one_or_none()
                if not e:
                    e = InterviewEvaluation(
                        candidate_id=q.candidate_id,
                        round=q.round,
                        question_id=q.id,
                    )
                    db.add(e)
                e.ai_score = result.get("score")
                e.ai_dimensions = result.get("dimensions", [])
                if not e.answer and ans:
                    e.answer = ans
                if not e.status or e.status == "pending":
                    e.status = "scoring"
            await db.flush()
        else:
            print(f"[upload_transcript] AI scoring skipped (disabled or manual mode) "
                  f"for {len(created_questions)} questions")

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "transcript": transcript_text,
            "stored": True,
            "qaCount": len(qa_list),
            "transcriptId": str(transcript_id),
            "filename": file.filename or "transcript",
        },
    }


@router.get("/interview/evaluation/{candidate_id}/transcripts")
async def list_transcripts(
    candidate_id: str,
    round: str = Query("first"),
    db: AsyncSession = Depends(get_db),
):
    """列出某候选人某轮的所有上传历史记录，按上传时间倒序。

    每条记录含 id、文件名、上传时间、抽取出的题目数。
    """
    result = await db.execute(
        select(InterviewTranscript).where(and_(
            InterviewTranscript.candidate_id == candidate_id,
            InterviewTranscript.round == round,
        )).order_by(InterviewTranscript.created_at.desc())
    )
    transcripts = result.scalars().all()

    # 批量统计每条记录下的题目数
    tids = [t.id for t in transcripts]
    counts: dict = {}
    if tids:
        count_result = await db.execute(
            select(InterviewQuestion.transcript_id, func.count(InterviewQuestion.id))
            .where(InterviewQuestion.transcript_id.in_(tids))
            .group_by(InterviewQuestion.transcript_id)
        )
        counts = {row[0]: row[1] for row in count_result.all()}

    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(t.id),
                "filename": t.filename or "transcript",
                "source": t.source,
                "createdAt": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else "",
                "qaCount": counts.get(t.id, 0),
            }
            for t in transcripts
        ],
    }


@router.delete("/interview/evaluation/transcript/{transcript_id}")
async def delete_transcript(
    transcript_id: str,
    db: AsyncSession = Depends(get_db),
):
    """删除一条上传历史记录，级联删除其抽取的题目与评分。"""
    t_result = await db.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    t = t_result.scalar_one_or_none()
    if not t:
        return {"code": 404, "message": "记录不存在", "data": None}
    await db.delete(t)
    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.post("/interview/evaluation/{candidate_id}/transcribe")
async def transcribe_audio(
    candidate_id: str,
    file: UploadFile = File(...),
    round: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    allow_audio = await get_setting(db, "allowAudioUpload", True)
    if not allow_audio:
        return {"code": 403, "message": "系统已关闭音频上传功能", "data": None}

    # Audio file — transcribe with DeepSeek (audio not stored, text only)
    audio_bytes = await file.read()

    # Use DeepSeek for transcription
    try:
        # Note: DeepSeek doesn't have native audio transcription.
        # We store a note and use text-based fallback.
        # For production, integrate a speech-to-text service.
        simulated_text = f"[音频转写] 文件: {file.filename}, 大小: {len(audio_bytes)} bytes. 请集成语音识别服务以获得实际转写内容。"

        t_result = await db.execute(
            select(InterviewTranscript).where(and_(
                InterviewTranscript.candidate_id == candidate_id,
                InterviewTranscript.round == round,
            ))
        )
        t = t_result.scalar_one_or_none()
        if t:
            t.content = simulated_text
            t.source = "transcribe"
        else:
            t = InterviewTranscript(
                candidate_id=candidate_id,
                round=round,
                content=simulated_text,
                source="transcribe",
            )
            db.add(t)

        # Mark as audio uploaded on evaluations
        evals = await db.execute(
            select(InterviewEvaluation).where(and_(
                InterviewEvaluation.candidate_id == candidate_id,
                InterviewEvaluation.round == round,
            ))
        )
        for e in evals.scalars().all():
            e.audio_uploaded = True

        await db.flush()
        return {
            "code": 0,
            "message": "ok",
            "data": {"transcript": simulated_text, "stored": True},
        }
    except Exception as e:
        return {"code": 500, "message": f"转写失败: {str(e)}", "data": None}


@router.post("/interview/evaluation/{candidate_id}/submit")
async def submit_evaluation(
    candidate_id: str,
    round: str = Query("first"),
    db: AsyncSession = Depends(get_db),
):
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")
    pass_threshold = int(await get_setting(db, "passScoreThreshold", 75) or 75)

    evals_result = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.candidate_id == candidate_id,
            InterviewEvaluation.round == round,
        ))
    )
    evals = list(evals_result.scalars().all())
    scores: List[float] = []
    for e in evals:
        if e.hr_score is not None:
            e.status = "scored"
        if scoring_mode == "manual":
            score_val = e.hr_score if e.hr_score is not None else e.ai_score
        else:
            score_val = e.ai_score if e.ai_score is not None else e.hr_score
        if score_val is not None:
            scores.append(float(score_val))

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    passed = avg_score >= pass_threshold

    cand_result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = cand_result.scalar_one_or_none()
    if candidate:
        if passed:
            if round == "second":
                candidate.status = "offer_pending"
                try:
                    from app.services.system.notification import notify_if
                    from app.services.system.webhook import dispatch_webhook
                    await notify_if(
                        db,
                        "notifyOfferPending",
                        "offer_pending",
                        f"二面通过，待发 Offer：{candidate.name}",
                        {"candidateId": candidate_id},
                    )
                    await dispatch_webhook(
                        db,
                        "candidate.offer_pending",
                        {"candidateId": candidate_id, "name": candidate.name, "avgScore": avg_score},
                    )
                except Exception:
                    pass
            else:
                candidate.status = "second_interview"
        else:
            candidate.status = "rejected"

    await db.flush()
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "avgScore": avg_score,
            "passThreshold": pass_threshold,
            "passed": passed,
            "scoringMode": scoring_mode,
        },
    }


async def _score_question_with_llm(
    question: InterviewQuestion,
    answer: str,
    db: AsyncSession,
) -> dict:
    """调用「面试评分 Agent」对单道题目评分，并把 AI 分数落库到 InterviewEvaluation
    （无记录则创建）。

    评分逻辑由专门的子 Agent（app.agent.scoring_agent）承担，独立于面试出题环节，
    只基于「问题 + 回答 + 岗位上下文」做多维度评分。返回 {"score", "dimensions", "summary"}。
    LLM 失败时 Agent 返回兜底分数，这里同样落库，保证前端始终能看到 AI 参考评分。
    """
    from app.agent.scoring_agent import score_answer

    fallback_score = int(await get_setting(db, "passScoreThreshold", 75) or 75)

    # 读取岗位名作为评分上下文
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(
            Candidate.id == question.candidate_id
        )
    )
    candidate = cand_result.scalar_one_or_none()
    position_name = candidate.position.name if (candidate and candidate.position) else "未知岗位"

    result = await score_answer(
        question_content=question.content or "",
        answer=answer or "",
        position_name=position_name,
        category=question.category or "",
        difficulty=question.difficulty or "medium",
        fallback_score=fallback_score,
    )

    # 无论 LLM 成功还是兜底，都把 AI 分数落库，确保前端能看到 AI 参考评分
    e_result = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.question_id == question.id,
        ))
    )
    e = e_result.scalar_one_or_none()
    if not e:
        e = InterviewEvaluation(
            candidate_id=question.candidate_id,
            round=question.round,
            question_id=question.id,
        )
        db.add(e)
    e.ai_score = result.get("score")
    e.ai_dimensions = result.get("dimensions", [])
    if not e.answer and answer:
        e.answer = answer
    if not e.status or e.status == "pending":
        e.status = "scoring"
    await db.flush()
    return result


@router.post("/interview/score/{question_id}")
async def ai_score_question(
    question_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    # AI 评分开关 + 手动模式拒绝
    ai_scoring = await get_setting(db, "aiInterviewScoring", True)
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")
    if not ai_scoring or scoring_mode == "manual":
        return {"code": 403, "message": "当前设置不允许 AI 评分", "data": None}

    answer = body.get("answer", "")
    question_result = await db.execute(
        select(InterviewQuestion).where(InterviewQuestion.id == question_id)
    )
    question = question_result.scalar_one_or_none()
    if not question:
        return {"code": 404, "message": "题目不存在", "data": None}

    result = await _score_question_with_llm(question, answer, db)
    return {"code": 0, "message": "ok", "data": result}


@router.get("/interview/rankings")
async def get_rankings(
    candidateId: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get same-position candidate rankings for sidebar."""
    # Find the candidate's position
    result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidateId)
    )
    candidate = result.scalar_one_or_none()
    if not candidate or not candidate.position_id:
        return {"code": 0, "message": "ok", "data": []}

    # Get all candidates in the same position
    all_result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.position))
        .where(Candidate.position_id == candidate.position_id)
        .order_by(Candidate.score.desc())
    )
    all_candidates = all_result.scalars().all()

    rankings = []
    for rank, c in enumerate(all_candidates, 1):
        rankings.append({
            "rank": rank,
            "candidateId": str(c.id),
            "name": c.name,
            "score": c.score or 0,
            "status": "done" if c.status in ("second_interview", "passed", "probation", "onboarded") else "pending",
            "isCurrent": str(c.id) == candidateId,
        })

    return {"code": 0, "message": "ok", "data": rankings}


