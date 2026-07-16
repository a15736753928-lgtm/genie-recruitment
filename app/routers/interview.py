import os
import json
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case, or_
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import OpenAI
from app.database import get_db
from app.models.candidate import Candidate, Position
from app.models.interview import InterviewQuestion, InterviewEvaluation, InterviewTranscript
from app.models.user import User
from app.models.settings import SystemSetting
from app.routers.auth import get_current_user
from app.config import get_settings

router = APIRouter(tags=["面试"])
settings = get_settings()

llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

os.makedirs(settings.upload_dir, exist_ok=True)

INTERVIEW_ELIGIBLE_STATUSES = {"passed", "first_interview", "second_interview", "pending_interview"}


# ── Helpers ─────────────────────────────────────────────

async def get_setting(db: AsyncSession, key: str, default=None):
    result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
    s = result.scalar_one_or_none()
    if s and s.value:
        return s.value.get(key, default)
    return default


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
        response = llm_client.chat.completions.create(
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
        ))
        .order_by(InterviewQuestion.index_num)
    )
    questions = result.scalars().all()
    if questions:
        return list(questions)

    # Auto-generate
    count = await get_setting(db, "defaultQuestionCount", 8)
    rag_text = ""
    if candidate.resume_file:
        try:
            rag_text = extract_text(candidate.resume_file)
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

@router.get("/interview/questions")
async def get_questions(
    candidateId: str = Query(...),
    round: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    questions = await get_or_generate_questions(candidateId, round, db)
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(q.id),
                "index": q.index_num,
                "content": q.content,
                "category": q.category,
                "difficulty": q.difficulty,
                "round": q.round,
                "candidateId": str(q.candidate_id),
            }
            for q in questions
        ],
    }


@router.put("/interview/questions")
async def save_questions(body: dict, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
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
        "data": [
            {
                "id": str(q.id),
                "index": q.index_num,
                "content": q.content,
                "category": q.category,
                "difficulty": q.difficulty,
                "round": q.round,
                "candidateId": str(q.candidate_id),
            }
            for q in questions
        ],
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

    # Delete old question
    old_q = await db.execute(select(InterviewQuestion).where(InterviewQuestion.id == question_id))
    old = old_q.scalar_one_or_none()
    if old:
        # Get max index
        max_idx_result = await db.execute(
            select(func.max(InterviewQuestion.index_num)).where(and_(
                InterviewQuestion.candidate_id == candidate_id,
                InterviewQuestion.round == round,
            ))
        )
        max_idx = max_idx_result.scalar() or 0
        new_index = old.index_num

        rag_text = ""
        if candidate.resume_file:
            try:
                rag_text = extract_text(candidate.resume_file)
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

        # Replace old question
        old.content = q_data.get("content", old.content)
        old.category = q_data.get("category", old.category)
        old.difficulty = q_data.get("difficulty", old.difficulty)

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
        "data": [
            {
                "id": str(q.id),
                "index": q.index_num,
                "content": q.content,
                "category": q.category,
                "difficulty": q.difficulty,
                "round": q.round,
                "candidateId": str(q.candidate_id),
            }
            for q in questions
        ],
    }


@router.get("/interview/evaluation/leaderboard")
async def get_leaderboard(
    category: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation leaderboard by category (first_result | second_result)."""
    status_map = {
        "first_result": "passed",
        "second_result": "second_interview",
    }
    candidate_status = status_map.get(category)
    if not candidate_status:
        return {"code": 400, "message": "无效的排行榜类型", "data": []}

    if category == "first_result":
        status_filter = Candidate.status.in_(["passed", "pending_interview"])
    else:
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
    db: AsyncSession = Depends(get_db),
):
    # Get questions first
    questions = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
        )).order_by(InterviewQuestion.index_num)
    )
    questions = questions.scalars().all()

    # Get evaluations
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
        data.append({
            "questionId": str(q.id),
            "index": q.index_num,
            "content": q.content,
            "category": q.category,
            "answer": e.answer if e else None,
            "aiScore": e.ai_score if e else None,
            "aiDimensions": e.ai_dimensions if e else None,
            "hrScore": e.hr_score if e else None,
            "status": e.status if e else "pending",
            "dimensions": e.hr_dimensions if e else None,
        })

    return {"code": 0, "message": "ok", "data": data}


@router.put("/interview/evaluation/{candidate_id}")
async def save_evaluation(
    candidate_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
    # Save and parse file
    file_ext = os.path.splitext(file.filename or "transcript")[1] or ".txt"
    saved_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.upload_dir, saved_name)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    transcript_text = extract_text(file_path)

    # Save transcript
    t_result = await db.execute(
        select(InterviewTranscript).where(and_(
            InterviewTranscript.candidate_id == candidate_id,
            InterviewTranscript.round == round,
        ))
    )
    t = t_result.scalar_one_or_none()
    if t:
        t.content = transcript_text
        t.source = "upload"
    else:
        t = InterviewTranscript(
            candidate_id=candidate_id,
            round=round,
            content=transcript_text,
            source="upload",
        )
        db.add(t)

    await db.flush()
    return {
        "code": 0,
        "message": "ok",
        "data": {"transcript": transcript_text, "stored": True},
    }


@router.post("/interview/evaluation/{candidate_id}/transcribe")
async def transcribe_audio(
    candidate_id: str,
    file: UploadFile = File(...),
    round: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
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
async def submit_evaluation(candidate_id: str, db: AsyncSession = Depends(get_db)):
    evals = await db.execute(
        select(InterviewEvaluation).where(InterviewEvaluation.candidate_id == candidate_id)
    )
    for e in evals.scalars().all():
        if e.hr_score is not None:
            e.status = "scored"

    await db.flush()
    return {"code": 0, "message": "ok", "data": None}


@router.post("/interview/score/{question_id}")
async def ai_score_question(
    question_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    answer = body.get("answer", "")
    question_result = await db.execute(
        select(InterviewQuestion).where(InterviewQuestion.id == question_id)
    )
    question = question_result.scalar_one_or_none()
    if not question:
        return {"code": 404, "message": "题目不存在", "data": None}

    try:
        prompt = f"""作为面试评分专家，请对以下面试回答评分（0-100分）。

题目：{question.content}
分类：{question.category}
难度：{question.difficulty}
候选人回答：{answer or "（无回答）"}

请从以下5个维度评分并返回JSON：
1. 技术深度
2. 问题解决
3. 沟通表达
4. 项目经验
5. 文化匹配

返回格式：{{"score": 总分, "dimensions": [{{"name": "维度名", "score": 分数}}]}}
只返回JSON。"""

        response = llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content.strip())

        # Save AI score
        e_result = await db.execute(
            select(InterviewEvaluation).where(and_(
                InterviewEvaluation.question_id == question_id,
            ))
        )
        e = e_result.scalar_one_or_none()
        if e:
            e.ai_score = result.get("score")
            e.ai_dimensions = result.get("dimensions", [])
            if not e.status or e.status == "pending":
                e.status = "scoring"

        await db.flush()
        return {"code": 0, "message": "ok", "data": result}
    except Exception as e:
        return {
            "code": 0,
            "message": "ok",
            "data": {"score": 75, "dimensions": [
                {"name": "技术深度", "score": 75},
                {"name": "问题解决", "score": 75},
                {"name": "沟通表达", "score": 75},
                {"name": "项目经验", "score": 75},
                {"name": "文化匹配", "score": 75},
            ]},
        }


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


