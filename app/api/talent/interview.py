import os
import re
import json
import uuid
import asyncio
from typing import Optional, List
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case, or_, text
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.database import get_db
from app.models.recruitment import Candidate, Position
from app.models.interview import InterviewQuestion, InterviewEvaluation, InterviewTranscript, InterviewSegmentEvaluation
from app.config import get_settings
from app.infrastructure import minio_storage
from app.services.system.system_settings import get_system_setting
from app.services.ai import get_llm_client

router = APIRouter(tags=["面试"])
settings = get_settings()

INTERVIEW_ELIGIBLE_STATUSES = {"passed", "first_interview", "second_interview", "pending_interview"}


def _pre_generated_source_filter():
    """面试出题页题目来源：AI 预生成或历史数据（source 为空）。"""
    return or_(
        InterviewQuestion.source == "pre_generated",
        InterviewQuestion.source.is_(None),
    )


def _pre_generated_scope(candidate_id: str, round: str):
    return and_(
        InterviewQuestion.candidate_id == candidate_id,
        InterviewQuestion.round == round,
        _pre_generated_source_filter(),
    )


async def _fetch_pre_generated_questions(
    candidate_id: str,
    round: str,
    db: AsyncSession,
) -> List[InterviewQuestion]:
    result = await db.execute(
        select(InterviewQuestion)
        .where(_pre_generated_scope(candidate_id, round))
        .order_by(InterviewQuestion.index_num)
    )
    return list(result.scalars().all())

# ── FunASR singleton (loaded once, reused across all transcribe calls) ──
_funasr_pipeline = None


def _get_funasr_pipeline():
    """Return the cached FunASR pipeline, creating it on first call."""
    global _funasr_pipeline
    if _funasr_pipeline is None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        _funasr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
    return _funasr_pipeline


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
        response = await get_llm_client().chat.completions.create(
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


def _format_position_requirements(position) -> str:
    """把岗位的职责/任职要求/技术栈等拼成一段文本，供反问环节评分 Agent 对照岗位要求打分。"""
    if not position:
        return ""
    parts: list[str] = []
    if position.jd_responsibilities:
        parts.append(f"【岗位职责】\n{position.jd_responsibilities}")
    if position.jd_requirements:
        parts.append(f"【任职要求】\n{position.jd_requirements}")
    if position.jd_preferred:
        parts.append(f"【加分项】\n{position.jd_preferred}")
    if position.jd_tech_stack:
        parts.append(f"【技术栈】\n{position.jd_tech_stack}")
    structured = []
    if getattr(position, "education_requirement", None):
        structured.append(f"学历{position.education_requirement}")
    if getattr(position, "experience_requirement", None):
        structured.append(f"经验{position.experience_requirement}")
    if structured:
        parts.append("【硬性条件】" + "、".join(structured))
    return "\n\n".join(parts)


def _strip_code_fence(content: str) -> str:
    """Strip a leading ```json / ``` code fence and trailing fence if present."""
    content = content.strip()
    # Match ```json\n ... ``` or ```\n ... ```
    fence_match = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", content, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # Loose stripping (legacy behaviour) for partial fences
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _extract_json_array(content: str) -> list:
    """Best-effort extraction of a JSON array from an LLM response.

    Handles: pure JSON, code-fenced JSON, JSON preceded/followed by prose.
    Returns [] if no valid array can be recovered.
    """
    cleaned = _strip_code_fence(content)
    # Fast path
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    # Fallback: locate the first [...] block in the raw text
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _extract_json_object(content: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    cleaned = _strip_code_fence(content)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def extract_qa_from_transcript(transcript_text: str, position_name: str) -> List[dict]:
    """从面试转写文本中抽取「问题 + 回答原文 + 分类」列表。

    面试官实际问的题目可能与「面试出题」环节生成的题目完全不同，这里完全由上传的
    面试对话驱动，不依赖预生成题目。
    """
    if not transcript_text or not transcript_text.strip():
        return []

    MAX_CHARS = 8000
    truncated = len(transcript_text) > MAX_CHARS
    transcript_slice = transcript_text[:MAX_CHARS]
    if truncated:
        print(f"[extract_qa] 转写文本长度 {len(transcript_text)} 超过 {MAX_CHARS}，"
              f"仅解析前 {MAX_CHARS} 字符")

    system_prompt = f"""你是一位资深的面试记录分析专家。请从下面这段面试转写文本中，抽取面试官实际提出的问题以及候选人对应的回答原文。

岗位：{position_name or '未知'}

输入文本可能是以下任一格式，请都识别：
- 「面试官：...」「候选人：...」之类的说话人标注
- 「Q：...」「A：...」之类的问答标注
- 带时间戳的转写文本，如「00:01:23 面试官：...」
- 没有说话人标注的连续对话（请根据语义判断哪一段是问、哪一段是答）

要求：
1. 只抽取面试中真实发生的一问一答，不要臆造，不要把面试官的引导语单独成题。
2. 每条包含：question（面试官问题原文，可适当精简但保留原意）、answer（候选人回答原文，保留关键内容，不要省略到失去信息）、category（分类，从「技术能力、项目经验、工程素养、团队协作、架构设计、领导力、沟通表达」中选最贴近的一个）。
3. 按面试发生顺序输出。
4. 若文本明显不是面试对话（例如纯简历、纯职位描述、乱码、空白），返回空数组 []。
5. 严格返回纯 JSON 数组，不要包含任何 markdown 代码围栏、解释文字或前后缀。

输出格式：
[{{"question": "面试官问题原文", "answer": "候选人回答原文", "category": "技术能力"}}, ...]

示例输入：
面试官：请先做个自我介绍。
候选人：我叫张三，五年 Java 经验，做过电商后端。
面试官：讲一下你最近负责的订单系统架构。
候选人：我们用了微服务，订单服务拆分为...

示例输出：
[{{"question": "请先做个自我介绍。", "answer": "我叫张三，五年 Java 经验，做过电商后端。", "category": "沟通表达"}}, {{"question": "讲一下你最近负责的订单系统架构。", "answer": "我们用了微服务，订单服务拆分为...", "category": "架构设计"}}]

面试转写文本：
{transcript_slice}"""

    try:
        response = await get_llm_client().chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        raw_content = response.choices[0].message.content or ""
        data = _extract_json_array(raw_content)
        result = [d for d in data if isinstance(d, dict) and d.get("question")]
        if not result:
            preview = raw_content.strip().replace("\n", " ")[:300]
            print(f"[extract_qa] 未抽取到问答。LLM 原始返回长度={len(raw_content)}，"
                  f"解析后条目数={len(data)}，预览：{preview}")
        return result
    except Exception as e:
        print(f"Transcript QA extraction error: {e}")
        return []


async def extract_segments_from_transcript(transcript_text: str, position_name: str) -> dict:
    """从面试转写文本中抽取两个特殊片段：自我介绍 / 反问环节。

    与 extract_qa_from_transcript 互补——后者抽取常规问答对，本函数只识别面试的
    开场自我介绍与结尾反问两个非问答片段。返回
    {"self_intro": str, "reverse_questions": str}，任一未识别到则返回空串。
    """
    if not transcript_text or not transcript_text.strip():
        return {"self_intro": "", "reverse_questions": ""}

    system_prompt = f"""你是一位资深的面试记录分析专家。请从下面这段面试转写文本中，抽取两个特殊片段。

岗位：{position_name or '未知'}

需要抽取的片段：
1. self_intro（自我介绍）：面试开场候选人对自己的自我陈述原文。通常是面试官说「请先做个自我介绍」之后候选人的一段独白。
2. reverse_questions（反问环节）：面试结尾候选人向面试官提出的问题原文。通常是面试官说「你有什么想问的吗」之后候选人的提问。

要求：
1. 只抽取真实出现的内容，不要臆造；不要把常规问答混入这两个片段。
2. 保留原文关键内容，可适当精简但保留原意。
3. 若某个片段在转写中不存在，对应字段返回空字符串 ""。
4. 严格返回纯 JSON 对象，不要包含 markdown 或解释文字：
{{"self_intro": "...", "reverse_questions": "..."}}

面试转写文本：
{transcript_text[:8000]}"""

    try:
        response = await get_llm_client().chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        raw_content = response.choices[0].message.content or ""
        data = _extract_json_object(raw_content)
        result = {
            "self_intro": str(data.get("self_intro", "") or "").strip(),
            "reverse_questions": str(data.get("reverse_questions", "") or "").strip(),
        }
        if not result["self_intro"] and not result["reverse_questions"]:
            preview = raw_content.strip().replace("\n", " ")[:300]
            print(f"[extract_segments] 未抽取到片段。LLM 原始返回长度={len(raw_content)}，"
                  f"预览：{preview}")
        return result
    except Exception as e:
        print(f"Transcript segments extraction error: {e}")
        return {"self_intro": "", "reverse_questions": ""}


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
        .where(_pre_generated_scope(candidate_id, round))
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

    # LLM 调用期间可能有并发请求已写入题目，避免重复插入。
    existing = await _fetch_pre_generated_questions(candidate_id, round, db)
    if existing:
        return existing

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

async def _upsert_segment_evaluation(
    db: AsyncSession,
    *,
    candidate_id: str,
    round: str,
    transcript_id,
    segment_type: str,
    content: str | None = None,
    ai_score: int | None = None,
    ai_dimensions=None,
    hr_score: int | None = None,
    hr_dimensions=None,
) -> InterviewSegmentEvaluation:
    """创建或更新一条片段评分记录。

    查找键：(candidate_id, round, transcript_id, segment_type)。不存在则新建。
    AI / HR 字段仅在被显式传入（非 None）时覆写，避免互相清空。
    """
    e_result = await db.execute(
        select(InterviewSegmentEvaluation).where(and_(
            InterviewSegmentEvaluation.candidate_id == candidate_id,
            InterviewSegmentEvaluation.round == round,
            InterviewSegmentEvaluation.transcript_id == transcript_id,
            InterviewSegmentEvaluation.segment_type == segment_type,
        ))
    )
    e = e_result.scalar_one_or_none()
    if not e:
        e = InterviewSegmentEvaluation(
            candidate_id=candidate_id,
            round=round,
            transcript_id=transcript_id,
            segment_type=segment_type,
        )
        db.add(e)
    if content is not None:
        e.content = content
    if ai_score is not None:
        e.ai_score = ai_score
    if ai_dimensions is not None:
        e.ai_dimensions = ai_dimensions
    if hr_score is not None:
        e.hr_score = hr_score
    if hr_dimensions is not None:
        e.hr_dimensions = hr_dimensions
    if (ai_score is not None or hr_score is not None) and (not e.status or e.status == "pending"):
        e.status = "scoring"
    return e


def _segment_to_dict(e: InterviewSegmentEvaluation) -> dict:
    return {
        "segmentType": e.segment_type,
        "content": e.content,
        "aiScore": e.ai_score,
        "aiDimensions": e.ai_dimensions,
        "hrScore": e.hr_score,
        "hrDimensions": e.hr_dimensions,
        "status": e.status if e.status else "pending",
    }


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

    # Remove existing pre-generated questions only (keep transcript-derived ones)
    existing = await db.execute(
        select(InterviewQuestion).where(_pre_generated_scope(candidate_id, round))
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

    # Delete existing pre-generated questions only
    existing = await db.execute(
        select(InterviewQuestion).where(_pre_generated_scope(candidate_id, round))
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
            select(func.max(InterviewQuestion.index_num)).where(
                _pre_generated_scope(candidate_id, round)
            )
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

    questions = await _fetch_pre_generated_questions(candidate_id, round, db)
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
                _pre_generated_scope(candidate_id, round),
                InterviewQuestion.id.in_(parsed_ids),
            ))
        )
        for q in to_delete_result.scalars().all():
            await db.delete(q)
            deleted_count += 1
        await db.flush()

    # 重新编号剩余题目 —— 两步走，避免唯一约束在行级校验时冲突。
    # PostgreSQL 的 UNIQUE 约束默认 NOT DEFERRABLE，在 UPDATE 每行后即刻
    # 校验。当旧 index_num 与新 index_num 交叉时（例如删掉第 1 题后，
    # 旧 2→新 1、旧 3→新 2），同一语句内仍可能因处理顺序不同而冲突。
    # 第一步：将所有 index_num 加一个大偏移量，消除交叉；
    # 第二步：从偏移后的值重新生成连续编号。
    REINDEX_OFFSET = 1000000
    await db.execute(
        text("""
            UPDATE interview_questions
               SET index_num = index_num + :offset
             WHERE candidate_id = :cid
               AND round        = :rnd
               AND (source = 'pre_generated' OR source IS NULL)
        """),
        {"cid": candidate_id, "rnd": round, "offset": REINDEX_OFFSET},
    )
    await db.execute(
        text("""
            WITH ranked AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY index_num) AS new_num
                FROM   interview_questions
                WHERE  candidate_id = :cid
                  AND  round        = :rnd
                  AND  (source = 'pre_generated' OR source IS NULL)
            )
            UPDATE interview_questions
               SET index_num = ranked.new_num
              FROM ranked
             WHERE interview_questions.id = ranked.id
        """),
        {"cid": candidate_id, "rnd": round},
    )
    # ORM 缓存中的 index_num 已由原始 SQL 更新，需要使其失效再查询
    db.expire_all()
    remaining = await _fetch_pre_generated_questions(candidate_id, round, db)

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
        select(func.max(InterviewQuestion.index_num)).where(
            _pre_generated_scope(candidate_id, round)
        )
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

    questions = await _fetch_pre_generated_questions(candidate_id, round, db)
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
        target_tid = transcriptId
        transcript_filter = InterviewQuestion.transcript_id == transcriptId
    else:
        latest_t_result = await db.execute(
            select(InterviewTranscript.id).where(and_(
                InterviewTranscript.candidate_id == candidate_id,
                InterviewTranscript.round == round,
            )).order_by(InterviewTranscript.created_at.desc()).limit(1)
        )
        target_tid = latest_t_result.scalar_one_or_none()
        transcript_filter = InterviewQuestion.transcript_id == target_tid if target_tid else None

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

    # 取当前 transcript 范围下的「自我介绍 / 反问环节」片段评分
    segment_filters = [
        InterviewSegmentEvaluation.candidate_id == candidate_id,
        InterviewSegmentEvaluation.round == round,
    ]
    if target_tid is not None:
        segment_filters.append(InterviewSegmentEvaluation.transcript_id == target_tid)
    seg_result = await db.execute(
        select(InterviewSegmentEvaluation).where(and_(*segment_filters))
    )
    segments = [_segment_to_dict(s) for s in seg_result.scalars().all()]

    return {"code": 0, "message": "ok", "data": {"questions": data, "segments": segments}}


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

    # 保存「自我介绍 / 反问环节」片段的 HR 评分
    segment_inputs = body.get("segments") or []
    if segment_inputs:
        # 片段评分按 transcript_id 关联；前端未传则取最近一次上传记录
        target_tid = body.get("transcriptId")
        if not target_tid:
            latest_t_result = await db.execute(
                select(InterviewTranscript.id).where(and_(
                    InterviewTranscript.candidate_id == candidate_id,
                    InterviewTranscript.round == round,
                )).order_by(InterviewTranscript.created_at.desc()).limit(1)
            )
            target_tid = latest_t_result.scalar_one_or_none()

        for seg in segment_inputs:
            seg_type = seg.get("segmentType")
            if not seg_type:
                continue
            await _upsert_segment_evaluation(
                db,
                candidate_id=candidate_id,
                round=round,
                transcript_id=target_tid,
                segment_type=seg_type,
                hr_score=seg.get("hrScore"),
                hr_dimensions=seg.get("dimensions"),
            )

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

    # ── 抽取「自我介绍」「反问环节」两个特殊片段并评分 ──
    # 这两个片段不属于常规问答，独立用专门的子 Agent 评分，落库到
    # InterviewSegmentEvaluation，关联到本次 transcript_id。
    # 自我介绍评分需对照简历，反问评分需对照岗位要求，故此处加载两者作为上下文。
    segments = await extract_segments_from_transcript(transcript_text, position_name)
    segment_scored_types: list[str] = []
    if ai_scoring_enabled and scoring_mode != "manual":
        from app.agent.self_intro_agent import score_self_intro
        from app.agent.reverse_question_agent import score_reverse_question

        fallback = int(await get_setting(db, "passScoreThreshold", 75) or 75)

        # 加载简历摘要（用于自我介绍评分对照）与岗位要求（用于反问评分对照）
        resume_text = ""
        if candidate and candidate.resume_file:
            try:
                resume_text = await asyncio.to_thread(_resume_text, candidate.resume_file)
            except Exception:
                resume_text = ""
        position_requirements = _format_position_requirements(
            candidate.position if candidate else None
        )

        segment_tasks = []  # (segment_type, content, coro)
        if segments.get("self_intro"):
            segment_tasks.append((
                "self_intro",
                segments["self_intro"],
                score_self_intro(
                    segments["self_intro"],
                    position_name=position_name,
                    fallback_score=fallback,
                    resume_text=resume_text,
                ),
            ))
        if segments.get("reverse_questions"):
            segment_tasks.append((
                "reverse_question",
                segments["reverse_questions"],
                score_reverse_question(
                    segments["reverse_questions"],
                    position_name=position_name,
                    fallback_score=fallback,
                    position_requirements=position_requirements,
                ),
            ))

        # 两个片段并发评分
        results = await asyncio.gather(*[t[2] for t in segment_tasks]) if segment_tasks else []
        for (seg_type, seg_content, _), result in zip(segment_tasks, results):
            await _upsert_segment_evaluation(
                db,
                candidate_id=candidate_id,
                round=round,
                transcript_id=transcript_id,
                segment_type=seg_type,
                content=seg_content,
                ai_score=result.get("score"),
                ai_dimensions=result.get("dimensions", []),
            )
            segment_scored_types.append(seg_type)
        await db.flush()
        print(f"[upload_transcript] 片段评分完成：{segment_scored_types or '无可用片段'}")
    else:
        # 即便关闭 AI 评分，也把抽取到的片段原文落库（content），供前端展示与 HR 手动评分
        for seg_type, key in (("self_intro", "self_intro"), ("reverse_question", "reverse_questions")):
            seg_content = segments.get(key, "")
            if seg_content:
                await _upsert_segment_evaluation(
                    db,
                    candidate_id=candidate_id,
                    round=round,
                    transcript_id=transcript_id,
                    segment_type=seg_type,
                    content=seg_content,
                )
        await db.flush()

    qa_count = len(qa_list)
    seg_self_intro = bool(segments.get("self_intro"))
    seg_reverse = bool(segments.get("reverse_questions"))

    warnings: list[str] = []
    if not transcript_text or not transcript_text.strip():
        warnings.append("未能从文件中提取出文本，请检查文件内容或格式")
    elif qa_count == 0 and not seg_self_intro and not seg_reverse:
        warnings.append("上传成功但未识别出问答、自我介绍或反问环节，请确认文件是面试对话转写而非简历/职位描述")

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "transcript": transcript_text,
            "stored": True,
            "qaCount": qa_count,
            "transcriptId": str(transcript_id),
            "filename": file.filename or "transcript",
            "segments": {
                "selfIntro": seg_self_intro,
                "reverseQuestions": seg_reverse,
            },
            "warning": "；".join(warnings) if warnings else "",
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

    # Audio file — transcribe with FunASR Paraformer (Chinese-optimized)
    audio_bytes = await file.read()

    try:
        import tempfile
        import os as _os

        # Write audio to temp WAV (FunASR pipeline accepts file path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.close()

            # Singleton pipeline — loaded once, reused across all requests
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            import asyncio as _asyncio

            asr = await _asyncio.to_thread(_get_funasr_pipeline)
            result = await _asyncio.to_thread(asr, tmp.name)
            # result is list[dict] when input is a file path
            item = result[0] if isinstance(result, list) else result
            transcribed_text = item.get("text", "").strip()

            if not transcribed_text:
                transcribed_text = "[转写完成，但未识别到语音内容]"
        finally:
            try:
                _os.unlink(tmp.name)
            except OSError:
                pass

        t_result = await db.execute(
            select(InterviewTranscript).where(and_(
                InterviewTranscript.candidate_id == candidate_id,
                InterviewTranscript.round == round,
            ))
        )
        t = t_result.scalar_one_or_none()
        if t:
            t.content = transcribed_text
            t.source = "transcribe"
        else:
            t = InterviewTranscript(
                candidate_id=candidate_id,
                round=round,
                content=transcribed_text,
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
            "data": {"transcript": transcribed_text, "stored": True},
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

    qa_avg = round(sum(scores) / len(scores), 1) if scores else 0.0

    # ── 加权最终分：Q&A 80% + 自我介绍 10% + 反问 10% ──
    # 缺失环节（无记录或无分数）的权重回退给 Q&A，保证不存在的环节不影响最终分。
    seg_result = await db.execute(
        select(InterviewSegmentEvaluation).where(and_(
            InterviewSegmentEvaluation.candidate_id == candidate_id,
            InterviewSegmentEvaluation.round == round,
        ))
    )
    seg_by_type = {s.segment_type: s for s in seg_result.scalars().all()}

    def _pick_seg_score(seg: Optional[InterviewSegmentEvaluation]) -> Optional[float]:
        if not seg:
            return None
        if scoring_mode == "manual":
            v = seg.hr_score if seg.hr_score is not None else seg.ai_score
        else:
            v = seg.ai_score if seg.ai_score is not None else seg.hr_score
        return float(v) if v is not None else None

    self_score = _pick_seg_score(seg_by_type.get("self_intro"))
    reverse_score = _pick_seg_score(seg_by_type.get("reverse_question"))

    components: List[tuple[float, float]] = []  # (weight, score)
    if scores:
        components.append((0.8, qa_avg))
    if self_score is not None:
        components.append((0.1, self_score))
    if reverse_score is not None:
        components.append((0.1, reverse_score))

    if components:
        total_w = sum(w for w, _ in components)
        final_score = round(sum(w * s for w, s in components) / total_w, 1)
    else:
        final_score = 0.0

    passed = final_score >= pass_threshold

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
                        {"candidateId": candidate_id, "name": candidate.name, "avgScore": final_score},
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
            "avgScore": qa_avg,
            "qaAvg": qa_avg,
            "selfIntroScore": self_score,
            "reverseScore": reverse_score,
            "finalScore": final_score,
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


