import builtins
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
from app.database import get_db, async_session_factory
from app.models.recruitment import Candidate, Position
from app.models.interview import InterviewQuestion, InterviewEvaluation, InterviewTranscript, InterviewSegmentEvaluation
from app.config import get_settings
from app.infrastructure import minio_storage
from app.services.system.system_settings import get_system_setting
from app.services.ai import get_llm_client
from app.services.ai.speech import transcribe_audio_bytes
from app.core.state_machine import transition, StateError
from app.core.security import get_current_user, PermissionError_, CurrentUser
import logging
from app.utils.responses import ok, fail, not_found, conflict
from app.utils.llm_json import extract_json_array, extract_json_object

logger = logging.getLogger("genie.interview")
router = APIRouter(tags=["面试"])
settings = get_settings()

# ── 鉴权：interview.py 此前全模块零鉴权（全仓已知遗留，2026-08-02 补齐）──
# 读端点放行任意「面试相关读权」（hr 管理 / 评分 / 简历查看）；
# 写端点要求 hr 或评分权之一（interviewer/manager 负责评分出题，hr 负责管理）。
_INTERVIEW_READ_PERMS = ("interview:manage", "interview:score", "resume:view")
_INTERVIEW_WRITE_PERMS = ("interview:manage", "interview:score")


async def _require_interview_read(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if any(current.has(p) for p in _INTERVIEW_READ_PERMS):
        return current
    raise PermissionError_(f"无权限: 需要 {' 或 '.join(_INTERVIEW_READ_PERMS)} 之一")


async def _require_interview_write(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if any(current.has(p) for p in _INTERVIEW_WRITE_PERMS):
        return current
    raise PermissionError_(f"无权限: 需要 {' 或 '.join(_INTERVIEW_WRITE_PERMS)} 之一")


INTERVIEW_ELIGIBLE_STATUSES = {"invited", "round1", "round2"}


def normalize_interview_round(round: str) -> str:
    """统一轮次标识：一面/first → first，二面/second → second。"""
    r = (round or "first").strip().lower()
    if r in ("first", "1", "r1", "一面", "第一轮", "first_interview"):
        return "first"
    if r in ("second", "2", "r2", "二面", "第二轮", "second_interview"):
        return "second"
    return (round or "first").strip()


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
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        questions = extract_json_array(content)
        if questions is None:
            raise ValueError("模型未返回可解析的题目数组")
        return questions
    except Exception as e:
        logger.warning("面试题生成失败，返回兜底题目: %s", e)
        # Return fallback questions
        return [
            {"category": "技术能力", "difficulty": "medium", "content": f"请介绍你在{position_name}领域的技术栈和项目经验。"},
            {"category": "项目经验", "difficulty": "easy", "content": "请分享一个你最有成就感的项目经历。"},
        ]


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        import fitz
        return "\n".join(page.get_text() for page in fitz.open(file_path))
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
    """从 LLM 回复里抽取 JSON 数组；单个对象也接受（包成单元素数组）。取不到返回 []。"""
    arr = extract_json_array(content)
    if arr is not None:
        return arr
    obj = extract_json_object(content)
    return [obj] if obj is not None else []


def _extract_json_object(content: str) -> dict:
    """从 LLM 回复里抽取 JSON 对象；取不到返回 {}。"""
    return extract_json_object(content) or {}


def _deduplicate_qa(items: list[dict], max_items: int = 15) -> list[dict]:
    """后处理去重：LLM 说去重但经常做不到，这里按问题文本相似度硬去重。

    策略：去掉常见问候/引导前缀，按字符集合 overlap 判定相似性（>70% 视为重复），
    重复时保留首条（先出现的）。硬限 max_items 条。
    """
    _GREETING_RE = re.compile(
        r"^(你好[，,]?\s*|欢迎.{0,10}(面试|参加|来到)[，,]?\s*"
        r"|我是负责.{0,20}的面试官[，,]?\s*|接下来.{0,10}|下面.{0,10})"
    )

    def _norm(q: str) -> set:
        q = _GREETING_RE.sub("", q)
        q = re.sub(r"[，。？！、；：\u201c\u201d\u2018\u2019\s]+", "", q)
        return set(q)

    deduped: list[dict] = []
    seen: list[set] = []
    for item in items:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        tokens = _norm(q)
        if not tokens:
            continue
        is_dup = any(
            len(tokens & prev) / max(len(tokens), len(prev)) > 0.70
            for prev in seen if prev
        )
        if not is_dup:
            deduped.append(item)
            seen.append(tokens)
        if len(deduped) >= max_items:
            break
    return deduped


async def extract_qa_from_transcript(transcript_text: str, position_name: str) -> List[dict]:
    """从面试转写文本中抽取「问题 + 回答原文 + 分类」列表。

    面试官实际问的题目可能与「面试出题」环节生成的题目完全不同，这里完全由上传的
    面试对话驱动，不依赖预生成题目。
    """
    if not transcript_text or not transcript_text.strip():
        return []

    MAX_CHARS = 15000
    truncated = len(transcript_text) > MAX_CHARS
    transcript_slice = transcript_text[:MAX_CHARS]
    if truncated:
        logger.info("[extract_qa] 转写文本长度 %d 超过 %d，仅解析前 %d 字符",
                    len(transcript_text), MAX_CHARS, MAX_CHARS)

    system_prompt = f"""你是一位资深的面试记录分析专家。请从下面这段面试转写文本中，抽取面试官实际提出的问题以及候选人对应的回答原文。

岗位：{position_name or '未知'}

输入文本可能是以下任一格式，请都识别：
- 「面试官：...」「候选人：...」之类的说话人标注
- 「Q：...」「A：...」之类的问答标注
- 带时间戳的转写文本，如「00:01:23 面试官：...」
- 没有说话人标注的连续对话（请根据语义判断哪一段是问、哪一段是答）

**严格限制：最多输出 15 条问答。** 一场45分钟面试，真实独立问题通常不超过15道。超过15道说明你没有去重。

要求：
1. **只抽取面试中真实发生的一问一答，不要臆造，不要把面试官的引导语单独成题。**
2. **question 必须是问题本身**，禁止包含：问候语（如"你好，欢迎参加面试"）、自我介绍（如"我是负责XX岗位的面试官"）、过渡语（如"接下来我们聊一下"）。这些是引导语，不是问题。例如：
   - ❌ "你好，桂云飞，欢迎参加今天的一面面试，我是负责Agent工程师岗位的面试官，先请你做一个简单的自我介绍吧。"
   - ✅ "请先做一个简单的自我介绍吧。"
3. **去重**（极其重要）：同一道问题如果在转写中出现多次（面试官口头重复、引导、过渡后重述），只保留最完整的一次。判断标准：去掉问候语/过渡语后，核心问题相同的只保留一条。
4. 自我介绍只出现一次——即使面试官在不同时间点多次要求「请先做自我介绍」，也只保留一条最完整的记录。
5. **排除反问环节（最关键，务必遵守）**：反问环节 = 面试结尾候选人向面试官提问、面试官解答的对话，
   通常由面试官说「你还有什么想问我的吗」「你有什么问题想问我吗」「要不要问我几个问题」等引出。
   这类对话的提问方是候选人、回答方是面试官，方向与普通问答相反，**一律不要抽取**（反问环节由专门的评分模块独立评估）。
   判断方法：若某句「问题」由候选人说出、面试官回答，即属反问环节，跳过。例如：
   - ❌ question="这个岗位目前主要的业务场景是什么？"（候选人问面试官，不是面试官问候选人）
   - ❌ question="团队目前的技术栈主要是什么？"（同上，方向反了）
6. **每条必须输出 `questioner` 字段**：标注本条的提问方，只能是 `"面试官"` 或 `"候选人"`。
   - 普通问答（面试官提问、候选人回答）填 `"面试官"`。
   - 反问环节（候选人提问、面试官回答）填 `"候选人"`——后端会据此丢弃该条目。
   - **不得省略该字段，也不得填其他值。**
7. 其余字段：question（问题原文，去掉问候语/引导语，只保留问题核心）、answer（回答原文，保留关键内容）、category（从「技术能力、项目经验、工程素养、团队协作、架构设计、领导力、沟通表达」中选一个）。
8. 按面试发生顺序输出。
9. 若文本明显不是面试对话，返回空数组 []。
10. 严格返回纯 JSON 数组，不要包含任何 markdown 代码围栏、解释文字或前后缀。

输出格式（最多15条）：
[{{"question": "问题原文", "answer": "回答原文", "category": "技术能力", "questioner": "面试官"}}, ...]

错误示例（禁止）：
- question="你好，欢迎参加面试，我是负责XX岗位的面试官，请先做个自我介绍吧。"  ← 包含问候语
- question="请先做个自我介绍。" 且出现两次  ← 未去重
- question="这个岗位目前主要的业务场景是什么？", questioner="候选人"  ← 反问环节（候选人问面试官），后端会丢弃，必须跳过
- question="团队目前的技术栈主要是什么？", questioner="候选人"  ← 同上，方向反了

正确示例：
- question="请先做一个简单的自我介绍吧。", answer="我叫张三...", category="沟通表达", questioner="面试官"
- question="讲一下你最近负责的订单系统架构。", answer="我们用了微服务...", category="架构设计", questioner="面试官"

面试转写文本：
{transcript_slice}"""

    # DeepSeek 偶发返回空串/非法 JSON，重试最多 3 次；重试时略升 temperature 打破确定性空返回。
    MAX_ATTEMPTS = 3
    last_reason = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await get_llm_client().chat.completions.create(
                model=settings.deepseek_model,
                extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.2 if attempt == 1 else 0.4,
                max_tokens=8192,
            )
            raw_content = response.choices[0].message.content or ""
            data = _extract_json_array(raw_content)
            # 反问环节（提问人为候选人、回答人是面试官）在此剔除，避免混入逐题 AI 评分。
            # prompt 已要求每条输出 questioner；字段缺失时默认视为面试官提问（放行，与旧行为一致）。
            result = [
                d for d in data
                if isinstance(d, dict) and d.get("question")
                and str(d.get("questioner") or "面试官").strip() != "候选人"
            ]
            result = _deduplicate_qa(result)
            if result:
                if attempt > 1:
                    logger.info("[extract_qa] 第 %d 次尝试成功，抽取到 %d 条问答", attempt, len(result))
                return result
            # 空结果：可能是「确实不是面试对话」返回 []，也可能是 LLM 空返回/坏 JSON。
            # 前者不该重试，后者该重试——用原始返回长度区分：有内容且解析出 [] 视为真空。
            last_reason = (f"LLM 原始返回长度={len(raw_content)}，解析后条目数={len(data)}，"
                           f"预览：{raw_content.strip().replace(chr(10), ' ')[:200]}")
            if raw_content.strip() and isinstance(data, list):
                # 有实质返回且是合法空数组 → 判定为「非面试对话」，不再重试
                logger.info("[extract_qa] 判定为非面试对话或无问答（第 %d 次）：%s", attempt, last_reason)
                return []
            logger.warning("[extract_qa] 空/坏返回，准备重试（第 %d/%d 次）：%s",
                           attempt, MAX_ATTEMPTS, last_reason)
        except Exception as e:
            last_reason = f"异常: {e}"
            logger.warning("[extract_qa] 调用异常，准备重试（第 %d/%d 次）：%s",
                           attempt, MAX_ATTEMPTS, e)
    logger.error("[extract_qa] %d 次尝试后仍未抽取到问答：%s", MAX_ATTEMPTS, last_reason)
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
{transcript_text[:15000]}"""

    try:
        response = await get_llm_client().chat.completions.create(
            model=settings.deepseek_model,
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
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


# ── Shared query logic (HTTP routes + agent tools) ───────

async def query_interview_questions(
    db: AsyncSession,
    *,
    candidate_id: str,
    round: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    """获取候选人某轮面试题（可筛选），供路由与 Agent 工具直接调用。"""
    round = normalize_interview_round(round)
    all_questions = await get_or_generate_questions(candidate_id, round, db)

    filtered = all_questions
    if category:
        filtered = [q for q in filtered if (q.category or "") == category]
    if difficulty:
        filtered = [q for q in filtered if (q.difficulty or "") == difficulty]

    return ok({
          "questions": [_question_to_dict(q) for q in filtered],
          "stats": _build_question_stats(all_questions),
          "filteredCount": len(filtered),
      })


async def query_interview_evaluation(
    db: AsyncSession,
    *,
    candidate_id: str,
    round: str = "first",
    transcript_id: Optional[str] = None,
) -> dict:
    """获取面试评定数据（转写抽取题 + 评分），供路由与 Agent 工具直接调用。"""
    round = normalize_interview_round(round)
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")

    if transcript_id:
        target_tid = transcript_id
        transcript_filter = InterviewQuestion.transcript_id == transcript_id
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

    assessment_report = None
    if target_tid is not None:
        t_result = await db.execute(
            select(InterviewTranscript).where(InterviewTranscript.id == target_tid)
        )
        transcript_row = t_result.scalar_one_or_none()
        if transcript_row and transcript_row.assessment_report:
            assessment_report = transcript_row.assessment_report

    return ok({
          "questions": data,
          "segments": segments,
          "assessmentReport": assessment_report,
          "transcriptId": str(target_tid) if target_tid else None,
      })


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


async def _build_assessment_context(
    db: AsyncSession,
    candidate_id: str,
    round: str,
    transcript_id,
) -> dict | None:
    """组装全方位评定所需的上下文。"""
    t_result = await db.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    transcript = t_result.scalar_one_or_none()
    if not transcript:
        return None

    cand_result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.skills),
            selectinload(Candidate.ai_analysis),
        )
        .where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        return None

    pos = candidate.position
    position_payload = {
        "name": pos.name if pos else "未知岗位",
        "requirements": " ".join(filter(None, [
            getattr(pos, "jd_requirements", None) if pos else None,
            getattr(pos, "jd_content", None) if pos else None,
        ])),
    }

    resume_analysis = {}
    if candidate.ai_analysis:
        a = candidate.ai_analysis
        resume_analysis = {
            "overallScore": a.overall_score,
            "summary": a.summary,
            "highlights": a.highlights or [],
            "risks": a.risks or [],
            "recommendation": a.recommendation,
        }

    q_result = await db.execute(
        select(InterviewQuestion).where(and_(
            InterviewQuestion.candidate_id == candidate_id,
            InterviewQuestion.round == round,
            InterviewQuestion.source == "transcript",
            InterviewQuestion.transcript_id == transcript_id,
        )).order_by(InterviewQuestion.index_num)
    )
    questions = q_result.scalars().all()

    eval_result = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.candidate_id == candidate_id,
            InterviewEvaluation.round == round,
        ))
    )
    evals = {str(e.question_id): e for e in eval_result.scalars().all()}

    qa_items = []
    ai_scores = []
    for q in questions:
        e = evals.get(str(q.id))
        ai_score = e.ai_score if e else None
        if ai_score is not None:
            ai_scores.append(ai_score)
        qa_items.append({
            "question": q.content,
            "answer": e.answer if e else None,
            "category": q.category,
            "aiScore": ai_score,
            "aiDimensions": e.ai_dimensions if e else None,
        })

    seg_result = await db.execute(
        select(InterviewSegmentEvaluation).where(and_(
            InterviewSegmentEvaluation.candidate_id == candidate_id,
            InterviewSegmentEvaluation.round == round,
            InterviewSegmentEvaluation.transcript_id == transcript_id,
        ))
    )
    segments = [_segment_to_dict(s) for s in seg_result.scalars().all()]

    return {
        "round": round,
        "transcriptExcerpt": transcript.content or "",
        "avgQaScore": builtins.round(sum(ai_scores) / len(ai_scores), 1) if ai_scores else None,
        "candidate": {
            "name": candidate.name,
            "education": candidate.education,
            "experience": candidate.experience,
            "skills": [s.skill for s in (candidate.skills or [])],
        },
        "position": position_payload,
        "resumeAnalysis": resume_analysis,
        "qaItems": qa_items,
        "segments": segments,
    }


async def _generate_and_save_assessment_report(
    db: AsyncSession,
    candidate_id: str,
    round: str,
    transcript_id,
) -> dict | None:
    """生成并持久化全方位评定报告。"""
    context = await _build_assessment_context(db, candidate_id, round, transcript_id)
    if not context:
        return None
    if not context.get("transcriptExcerpt", "").strip():
        return None

    from app.agent.assessment_agent import generate_assessment_report

    report = await generate_assessment_report(context)
    t_result = await db.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    transcript = t_result.scalar_one_or_none()
    if transcript:
        transcript.assessment_report = report
        await db.flush()
    return report


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
    current: CurrentUser = Depends(_require_interview_read),
):
    """获取候选人某轮的面试题,支持按分类/难度筛选,并返回全量统计。

    `data` 为筛选后的题目列表;`stats` 始终基于本轮全量题目(忽略筛选),
    供前端工具条展示「共 N 题 · 简单 a · 中等 b · 较难 c」。
    """
    return await query_interview_questions(
        db,
        candidate_id=candidateId,
        round=round,
        category=category,
        difficulty=difficulty,
    )


@router.put("/interview/questions")
async def save_questions(body: dict, db: AsyncSession = Depends(get_db),
                         current: CurrentUser = Depends(_require_interview_write)):
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
    return ok()


@router.post("/interview/questions/regenerate")
async def regenerate_questions(body: dict, db: AsyncSession = Depends(get_db),
                               current: CurrentUser = Depends(_require_interview_write)):
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
    return ok({
          "questions": [_question_to_dict(q) for q in questions],
          "stats": _build_question_stats(questions),
          "filteredCount": len(questions),
      })


@router.post("/interview/questions/{question_id}/replace")
async def replace_question(question_id: str, body: dict, db: AsyncSession = Depends(get_db),
                           current: CurrentUser = Depends(_require_interview_write)):
    candidate_id = body.get("candidateId")
    round = body.get("round")

    # Get candidate info
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

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
    return ok({
          "questions": [_question_to_dict(q) for q in questions],
          "stats": _build_question_stats(questions),
          "filteredCount": len(questions),
      })


@router.post("/interview/questions/batch-delete")
async def batch_delete_questions(body: dict, db: AsyncSession = Depends(get_db),
                                 current: CurrentUser = Depends(_require_interview_write)):
    """批量删除面试题,删除后对剩余题目按难度排序重新连续编号。

    body: { candidateId, round, questionIds: [uuid str, ...] }
    返回剩余题目列表与全量统计。
    """
    candidate_id = body.get("candidateId")
    round = body.get("round")
    raw_ids = body.get("questionIds") or []
    if not candidate_id or not round or not raw_ids:
        return fail(400, "candidateId / round / questionIds 不能为空")

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

    return ok({
          "deletedCount": deleted_count,
          "questions": [_question_to_dict(q) for q in remaining],
          "stats": _build_question_stats(remaining),
          "filteredCount": len(remaining),
      })


@router.post("/interview/questions/append")
async def append_question(body: dict, db: AsyncSession = Depends(get_db),
                          current: CurrentUser = Depends(_require_interview_write)):
    """AI 生成一道面试题并追加到本轮末尾。

    body: { candidateId, round, prompt?, category?, difficulty? }
    - prompt 可选,为空时 AI 根据候选人背景与岗位自动出题。
    - category / difficulty 可选,默认 "技术能力" / "medium"。
    返回本轮全量题目列表与统计。
    """
    candidate_id = body.get("candidateId")
    round = body.get("round")
    if not candidate_id or not round:
        return fail(400, "candidateId / round 不能为空")

    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        return not_found("候选人不存在")

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
    return ok({
          "questions": [_question_to_dict(q) for q in questions],
          "stats": _build_question_stats(questions),
          "filteredCount": len(questions),
      })


@router.get("/interview/evaluation/leaderboard")
async def get_leaderboard(
    category: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    """Get evaluation leaderboard by category (first_result | second_result)."""
    status_map = {
        "first_result": "round1",
        "second_result": "round2",
    }
    candidate_status = status_map.get(category)
    if not candidate_status:
        return fail(400, "无效的排行榜类型", [])

    status_filter = Candidate.status == candidate_status

    # Get candidates with the right status
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.evaluations),
            selectinload(Candidate.educations),
            selectinload(Candidate.work_experiences),
            selectinload(Candidate.project_experiences),
            selectinload(Candidate.skills),
            selectinload(Candidate.ai_analysis),
        )
        .where(status_filter)
    )
    candidates = result.scalars().all()

    from app.api.recruitment.resume_serializer import serialize_candidate

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

        profile = serialize_candidate(c)
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
            "resumeScore": int(c.score or 0),
            "gender": profile.get("gender"),
            "age": profile.get("age"),
            "education": profile.get("education"),
            "experience": profile.get("experience"),
            "educationHistory": profile.get("educationHistory") or [],
            "workHistory": profile.get("workHistory") or [],
        })

    status_priority = {"completed": 0, "in_progress": 1, "pending": 2}

    def _sort_key(item: dict) -> tuple:
        return (
            status_priority.get(item["evalStatus"], 3),
            -item["totalScore"],
            item["name"],
        )

    # Rank within each position (not global across all positions).
    from collections import defaultdict

    by_position: dict[str, list] = defaultdict(list)
    for item in leaderboard:
        by_position[item["position"] or ""].append(item)

    ranked_leaderboard: list[dict] = []
    for position in sorted(by_position.keys(), key=lambda p: p or ""):
        group = sorted(by_position[position], key=_sort_key)
        for i, item in enumerate(group, 1):
            item["rank"] = i
            ranked_leaderboard.append(item)

    return ok(ranked_leaderboard)


@router.get("/interview/evaluation/{candidate_id}")
async def get_evaluation(
    candidate_id: str,
    round: str = Query("first"),
    transcriptId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    return await query_interview_evaluation(
        db,
        candidate_id=candidate_id,
        round=round,
        transcript_id=transcriptId,
    )


@router.put("/interview/evaluation/{candidate_id}")
async def save_evaluation(
    candidate_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
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
    return ok()


async def _run_transcript_pipeline(
    db: AsyncSession,
    candidate_id: str,
    round: str,
    transcript_id,
    transcript_text: str,
    log_tag: str = "transcript_pipeline",
    on_progress=None,
) -> dict:
    """面试转写文本的完整评定流水线（文本上传 / 音频转写共用）。

    从转写文本抽取问答 → 落库为 source='transcript' 的题目并 AI 评分 →
    抽取「自我介绍 / 反问环节」片段并评分 → 生成全方位综合评定报告。

    on_progress: 可选异步回调 async (status:str, progress:int, stage:str)，
    每阶段回写进度供前端轮询；None 时为纯同步调用（不追踪进度）。

    返回 {"qaCount", "selfIntro", "reverseQuestions", "assessmentReport", "warning"}。
    """
    async def _progress(status: str, pct: int, stage: str):
        if on_progress is not None:
            try:
                await on_progress(status, pct, stage)
            except Exception:
                logger.exception("[%s] 进度回调失败(忽略) stage=%s", log_tag, stage)

    ai_scoring_enabled = await get_setting(db, "aiInterviewScoring", True)
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")

    # 读取岗位名用于抽取/评分上下文
    cand_result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()
    position_name = candidate.position.name if (candidate and candidate.position) else "未知岗位"

    logger.info(
        "[%s] 流水线启动 candidate=%s round=%s transcript_id=%s text_len=%d",
        log_tag, candidate_id, round, transcript_id, len(transcript_text or ""),
    )
    await _progress("extracting", 45, "抽取问答中")
    qa_list = await extract_qa_from_transcript(transcript_text, position_name)
    logger.info(
        "[%s] ai_scoring_enabled=%s scoring_mode=%s extracted_qa_count=%d",
        log_tag, ai_scoring_enabled, scoring_mode, len(qa_list),
    )

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
            logger.info("[%s] 抽取到 %d 道题，最多 5 道题并行评分", log_tag, len(batch_items))
            await _progress("scoring", 55, f"评分中（{len(batch_items)} 题）")
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
            logger.info("[%s] AI 评分已跳过(关闭或手动模式) 共 %d 题", log_tag, len(created_questions))

    # ── 抽取「自我介绍」「反问环节」两个特殊片段并评分 ──
    # 这两个片段不属于常规问答，独立用专门的子 Agent 评分，落库到
    # InterviewSegmentEvaluation，关联到本次 transcript_id。
    # 自我介绍评分需对照简历，反问评分需对照岗位要求，故此处加载两者作为上下文。
    await _progress("segment", 80, "评估自我介绍与反问环节")
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
        logger.info("[%s] 片段评分完成：%s", log_tag, segment_scored_types or "无可用片段")
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

    # 生成全方位 AI 评定报告（基于转写 + 分项评分 + 简历上下文）。
    # 报告是「AI 供参考」，无论 ai/manual 评分模式都在后台直接生成——manual 模式下仅缺少
    # 分项 AI 分作输入，报告仍基于转写全文+简历+岗位生成，避免前端再走旧同步端点补生成。
    assessment_report = None
    if ai_scoring_enabled:
        try:
            await _progress("report", 90, "生成综合评定报告")
            assessment_report = await _generate_and_save_assessment_report(
                db, candidate_id, round, transcript_id
            )
            logger.info("[%s] 综合评定报告已生成 overall=%s", log_tag,
                        assessment_report.get('overallScore') if assessment_report else '—')
        except Exception:
            logger.exception("[%s] 综合评定报告生成失败", log_tag)

    qa_count = len(qa_list)
    seg_self_intro = bool(segments.get("self_intro"))
    seg_reverse = bool(segments.get("reverse_questions"))

    warnings: list[str] = []
    if not transcript_text or not transcript_text.strip():
        warnings.append("未能从文件中提取出文本，请检查文件内容或格式")
    elif qa_count == 0 and not seg_self_intro and not seg_reverse:
        warnings.append("上传成功但未识别出问答、自我介绍或反问环节，请确认文件是面试对话转写而非简历/职位描述")

    return {
        "qaCount": qa_count,
        "selfIntro": seg_self_intro,
        "reverseQuestions": seg_reverse,
        "assessmentReport": assessment_report,
        "warning": "；".join(warnings) if warnings else "",
    }


async def _set_transcript_progress(
    session: AsyncSession,
    transcript_id,
    *,
    status: str | None = None,
    progress: int | None = None,
    stage: str | None = None,
    message: str | None = None,
    content: str | None = None,
    assessment_report=None,
) -> None:
    """回写一条转写记录的处理进度并提交（后台任务用独立会话调用）。"""
    t_result = await session.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    t = t_result.scalar_one_or_none()
    if not t:
        return
    if status is not None:
        t.process_status = status
    if progress is not None:
        t.process_progress = progress
    if stage is not None:
        t.process_stage = stage
    if message is not None:
        t.process_message = message
    if content is not None:
        t.content = content
    if assessment_report is not None:
        t.assessment_report = assessment_report
    await session.commit()


async def _process_transcript_in_background(
    candidate_id: str,
    round: str,
    transcript_id,
    *,
    audio_bytes: bytes | None = None,
    log_tag: str = "bg_pipeline",
) -> None:
    """后台任务：（音频先转写→）跑完整评定流水线，全程回写进度到转写记录。

    用独立 async 会话（请求会话在响应返回时已关闭）。任一环节抛错都落 failed 状态。
    """
    async with async_session_factory() as session:
        async def on_progress(status: str, pct: int, stage: str):
            await _set_transcript_progress(
                session, transcript_id, status=status, progress=pct, stage=stage
            )

        try:
            transcript_text = ""
            # ── 音频：先转写（占 5-40%），再回写 content ──
            if audio_bytes is not None:
                await _set_transcript_progress(
                    session, transcript_id,
                    status="transcribing", progress=8, stage="语音转写中",
                )
                transcript_text = await transcribe_audio_bytes(audio_bytes)
                logger.info("[%s] 云端转写完成 chars=%d", log_tag, len(transcript_text))
                await _set_transcript_progress(
                    session, transcript_id,
                    status="extracting", progress=40, stage="转写完成，开始分析",
                    content=transcript_text,
                )
            else:
                # 文本上传：content 已在请求阶段落库，从库里读回
                t_result = await session.execute(
                    select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
                )
                row = t_result.scalar_one_or_none()
                transcript_text = row.content if row else ""

            pipeline = await _run_transcript_pipeline(
                session, candidate_id, round, transcript_id, transcript_text,
                log_tag=log_tag, on_progress=on_progress,
            )

            # 完成：报告已在流水线内落库，这里补一个终态 + warning 提示
            final_msg = pipeline.get("warning") or ""
            await _set_transcript_progress(
                session, transcript_id,
                status="completed", progress=100, stage="已完成",
                message=final_msg,
            )
            logger.info("[%s] 后台流水线完成 transcript_id=%s qa=%s",
                        log_tag, transcript_id, pipeline.get("qaCount"))
        except Exception as exc:
            import traceback as _tb
            logger.error("[%s] 后台流水线异常 transcript_id=%s\n%s",
                         log_tag, transcript_id, _tb.format_exc())
            try:
                # flush 失败后 session 处于 PendingRollback，必须先回滚事务才能复用写失败状态
                await session.rollback()
                await _set_transcript_progress(
                    session, transcript_id,
                    status="failed", progress=100, stage="处理失败",
                    message=f"处理失败: {exc}",
                )
            except Exception:
                logger.exception("[%s] 写入失败状态也失败了", log_tag)


@router.post("/interview/evaluation/{candidate_id}/transcript")
async def upload_transcript(
    candidate_id: str,
    file: UploadFile = File(...),
    round: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
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
        return fail(500, f"转写文件存储失败: {e}")

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

    if not transcript_text or not transcript_text.strip():
        return fail(400, "未能从文件中提取出文本，请检查文件内容或格式")

    # Save transcript —— 每次上传创建一条新的历史记录（不再覆盖），初始为 pending
    t = InterviewTranscript(
        candidate_id=candidate_id,
        round=round,
        content=transcript_text,
        source="upload",
        filename=file.filename or "transcript",
        process_status="pending",
        process_progress=0,
        process_stage="排队中",
    )
    db.add(t)
    await db.flush()
    transcript_id = t.id
    await db.commit()  # 提交，让后台任务的独立会话能读到这条记录

    # 立即把完整评定流水线丢到后台跑（抽问答→评分→片段→报告），前端轮询进度。
    asyncio.create_task(_process_transcript_in_background(
        candidate_id, round, transcript_id, audio_bytes=None, log_tag="upload_transcript",
    ))

    return ok({
          "transcriptId": str(transcript_id),
          "filename": file.filename or "transcript",
          "status": "pending",
      })


@router.get("/interview/evaluation/{candidate_id}/transcripts")
async def list_transcripts(
    candidate_id: str,
    round: str = Query("first"),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
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

    return ok([
          {
              "id": str(t.id),
              "round": t.round,
              "filename": t.filename or "transcript",
              "source": t.source,
              "createdAt": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else "",
              "qaCount": counts.get(t.id, 0),
              # AI 分析是否可用：转写内容非空（后台 ASR 回写 content 后才有）
              "hasContent": bool(t.content and t.content.strip()),
          }
          for t in transcripts
      ])


@router.delete("/interview/evaluation/transcript/{transcript_id}")
async def delete_transcript(
    transcript_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
):
    """删除一条上传历史记录，级联删除其抽取的题目与评分。"""
    t_result = await db.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    t = t_result.scalar_one_or_none()
    if not t:
        return not_found("记录不存在")
    await db.delete(t)
    await db.flush()
    return ok()


@router.get("/interview/evaluation/transcript/{transcript_id}/status")
async def get_transcript_status(
    transcript_id: str,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    """轮询一条转写记录的后台处理进度。完成时一并返回综合评定报告。"""
    t_result = await db.execute(
        select(InterviewTranscript).where(InterviewTranscript.id == transcript_id)
    )
    t = t_result.scalar_one_or_none()
    if not t:
        return not_found("记录不存在")
    return ok({
          "transcriptId": str(t.id),
          "status": t.process_status or "completed",
          "progress": t.process_progress if t.process_progress is not None else 100,
          "stage": t.process_stage or "",
          "message": t.process_message or "",
          "assessmentReport": t.assessment_report if t.process_status == "completed" else None,
      })


@router.post("/interview/evaluation/{candidate_id}/transcribe")
async def transcribe_audio(
    candidate_id: str,
    file: UploadFile = File(...),
    round: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
):
    logger.info("[transcribe_audio] 收到音频上传 candidate=%s round=%s file=%s",
                candidate_id, round, file.filename)
    allow_audio = await get_setting(db, "allowAudioUpload", True)
    if not allow_audio:
        return fail(403, "系统已关闭音频上传功能")

    audio_bytes = await file.read()
    if not audio_bytes:
        return fail(400, "音频文件为空")

    # 仅接受音频：MIME 或以扩展名兜底（浏览器/客户端可能把音频标成 octet-stream，
    # 且 m4a 的 MIME 是 audio/mp4 / audio/x-m4a，统一按 audio/* 放行）。
    AUDIO_MIME_PREFIX = "audio/"
    AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".oga", ".wma", ".amr", ".webm")
    mime = (file.content_type or "").lower()
    fname_ext = (os.path.splitext(file.filename or "")[1] or "").lower()
    if not (mime.startswith(AUDIO_MIME_PREFIX) or fname_ext in AUDIO_EXTS):
        return fail(400, "仅支持上传音频文件（mp3/wav/m4a 等）")

    # 建一条 pending 转写记录（content 先空，转写完成后由后台任务回写），
    # 每次上传独立历史记录，可选择/删除，并关联本次抽取的题目与评定报告。
    t = InterviewTranscript(
        candidate_id=candidate_id,
        round=round,
        content="",
        source="transcribe",
        filename=file.filename or "audio",
        process_status="pending",
        process_progress=0,
        process_stage="排队中",
    )
    db.add(t)
    await db.flush()
    transcript_id = t.id

    # 标记已上传音频（供榜单「已上传音频」逻辑），本轮已有评分记录时置位
    evals = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.candidate_id == candidate_id,
            InterviewEvaluation.round == round,
        ))
    )
    for e in evals.scalars().all():
        e.audio_uploaded = True
    await db.commit()  # 提交，让后台任务独立会话能读到这条记录

    # 立即把「转写 + 完整评定流水线」丢到后台跑（长录音转写~90s + 多次 LLM），前端轮询进度。
    asyncio.create_task(_process_transcript_in_background(
        candidate_id, round, transcript_id, audio_bytes=audio_bytes, log_tag="transcribe_audio",
    ))

    return ok({
          "transcriptId": str(transcript_id),
          "filename": file.filename or "audio",
          "status": "pending",
      })


@router.post("/interview/evaluation/{candidate_id}/assessment-report")
async def generate_assessment_report_endpoint(
    candidate_id: str,
    round: str = Query("first"),
    transcriptId: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
):
    """手动触发或重新生成全方位面试评定报告。"""
    target_tid = transcriptId
    if not target_tid:
        latest = await db.execute(
            select(InterviewTranscript.id).where(and_(
                InterviewTranscript.candidate_id == candidate_id,
                InterviewTranscript.round == round,
            )).order_by(InterviewTranscript.created_at.desc()).limit(1)
        )
        target_tid = latest.scalar_one_or_none()
    if not target_tid:
        return not_found("未找到面试转写记录")

    try:
        report = await _generate_and_save_assessment_report(
            db, candidate_id, round, target_tid
        )
        if not report:
            return fail(400, "无法生成评定报告")
        return ok(report)
    except Exception as exc:
        return fail(500, f"生成失败: {exc}")


@router.post("/interview/evaluation/{candidate_id}/submit")
async def submit_evaluation(
    candidate_id: str,
    round_: str = Query("first", alias="round"),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
):
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")
    pass_threshold = int(await get_setting(db, "passScoreThreshold", 75) or 75)

    evals_result = await db.execute(
        select(InterviewEvaluation).where(and_(
            InterviewEvaluation.candidate_id == candidate_id,
            InterviewEvaluation.round == round_,
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
            InterviewSegmentEvaluation.round == round_,
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
        # 注意：本端点当前无鉴权(无 Depends(get_current_user))，属于全仓已知的
        # 零鉴权路由之一(另案处理)，故 transition() 无 actor 信息可传，
        # actor_name 用默认值"系统"。状态变更一律经状态机写入，不再直接赋值。
        try:
            if passed:
                if round_ == "second":
                    # 二面通过 → 待发 Offer，等候「录用审批」流程处理。
                    # 注意：员工记录只能由 offer.py 的录用审批通过后自动生成
                    # （见 CLAUDE.md「员工数据来源」）。这里过去会直接调用
                    # ensure_employee_for_candidate() 绕开审批创建员工，
                    # 与 offer.py 形成两条并行的入职通道，现已移除。
                    await transition(db, "candidate", candidate, "pending_offer",
                                      reason=f"二面评定通过，加权总分 {final_score}")
                    try:
                        from app.services.system.notification import notify_if
                        from app.services.system.webhook import dispatch_webhook
                        await notify_if(
                            db,
                            "notifyOfferPending",
                            "pending_offer",
                            f"二面通过，待发起录用审批：{candidate.name}",
                            {"candidateId": candidate_id},
                        )
                        await dispatch_webhook(
                            db,
                            "candidate.pending_offer",
                            {"candidateId": candidate_id, "name": candidate.name, "avgScore": final_score},
                        )
                    except Exception:
                        pass
                else:
                    await transition(db, "candidate", candidate, "round2",
                                      reason=f"一面评定通过，加权总分 {final_score}")
            else:
                await transition(db, "candidate", candidate, "rejected",
                                  reason=f"{'二面' if round_ == 'second' else '一面'}评定未通过，加权总分 {final_score}")
        except StateError as e:
            # 上面已把本轮所有评分记录标成 scored（autoflush 已刷进事务），
            # 而 get_db 只在抛异常时回滚，正常 return 会把它们静默提交，
            # 留下「评分显示已评定、候选人却还停在上一轮」的不一致。
            await db.rollback()
            return conflict(e.message)

    await db.flush()
    return ok({
          "avgScore": qa_avg,
          "qaAvg": qa_avg,
          "selfIntroScore": self_score,
          "reverseScore": reverse_score,
          "finalScore": final_score,
          "passThreshold": pass_threshold,
          "passed": passed,
          "scoringMode": scoring_mode,
      })


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
    body: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_write),
):
    # AI 评分开关 + 手动模式拒绝
    ai_scoring = await get_setting(db, "aiInterviewScoring", True)
    scoring_mode = await get_setting(db, "defaultScoringMode", "ai")
    if not ai_scoring or scoring_mode == "manual":
        return fail(403, "当前设置不允许 AI 评分")

    answer = (body or {}).get("answer", "")
    question_result = await db.execute(
        select(InterviewQuestion).where(InterviewQuestion.id == question_id)
    )
    question = question_result.scalar_one_or_none()
    if not question:
        return not_found("题目不存在")

    # 前端补评分时通常不带 answer——回答已在转写抽取阶段落库到
    # InterviewEvaluation.answer，这里回退读取，保证评分有回答上下文。
    if not answer:
        e_result = await db.execute(
            select(InterviewEvaluation).where(
                InterviewEvaluation.question_id == question.id
            )
        )
        existing_eval = e_result.scalar_one_or_none()
        if existing_eval and existing_eval.answer:
            answer = existing_eval.answer

    result = await _score_question_with_llm(question, answer, db)
    return ok(result)


@router.get("/interview/rankings")
async def get_rankings(
    candidateId: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current: CurrentUser = Depends(_require_interview_read),
):
    """Get same-position candidate rankings for sidebar."""
    # Find the candidate's position
    result = await db.execute(
        select(Candidate).options(selectinload(Candidate.position)).where(Candidate.id == candidateId)
    )
    candidate = result.scalar_one_or_none()
    if not candidate or not candidate.position_id:
        return ok([])

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
            "status": "done" if c.status in ("pending_offer", "hired", "rejected") else "pending",
            "isCurrent": str(c.id) == candidateId,
        })

    return ok(rankings)


