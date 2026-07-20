import os
import uuid
import json
import logging
import re
import asyncio
from datetime import date
from typing import Optional, List, Tuple
from fastapi import APIRouter, Depends, File, Form, UploadFile, Query
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func, desc, asc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.database import get_db, async_session_factory
from app.models.recruitment import (
    Candidate, Position, CandidateSkill, CandidateEducation,
    CandidateWorkExperience, CandidateProjectExperience, CandidateAIAnalysis
)
from app.config import get_settings
from app.infrastructure import minio_storage
from app.services.recruitment.portrait_gender import infer_gender_from_resume_file
from app.services.recruitment.resume_dedup import find_duplicate_candidate
from app.services.recruitment.resume_field_verify import resolve_verified_education
from app.services.recruitment.resume_validator import validate_is_resume
from app.services.recruitment.education_tier_tag import (
    resolve_school_tier_tag,
    resolve_school_tier_tag_sync,
    inject_school_tier_into_keywords,
    SCHOOL_TIER_TAGS,
)
from app.services.system.system_settings import get_system_setting
from app.services.ai import get_llm_client

router = APIRouter(tags=["简历"])
settings = get_settings()
logger = logging.getLogger(__name__)

UNKNOWN = "未知"
IN_SCHOOL = "在校中"
DEFAULT_ETHNICITY = "汉族"
NO_WORK_EXPERIENCE_VALUES = {
    UNKNOWN, "应届", "无", "暂无", "无工作经验", "0", "0年",
}

os.makedirs(settings.upload_dir, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


def resolve_resume_file_path(stored_path: str) -> Optional[str]:
    """Resolve a stored resume reference to a local file path.

    ``stored_path`` is now a MinIO object key (e.g. ``resumes/<uuid>.pdf``).
    For backward compatibility, legacy absolute on-disk paths are still honored.
    The returned temp file (when downloaded from MinIO) is the caller's
    responsibility to remove. For a cleanup-free alternative use
    ``minio_storage.resolved_local_path``.
    """
    if not stored_path:
        return None
    # Legacy on-disk path
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    # MinIO object key — download to a temp file
    try:
        ext = os.path.splitext(stored_path)[1] or ""
        return minio_storage.download_to_temp(stored_path, suffix=ext)
    except Exception as e:
        logger.warning("从 MinIO 下载简历失败: %s (key=%s)", e, stored_path)
        return None


def extract_text_from_file(file_path: str) -> Tuple[str, Optional[str]]:
    """Extract text from PDF, DOCX, or plain text files. Returns (text, error).

    ``file_path`` may be a MinIO object key or a legacy local path. The file is
    materialized locally (temp file) only for the duration of extraction.
    """
    with minio_storage.resolved_local_path(file_path) as resolved:
        if not resolved:
            return "", f"简历文件不存在: {file_path}"

        ext = os.path.splitext(resolved)[1].lower()
        try:
            if ext == ".pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(resolved)
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages).strip()
                if not text:
                    return "", "PDF 未提取到文本，可能是扫描件或图片简历，请上传可搜索文本的 PDF"
                return text, None
            if ext in (".docx",):
                from docx import Document
                doc = Document(resolved)
                text = "\n".join(p.text for p in doc.paragraphs).strip()
                if not text:
                    return "", "Word 文档未提取到文本"
                return text, None
            if ext == ".doc":
                return "", "暂不支持旧版 .doc 格式，请转换为 .docx 或 PDF 后重新上传"
            if ext in (".txt", ".md"):
                with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip(), None
            return "", f"不支持的文件格式: {ext}"
        except ImportError as e:
            return "", f"缺少文件解析依赖: {e}"
        except Exception as e:
            logger.exception("Failed to extract text from %s", resolved)
            return "", f"文件解析失败: {e}"


def normalize_text_field(value, default: str = UNKNOWN) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    return str(value).strip() or default


def calculate_age(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> Optional[int]:
    """根据出生年月日计算周岁年龄。"""
    try:
        birth = date(birth_year, birth_month, birth_day)
    except ValueError:
        return None
    today = date.today()
    if birth > today:
        return None
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age if 16 <= age <= 70 else None


def infer_age_from_text(text: str) -> Optional[int]:
    """从简历文本中的出生日期推算年龄（当前日期减出生日期）。"""
    if not text:
        return None

    # 完整日期：2003.12.12 / 2003-12-12 / 2003年12月12日
    full_date_patterns = [
        r"(?:出生(?:日期|年月)?|生日|生)[日：:\s]*?(20\d{2}|19\d{2})\s*[年./-]\s*(\d{1,2})\s*[月./-]?\s*(\d{1,2})?\s*日?",
        r"(20\d{2}|19\d{2})\s*[年./-]\s*(\d{1,2})\s*[月./-]\s*(\d{1,2})\s*日?",
        r"(20\d{2}|19\d{2})\.(\d{1,2})\.(\d{1,2})",
        r"(20\d{2}|19\d{2})/(\d{1,2})/(\d{1,2})",
    ]
    for pattern in full_date_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.lastindex and match.lastindex >= 3 and match.group(3) else 1
        age = calculate_age(year, month, day)
        if age is not None:
            return age

    # 仅出生年份
    year_only_patterns = [
        r"(?:出生(?:日期|年月)?|生日|生)[日：:\s]*?(20\d{2}|19\d{2})\s*年?",
        r"(?:生于|出生于)\s*(20\d{2}|19\d{2})\s*年?",
    ]
    for pattern in year_only_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        age = calculate_age(int(match.group(1)))
        if age is not None:
            return age

    return None


def infer_age_from_education(parsed: dict) -> Optional[int]:
    """出生日期缺失时，从教育经历反推年龄。

    按国内常规"18 岁上大学"估算：age = 当前年份 - 入学年份 + 18。
    取教育经历里最早的入学年份（通常是本科入学），并排除高中/中学条目。
    """
    today = date.today()
    start_years: list[int] = []
    for edu in (parsed.get("educationHistory") or []):
        if not isinstance(edu, dict):
            continue
        degree = str(edu.get("degree") or "")
        school = str(edu.get("school") or "")
        if "高中" in degree or "中学" in school or "高中" in school:
            continue
        years = re.findall(r"(20\d{2}|19\d{2})", str(edu.get("period") or ""))
        start_years.extend(int(y) for y in years)
    if not start_years:
        return None
    enrollment_year = min(start_years)
    age = today.year - enrollment_year + 18
    return age if 16 <= age <= 70 else None


def resolve_age(parsed: dict, raw_text: str) -> Optional[int]:
    """优先从出生日期计算年龄；缺失则从教育经历按"18岁上大学"反推；最后才用解析结果中的年龄字段。"""
    for source in (raw_text, str(parsed.get("birthDate") or "")):
        if not source.strip():
            continue
        inferred = infer_age_from_text(source)
        if inferred is not None:
            return inferred

    edu_age = infer_age_from_education(parsed)
    if edu_age is not None:
        return edu_age

    age = parsed.get("age")
    if age in (None, "", UNKNOWN, 0, "0"):
        return None
    try:
        parsed_age = int(age)
        return parsed_age if parsed_age > 0 else None
    except (TypeError, ValueError):
        return None


# 维度打分已拆分为 5 个独立子 Agent，见 app/services/resume_scoring/。
# run_resume_parse 调用 resume_scoring.score_all() 并发评分并按固定顺序组装。


def infer_native_place_from_text(text: str) -> Optional[str]:
    patterns = [
        r"籍贯[：:\s]*([^\n\r，,；;]{2,20})",
        r"户籍[：:\s]*([^\n\r，,；;]{2,20})",
        r"现居地[：:\s]*([^\n\r，,；;]{2,20})",
        r"所在地[：:\s]*([^\n\r，,；;]{2,20})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            place = match.group(1).strip()
            if place and place not in (UNKNOWN, "-"):
                return place
    return None


def normalize_ethnicity(value) -> str:
    text = normalize_text_field(value, default="")
    if not text or text == UNKNOWN:
        return DEFAULT_ETHNICITY
    return text


def normalize_experience(value, *, has_work_history: bool = False) -> str:
    text = normalize_text_field(value, default="")
    if has_work_history:
        return text if text else UNKNOWN
    if not text or text in NO_WORK_EXPERIENCE_VALUES:
        return IN_SCHOOL
    return text


def format_experience_display(experience, work_experiences) -> str:
    text = (experience or "").strip()
    if work_experiences:
        return display_text(text)
    if not text or text in NO_WORK_EXPERIENCE_VALUES:
        return IN_SCHOOL
    return display_text(text)


def enrich_parsed_fields(parsed: dict, raw_text: str) -> dict:
    """Fill missing parsed fields with inferred values or 未知."""
    result = dict(parsed or {})

    if not normalize_text_field(result.get("name"), default=""):
        result["name"] = UNKNOWN

    gender = result.get("gender")
    result["gender"] = gender if gender in ("男", "女") else UNKNOWN

    result["age"] = resolve_age(result, raw_text)

    result["education"] = resolve_verified_education(result, raw_text)
    result["experience"] = normalize_experience(
        result.get("experience"),
        has_work_history=bool(result.get("workHistory")),
    )
    result["phone"] = normalize_text_field(result.get("phone"))
    result["email"] = normalize_text_field(result.get("email"))
    result["ethnicity"] = normalize_ethnicity(result.get("ethnicity"))

    native_place = normalize_text_field(result.get("nativePlace"), default="")
    if not native_place:
        native_place = infer_native_place_from_text(raw_text) or UNKNOWN
    result["nativePlace"] = native_place

    return result


def augment_gender_from_portrait(parsed: dict, resume_file: str) -> dict:
    """When text parsing cannot determine gender, infer it from resume headshot."""
    if parsed.get("gender") in ("男", "女"):
        return parsed

    with minio_storage.resolved_local_path(resume_file or "") as resolved:
        if not resolved:
            return parsed

        # Pass client=None so infer_gender_from_vision creates its own sync
        # OpenAI client (the async llm_client in this module is not usable
        # from sync code).
        portrait_gender = infer_gender_from_resume_file(resolved, client=None)
        if portrait_gender in ("男", "女"):
            parsed["gender"] = portrait_gender
    return parsed


async def parse_resume_with_llm(text: str, position_name: str = "") -> Tuple[dict, Optional[str]]:
    """Use DeepSeek to parse resume text into structured data."""
    position_hint = f"\n目标应聘岗位：{position_name}" if position_name else ""
    prompt = f"""你是一个专业的简历解析器。请从以下简历文本中提取结构化信息，返回纯JSON格式。{position_hint}

【注意】不要输出 analysis.dimensions 字段，维度评分由系统独立子 Agent 计算。

【简历文本】
{text[:12000]}

请返回如下JSON结构（找不到的字符串字段请填"未知"，应届生经验填"应届"）：
{{
    "name": "姓名",
    "gender": "男/女/未知",
    "age": null,
    "birthDate": "出生日期，格式如2003-12-12，无则null",
    "education": "最高学历：大专/本科/硕士/博士/其他/未知",
    "experience": "工作年限，如7年；无工作经历填在校中",
    "phone": "手机号，未知填未知",
    "email": "邮箱，未知填未知",
    "ethnicity": "民族，简历未提及则填汉族",
    "nativePlace": "籍贯或现居地，未知填未知",
    "skills": ["技能1", "技能2"],
    "educationHistory": [{{"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}}],
    "workHistory": [{{"company": "公司", "role": "职位", "period": "时间段", "description": "工作描述"}}],
    "projectHistory": [{{"name": "项目名", "role": "角色", "period": "时间段", "description": "项目描述"}}],
    "analysis": {{
        "overallScore": 0-100的整数,
        "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
        "summary": "综合评价摘要",
        "positionMatch": "岗位匹配度分析",
        "experienceInsight": "经验洞察",
        "highlights": ["亮点"],
        "risks": ["风险点"],
        "recommendation": "推荐建议"
    }}
}}

只返回JSON，不要任何其他文字。

【严格要求】
1. 只能提取简历原文中明确出现的信息，严禁根据岗位或常识推测学历、年限、学校等信息。
2. education、educationHistory 仅在原文出现学校名、学历词（如本科/硕士/大专/学士）或教育时间段时填写；原文完全没有则 education 填"未知"，educationHistory 返回 []。
3. workHistory 同样仅填写原文明确写出的公司与岗位，不得臆造。
4. keywords 必须输出恰好 5 个，从简历技能、项目、工具栈中提取，不足 5 个时用技能字段补足。

注意：age字段不要自行填写，留null即可；若识别到出生日期请填入birthDate，系统会自动计算年龄。
注意：不要输出 analysis.dimensions 字段，维度评分由系统独立子 Agent 计算。"""
    try:
        response = await get_llm_client().chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        parsed = json.loads(content.strip())
        parsed = enrich_parsed_fields(parsed, text)
        if parsed.get("name") == UNKNOWN:
            return parsed, "AI 未能识别姓名，请检查简历格式"
        return parsed, None
    except json.JSONDecodeError as e:
        logger.exception("LLM returned invalid JSON")
        return {}, f"AI 返回格式错误: {e}"
    except Exception as e:
        logger.exception("LLM parse error")
        return {}, f"AI 解析失败: {e}"


CANDIDATE_LOAD_OPTIONS = (
    selectinload(Candidate.position),
    selectinload(Candidate.skills),
    selectinload(Candidate.educations),
    selectinload(Candidate.work_experiences),
    selectinload(Candidate.project_experiences),
    selectinload(Candidate.ai_analysis),
)


async def load_candidate(db: AsyncSession, candidate_id) -> Optional[Candidate]:
    result = await db.execute(
        select(Candidate)
        .options(*CANDIDATE_LOAD_OPTIONS)
        .where(Candidate.id == candidate_id)
    )
    return result.scalar_one_or_none()


def is_candidate_parsed(candidate: Candidate) -> bool:
    return bool(
        candidate.phone
        or candidate.email
        or candidate.education
        or candidate.experience
        or candidate.score
    )


def display_text(value, default: str = UNKNOWN) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    return str(value)


def serialize_candidate(c: Candidate) -> dict:
    """Convert ORM Candidate + relationships to frontend-expected dict."""
    resume_path = c.resume_file or ""
    resume_ext = os.path.splitext(resume_path)[1].lower() if resume_path else ""
    gender = c.gender if c.gender in ("男", "女") else UNKNOWN
    data = {
        "id": str(c.id),
        "name": display_text(c.name, UNKNOWN),
        "gender": gender,
        "age": c.age if c.age and c.age > 0 else None,
        "education": display_text(c.education),
        "experience": format_experience_display(c.experience, c.work_experiences),
        "position": c.position.name if c.position else UNKNOWN,
        "positionId": str(c.position_id) if c.position_id else "",
        "score": c.score or 0,
        "status": c.status or "job_hunting",
        "uploadTime": c.upload_time.isoformat() if c.upload_time else "",
        "phone": display_text(c.phone),
        "email": display_text(c.email),
        "ethnicity": normalize_ethnicity(c.ethnicity),
        "nativePlace": display_text(c.native_place),
        "skills": [s.skill for s in (c.skills or [])],
        "workHistory": [
            {"company": w.company, "role": w.role, "period": w.period, "description": w.description or ""}
            for w in (c.work_experiences or [])
        ],
        "educationHistory": [
            {"school": e.school, "degree": e.degree, "major": e.major, "period": e.period}
            for e in (c.educations or [])
        ],
        "projectHistory": [
            {"name": p.name, "role": p.role, "period": p.period, "description": p.description or ""}
            for p in (c.project_experiences or [])
        ],
        "resumeFileUrl": f"/resumes/{c.id}/file" if resume_path else "",
        "resumeFileName": os.path.basename(resume_path) if resume_path else "",
        "resumeFileType": resume_ext.lstrip("."),
        "parseStatus": "parsed" if is_candidate_parsed(c) else "pending",
        "aiAnalysis": None,
        "schoolTierLabel": None,
    }
    tier_from_keywords = None
    if c.ai_analysis and c.ai_analysis.keywords:
        for keyword in c.ai_analysis.keywords:
            if keyword in SCHOOL_TIER_TAGS:
                tier_from_keywords = keyword
                break
    data["schoolTierLabel"] = tier_from_keywords or resolve_school_tier_tag_sync({
        "education": data["education"],
        "educationHistory": data["educationHistory"],
    })
    if c.ai_analysis:
        a = c.ai_analysis
        data["aiAnalysis"] = {
            "overallScore": a.overall_score or 0,
            "dimensions": a.dimensions or [],
            "keywords": a.keywords or [],
            "summary": a.summary or "",
            "positionMatch": a.position_match or "",
            "experienceInsight": a.experience_insight or "",
            "highlights": a.highlights or [],
            "risks": a.risks or [],
            "recommendation": a.recommendation or "",
        }
    return data


async def extract_and_parse_resume(
    object_key: str,
    position_name: str = "",
    *,
    db: AsyncSession,
    resume_text: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """抽取简历文本并用 LLM 解析为结构化数据，供入库前查重或写入候选人。"""
    ai_enabled = await get_system_setting(db, "aiResumeAnalysis", True)
    if not ai_enabled:
        return None, "AI 简历分析已关闭"

    text = resume_text
    if text is None:
        text, extract_error = extract_text_from_file(object_key)
        if extract_error:
            return None, extract_error
    if not (text or "").strip():
        return None, "简历文件内容为空"

    parsed, parse_error = await parse_resume_with_llm(text, position_name)
    if parse_error and not parsed:
        return None, parse_error
    if not parsed:
        return None, "AI 未能解析简历内容"

    parsed = enrich_parsed_fields(parsed, text)
    parsed = await asyncio.to_thread(augment_gender_from_portrait, parsed, object_key or "")
    return parsed, parse_error


async def run_resume_parse(
    candidate: Candidate,
    position_name: str = "",
    db: AsyncSession = None,
    pre_parsed: Optional[dict] = None,
) -> Optional[str]:
    """Extract text, parse with LLM, and fill candidate. Returns error message if any."""
    if not candidate.resume_file:
        return "未找到简历文件"

    if db is None:
        return "内部错误：缺少数据库会话"

    # AI 简历分析开关：关闭时仅保留文件元数据，不调用 LLM
    ai_enabled = await get_system_setting(db, "aiResumeAnalysis", True)
    if not ai_enabled:
        logger.info("aiResumeAnalysis 已关闭，跳过 AI 解析 candidate=%s", candidate.id)
        return "AI 简历分析已关闭，仅保存文件"

    parse_error = None
    text, extract_error = extract_text_from_file(candidate.resume_file)
    if extract_error:
        return extract_error
    if not text.strip():
        return "简历文件内容为空"

    if pre_parsed is not None:
        parsed = pre_parsed
    else:
        parsed, parse_error = await extract_and_parse_resume(
            candidate.resume_file,
            position_name,
            db=db,
        )
        if parse_error and not parsed:
            return parse_error
        if not parsed:
            return "AI 未能解析简历内容"

    # 5 个维度各自由独立子 Agent 打分（教育背景走规则，其余 4 个走 LLM），
    # 按固定顺序组装 dimensions，覆盖主解析里可能存在的维度字段。
    from app.services.resume_scoring import score_all
    analysis = parsed.setdefault("analysis", {})
    if isinstance(analysis, dict):
        analysis["dimensions"] = await score_all(text, parsed, position_name)

    await fill_candidate_from_parsed(candidate, parsed, db)

    # 匹配分阈值：写入 screening_ai_score，低于阈值仅记录分数，不改变简历状态
    # 简历状态只能取筛选下拉框中的 6 种合法值，新上传简历保持「求职中」
    min_score = int(await get_system_setting(db, "minMatchScore", 70) or 70)
    overall = None
    if isinstance(analysis, dict):
        overall = analysis.get("overallScore")
    if overall is None:
        overall = candidate.score
    if isinstance(overall, (int, float)):
        candidate.screening_ai_score = int(overall)
        candidate.score = int(overall)
        _ = min_score  # 低匹配阈值仅用于前端按综合分数过滤展示，不再覆写状态

    return parse_error


async def fill_candidate_from_parsed(candidate: Candidate, parsed: dict, db: AsyncSession):
    """Fill candidate fields from LLM parsed result."""
    candidate.name = normalize_text_field(parsed.get("name"))
    candidate.gender = parsed.get("gender") if parsed.get("gender") in ("男", "女", UNKNOWN) else UNKNOWN

    candidate.age = parsed.get("age") if isinstance(parsed.get("age"), int) and parsed.get("age") > 0 else None

    candidate.education = normalize_text_field(parsed.get("education"))
    candidate.phone = normalize_text_field(parsed.get("phone"))
    candidate.email = normalize_text_field(parsed.get("email"))
    candidate.ethnicity = normalize_ethnicity(parsed.get("ethnicity"))
    candidate.native_place = normalize_text_field(parsed.get("nativePlace"))

    for skill in list(candidate.skills):
        await db.delete(skill)
    for skill_name in (parsed.get("skills") or []):
        if skill_name:
            candidate.skills.append(CandidateSkill(skill=str(skill_name)))

    for edu in list(candidate.educations):
        await db.delete(edu)
    for edu in (parsed.get("educationHistory") or []):
        candidate.educations.append(CandidateEducation(
            school=edu.get("school"), degree=edu.get("degree"),
            major=edu.get("major"), period=edu.get("period"),
        ))

    for work in list(candidate.work_experiences):
        await db.delete(work)
    for work in (parsed.get("workHistory") or []):
        candidate.work_experiences.append(CandidateWorkExperience(
            company=work.get("company"), role=work.get("role"),
            period=work.get("period"), description=work.get("description"),
        ))

    candidate.experience = normalize_experience(
        parsed.get("experience"),
        has_work_history=bool(candidate.work_experiences),
    )

    for proj in list(candidate.project_experiences):
        await db.delete(proj)
    for proj in (parsed.get("projectHistory") or []):
        candidate.project_experiences.append(CandidateProjectExperience(
            name=proj.get("name"), role=proj.get("role"),
            period=proj.get("period"), description=proj.get("description"),
        ))

    if candidate.ai_analysis:
        await db.delete(candidate.ai_analysis)

    tier_tag = await resolve_school_tier_tag(parsed)
    if tier_tag:
        inject_school_tier_into_keywords(parsed, tier_tag)

    analysis_data = parsed.get("analysis") or {}
    if analysis_data:
        candidate.ai_analysis = CandidateAIAnalysis(
            overall_score=analysis_data.get("overallScore"),
            summary=analysis_data.get("summary"),
            position_match=analysis_data.get("positionMatch"),
            experience_insight=analysis_data.get("experienceInsight"),
            recommendation=analysis_data.get("recommendation"),
            keywords=analysis_data.get("keywords"),
            highlights=analysis_data.get("highlights"),
            risks=analysis_data.get("risks"),
            dimensions=analysis_data.get("dimensions"),
        )
        candidate.score = analysis_data.get("overallScore") or candidate.score or 0


async def get_auto_parse_setting(db: AsyncSession) -> bool:
    return bool(await get_system_setting(db, "autoParseResume", True))


# ── Shared upload helper ─────────────────────────────────

async def _upload_one_resume(
    db: AsyncSession,
    *,
    original_name: str,
    content: bytes,
    file_ext: str,
    position_id: Optional[str] = None,
) -> dict:
    """处理单份简历上传的核心逻辑，返回统一的结果 dict。

    供 ``upload_resume``（单份）和 ``batch_upload_resumes``（批量并发）共用。
    每份文件独立管理自己的 DB 会话。
    """
    def _ok(data=None, message="ok"):
        return {"status": "success", "statusCode": 200, "message": message, "data": data}

    def _fail(status, code, message, **extra):
        return {"status": status, "statusCode": code, "message": message, "data": None, **extra}

    # 1) 保存文件到 MinIO
    object_key = f"resumes/{uuid.uuid4()}{file_ext}"
    try:
        await asyncio.to_thread(
            minio_storage.upload_bytes, object_key, content, "application/octet-stream"
        )
    except Exception as e:
        return _fail("failed", 500, f"简历存储失败: {e}")

    resume_text, extract_error = extract_text_from_file(object_key)
    if extract_error or not resume_text.strip():
        try:
            await asyncio.to_thread(minio_storage.delete_object, object_key)
        except Exception:
            pass
        return _fail("invalid", 422, f"无法读取文档内容：{extract_error or '内容为空'}")

    # 2) 解析应聘岗位
    resolved_position_id = (position_id or "").strip()
    position_source = "user"
    match_reason = ""
    position: Optional[Position] = None

    if resolved_position_id:
        pos_result = await db.execute(select(Position).where(Position.id == resolved_position_id))
        position = pos_result.scalar_one_or_none()
        if not position:
            return _fail("failed", 404, "岗位不存在")
    else:
        from app.services.recruitment.position_matcher import match_position_for_resume
        resolved_position_id, position_name, match_reason = await match_position_for_resume(db, resume_text)
        if resolved_position_id:
            position_source = "agent"
            pos_result = await db.execute(select(Position).where(Position.id == resolved_position_id))
            position = pos_result.scalar_one_or_none()
            if not position:
                logger.warning("AI 匹配到的岗位 id=%s 在数据库中不存在", resolved_position_id)
                resolved_position_id = ""
                position = None

        if not position:
            fallback_id = str(await get_system_setting(db, "defaultPositionId", "") or "")
            if fallback_id:
                pos_result = await db.execute(select(Position).where(Position.id == fallback_id))
                position = pos_result.scalar_one_or_none()
                if position:
                    resolved_position_id = fallback_id
                    position_source = "default"
                    match_reason = match_reason or "AI 匹配未命中，已回退到系统默认岗位"

        if not position:
            try:
                await asyncio.to_thread(minio_storage.delete_object, object_key)
            except Exception:
                pass
            return _fail("no_position", 400,
                         f"AI 未能判断该简历的应聘岗位：{match_reason or '无匹配'}。请手动选择岗位后重新上传。")

    # 3) 校验是否为个人求职简历
    ai_enabled = await get_system_setting(db, "aiResumeAnalysis", True)
    pre_parsed: Optional[dict] = None
    if ai_enabled:
        pre_parsed, _ = await extract_and_parse_resume(
            object_key, position.name if position else "", db=db, resume_text=resume_text
        )

    is_resume_doc, reject_reason, document_type = await validate_is_resume(
        resume_text, parsed=pre_parsed, use_llm=ai_enabled
    )
    if not is_resume_doc:
        try:
            await asyncio.to_thread(minio_storage.delete_object, object_key)
        except Exception:
            pass
        return _fail("invalid", 422, f"上传的文件不是简历：{reject_reason}")

    # 4) 姓名查重（仅按姓名判断）
    if ai_enabled and pre_parsed:
        duplicate_person = await find_duplicate_candidate(db, pre_parsed)
        if duplicate_person:
            try:
                await asyncio.to_thread(minio_storage.delete_object, object_key)
            except Exception:
                pass
            return _fail("duplicate", 409,
                         f"该候选人与已有记录为同一人（姓名一致），对应候选人：{duplicate_person.name}",
                         existingCandidateId=str(duplicate_person.id),
                         existingCandidateName=duplicate_person.name)

    # 5) 创建候选人
    candidate = Candidate(
        name=original_name,
        position_id=resolved_position_id,
        status="job_hunting",
        resume_file=object_key,
        upload_time=date.today(),
    )
    db.add(candidate)
    await db.flush()

    parse_message = None
    auto_parse = await get_auto_parse_setting(db)
    if auto_parse:
        candidate = await load_candidate(db, candidate.id)
        if candidate:
            parse_message = await run_resume_parse(
                candidate, position.name, db, pre_parsed=pre_parsed
            )
            if parse_message and is_candidate_parsed(candidate):
                parse_message = None

    await db.flush()

    # 6) RAG 知识库入库
    try:
        from app.services.recruitment.resume_kb import ingest_resume_to_kb
        await ingest_resume_to_kb(
            db, content=content, file_name=original_name,
            source_object_key=object_key, candidate_id=str(candidate.id),
        )
    except Exception as e:
        logger.warning("简历知识库入库钩子失败: %s", e)

    # 7) 通知 + Webhook
    try:
        from app.services.system.notification import notify_if
        from app.services.system.webhook import dispatch_webhook
        position_label = position.name
        if position_source == "agent":
            position_label = f"{position.name}（AI 匹配：{match_reason}）"
        await notify_if(db, "notifyNewResume", "new_resume",
                        f"新简历入库：{original_name}（岗位：{position_label}）",
                        {"candidateId": str(candidate.id)})
        await dispatch_webhook(db, "candidate.uploaded",
                               {"candidateId": str(candidate.id), "name": original_name,
                                "positionId": resolved_position_id})
    except Exception as e:
        logger.warning("通知/Webhook 钩子失败: %s", e)

    # 8) 返回结果
    candidate = await load_candidate(db, candidate.id)
    if not candidate:
        return _fail("failed", 500, "候选人加载失败")

    response_data = serialize_candidate(candidate)
    if position_source == "agent":
        response_data["positionMatchReason"] = match_reason
    if parse_message and response_data["parseStatus"] != "parsed":
        return _ok(response_data, f"简历已上传，但自动解析未完成：{parse_message}")
    return _ok(response_data)


# ── Endpoints ───────────────────────────────────────────

@router.get("/resumes")
async def list_resumes(
    positionId: str = Query("all"),
    statuses: str = Query(""),
    keyword: str = Query(""),
    sortBy: str = Query("uploadTime"),
    sortOrder: str = Query("desc"),
    page: int = Query(1),
    pageSize: int = Query(10),
    minScore: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    query = select(Candidate).options(
        selectinload(Candidate.position),
        selectinload(Candidate.skills),
        selectinload(Candidate.educations),
        selectinload(Candidate.work_experiences),
        selectinload(Candidate.project_experiences),
        selectinload(Candidate.ai_analysis),
    )

    # Filter by position
    if positionId and positionId != "all":
        query = query.where(Candidate.position_id == positionId)

    # 最低匹配分过滤：未传 minScore 时读系统设置
    score_threshold = minScore
    if score_threshold is None:
        score_threshold = int(await get_system_setting(db, "minMatchScore", 70) or 70)
    # 仅当显式传入 minScore 时强制过滤；默认阈值用于「低匹配」标记，列表仍展示全部
    if minScore is not None:
        query = query.where(Candidate.score >= score_threshold)

    # Filter by statuses
    if statuses:
        status_list = [s.strip() for s in statuses.split(",") if s.strip()]
        if status_list:
            expanded: list[str] = []
            for status in status_list:
                expanded.append(status)
                if status == "passed":
                    expanded.extend(["pending_interview"])
            query = query.where(Candidate.status.in_(expanded))

    # Keyword search
    if keyword:
        kw = f"%{keyword}%"
        query = query.outerjoin(Position, Candidate.position_id == Position.id).where(
            or_(
                Candidate.name.ilike(kw),
                Candidate.skills.any(CandidateSkill.skill.ilike(kw)),
                Position.name.ilike(kw),
            )
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Sort
    sort_col = Candidate.upload_time if sortBy == "uploadTime" else Candidate.score
    if sortOrder == "asc":
        query = query.order_by(asc(sort_col))
    else:
        query = query.order_by(desc(sort_col))

    # Paginate
    offset = (page - 1) * pageSize
    query = query.offset(offset).limit(pageSize)
    result = await db.execute(query)
    candidates = result.unique().scalars().all()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "list": [serialize_candidate(c) for c in candidates],
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.get("/resumes/{resume_id}/file")
async def get_resume_file(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Candidate).where(Candidate.id == resume_id))
    candidate = result.scalar_one_or_none()
    if not candidate or not candidate.resume_file:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "简历文件不存在", "data": None},
        )

    stored = candidate.resume_file
    filename = os.path.basename(stored) or stored

    from app.services.rag.utils import content_disposition_header
    cd_header = content_disposition_header(filename, "inline")

    # Legacy on-disk path — serve directly
    if os.path.isabs(stored) and os.path.exists(stored):
        ext = os.path.splitext(stored)[1].lower()
        media_type = MIME_TYPES.get(ext, "application/octet-stream")
        return FileResponse(
            stored,
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": cd_header},
        )

    # MinIO object key — stream from object storage
    try:
        response = await asyncio.to_thread(minio_storage.get_object_stream, stored)
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": f"简历文件不存在: {e}", "data": None},
        )

    ext = os.path.splitext(filename)[1].lower()
    media_type = MIME_TYPES.get(ext, "application/octet-stream")

    def _iter():
        try:
            for chunk in response.stream(amt=64 * 1024):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    return StreamingResponse(
        _iter(),
        media_type=media_type,
        headers={"Content-Disposition": cd_header},
    )


@router.get("/resumes/{resume_id}")
async def get_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.skills),
            selectinload(Candidate.educations),
            selectinload(Candidate.work_experiences),
            selectinload(Candidate.project_experiences),
            selectinload(Candidate.ai_analysis),
        )
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


@router.post("/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    positionId: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    original_name = file.filename or "resume.pdf"
    file_ext = os.path.splitext(original_name)[1].lower() or ".pdf"
    content = await file.read()

    result = await _upload_one_resume(
        db,
        original_name=original_name,
        content=content,
        file_ext=file_ext,
        position_id=positionId if positionId else None,
    )

    if result["status"] == "success":
        response_data = result["data"]
        return {"code": 0, "message": result["message"], "data": response_data}
    else:
        return {
            "code": result["statusCode"],
            "message": result["message"],
            "data": result.get("data"),
        }


@router.post("/resumes/batch-upload")
async def batch_upload_resumes(
    files: List[UploadFile] = File(...),
    positionId: Optional[str] = Form(None),
):
    """批量上传简历，并发处理（最多 5 份并发，双 Key 负载均衡）。返回每份文件的处理结果汇总。"""
    position_id = positionId if positionId else None

    # 先读取所有文件内容
    file_payloads: list[dict] = []
    for f in files:
        original_name = f.filename or "resume.pdf"
        file_ext = os.path.splitext(original_name)[1].lower() or ".pdf"
        content = await f.read()
        file_payloads.append({
            "original_name": original_name,
            "content": content,
            "file_ext": file_ext,
        })

    sem = asyncio.Semaphore(5)

    async def _process_one(payload: dict) -> dict:
        async with sem:
            async with async_session_factory() as task_db:
                try:
                    result = await _upload_one_resume(
                        task_db,
                        original_name=payload["original_name"],
                        content=payload["content"],
                        file_ext=payload["file_ext"],
                        position_id=position_id,
                    )
                    await task_db.commit()
                    return {
                        "fileName": payload["original_name"],
                        "status": result["status"],
                        "message": result["message"],
                        "candidateId": result["data"]["id"] if result["data"] and result["status"] == "success" else None,
                        "position": result["data"]["position"] if result["data"] and result["status"] == "success" else None,
                        "score": result["data"]["score"] if result["data"] and result["status"] == "success" else None,
                        "parseStatus": result["data"]["parseStatus"] if result["data"] and result["status"] == "success" else None,
                        "existingCandidateName": result.get("existingCandidateName"),
                    }
                except Exception:
                    await task_db.rollback()
                    raise

    tasks = [_process_one(p) for p in file_payloads]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    items = []
    success = 0
    failed = 0
    skipped = 0
    invalid = 0
    no_position = 0

    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            failed += 1
            items.append({
                "fileName": file_payloads[i]["original_name"],
                "status": "failed",
                "message": str(r),
            })
        elif isinstance(r, dict):
            s = r.get("status", "failed")
            if s == "success":
                success += 1
            elif s == "duplicate":
                skipped += 1
            elif s == "invalid":
                invalid += 1
            elif s == "no_position":
                no_position += 1
                failed += 1
            else:
                failed += 1
            items.append(r)
        else:
            failed += 1
            items.append({
                "fileName": file_payloads[i]["original_name"],
                "status": "failed",
                "message": "未知错误",
            })

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "total": len(files),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "invalid": invalid,
            "noPosition": no_position,
            "items": items,
        },
    }


@router.post("/resumes/batch-parse")
async def batch_parse(body: dict, db: AsyncSession = Depends(get_db)):
    ids = body.get("ids", [])
    if not ids:
        return {"code": 400, "message": "请提供候选人 ID 列表", "data": None}

    for cid in ids:
        result = await db.execute(
            select(Candidate)
            .options(
                selectinload(Candidate.position),
                selectinload(Candidate.skills),
                selectinload(Candidate.educations),
                selectinload(Candidate.work_experiences),
                selectinload(Candidate.project_experiences),
                selectinload(Candidate.ai_analysis),
            )
            .where(Candidate.id == cid)
        )
        candidate = result.scalar_one_or_none()
        if not candidate or not candidate.resume_file:
            continue

        error = await run_resume_parse(
            candidate, candidate.position.name if candidate.position else "", db
        )
        if error:
            logger.warning("Batch parse error for %s: %s", cid, error)

    return {"code": 0, "message": "ok", "data": None}


@router.patch("/resumes/{resume_id}")
async def update_resume(
    resume_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.skills),
            selectinload(Candidate.educations),
            selectinload(Candidate.work_experiences),
            selectinload(Candidate.project_experiences),
            selectinload(Candidate.ai_analysis),
        )
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    # Simple fields
    for field in ["name", "gender", "age", "education", "experience", "status", "phone", "email", "ethnicity"]:
        if field in body:
            setattr(candidate, field, body[field])

    if "ethnicity" in body:
        candidate.ethnicity = normalize_ethnicity(candidate.ethnicity)

    if "nativePlace" in body:
        candidate.native_place = body["nativePlace"]

    if "positionId" in body and body["positionId"]:
        pos_result = await db.execute(select(Position).where(Position.id == body["positionId"]))
        if pos_result.scalar_one_or_none():
            candidate.position_id = body["positionId"]

    # Skills
    if "skills" in body:
        if candidate.skills:
            for s in list(candidate.skills):
                await db.delete(s)
        for skill_name in (body["skills"] or []):
            candidate.skills.append(CandidateSkill(skill=skill_name))

    # Education history
    if "educationHistory" in body:
        if candidate.educations:
            for e in list(candidate.educations):
                await db.delete(e)
        for edu in (body["educationHistory"] or []):
            candidate.educations.append(CandidateEducation(
                school=edu.get("school"), degree=edu.get("degree"),
                major=edu.get("major"), period=edu.get("period"),
            ))

    # Work history
    if "workHistory" in body:
        if candidate.work_experiences:
            for w in list(candidate.work_experiences):
                await db.delete(w)
        for work in (body["workHistory"] or []):
            candidate.work_experiences.append(CandidateWorkExperience(
                company=work.get("company"), role=work.get("role"),
                period=work.get("period"), description=work.get("description"),
            ))

    # Project history
    if "projectHistory" in body:
        if candidate.project_experiences:
            for p in list(candidate.project_experiences):
                await db.delete(p)
        for proj in (body["projectHistory"] or []):
            candidate.project_experiences.append(CandidateProjectExperience(
                name=proj.get("name"), role=proj.get("role"),
                period=proj.get("period"), description=proj.get("description"),
            ))

    await db.flush()
    await db.refresh(candidate)
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


@router.post("/resumes/{resume_id}/reanalyze")
async def reanalyze_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate)
        .options(
            selectinload(Candidate.position),
            selectinload(Candidate.skills),
            selectinload(Candidate.educations),
            selectinload(Candidate.work_experiences),
            selectinload(Candidate.project_experiences),
            selectinload(Candidate.ai_analysis),
        )
        .where(Candidate.id == resume_id)
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        return {"code": 404, "message": "候选人不存在", "data": None}

    if not candidate.resume_file:
        return {"code": 400, "message": "该候选人没有上传简历文件", "data": None}

    error = await run_resume_parse(
        candidate, candidate.position.name if candidate.position else "", db
    )
    if error and not is_candidate_parsed(candidate):
        return {"code": 500, "message": f"解析失败: {error}", "data": None}

    await db.flush()
    candidate = await load_candidate(db, resume_id)
    if not candidate:
        return {"code": 500, "message": "候选人加载失败", "data": None}
    return {"code": 0, "message": "ok", "data": serialize_candidate(candidate)}


@router.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    db: AsyncSession = Depends(get_db),
):
    """删除简历（候选人）。

    删除该候选人意味着此人不存在了，所有相关数据一并清理。
    与「知识库管理 → 删除简历文档」共用同一套级联删除服务
    （app.services.cascade_delete），保证两边行为一致。
    """
    from app.services.system.cascade_delete import cascade_delete_by_candidate
    logger.info("删除候选人开始 candidate_id=%s", resume_id)
    deleted = await cascade_delete_by_candidate(db, resume_id, delete_resume_file=True)
    if not deleted:
        return {"code": 404, "message": "候选人不存在", "data": None}
    return {"code": 0, "message": "ok", "data": None}
