"""Resume parsing — file extraction, LLM parsing, field enrichment, age inference.

Extracted from ``resumes.py``.  All parsing logic lives here so the route
file stays focused on HTTP request/response handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import asyncio
from datetime import date
from typing import Optional, Tuple

from app.config import get_settings
from app.infrastructure import minio_storage
from app.api.recruitment.resume_serializer import UNKNOWN, IN_SCHOOL, DEFAULT_ETHNICITY

settings = get_settings()
logger = logging.getLogger(__name__)

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


# ── File extraction ──────────────────────────────────────────

def resolve_resume_file_path(stored_path: str) -> Optional[str]:
    """Resolve a stored resume reference to a local file path."""
    if not stored_path:
        return None
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    try:
        ext = os.path.splitext(stored_path)[1] or ""
        return minio_storage.download_to_temp(stored_path, suffix=ext)
    except Exception as e:
        logger.warning("从 MinIO 下载简历失败: %s (key=%s)", e, stored_path)
        return None


def extract_text_from_file(file_path: str) -> Tuple[str, Optional[str]]:
    """Extract text from PDF, DOCX, or plain text files. Returns (text, error)."""
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


# ── Field normalization ──────────────────────────────────────

def normalize_text_field(value, default: str = UNKNOWN) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else default
    return str(value).strip() or default


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


# ── Age inference ───────────────────────────────────────────

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
    """从简历文本中的出生日期推算年龄。"""
    if not text:
        return None

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
    """出生日期缺失时，从教育经历按"18岁上大学"反推年龄。"""
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
    """优先从出生日期计算年龄；缺失则从教育经历按"18岁上大学"反推。"""
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


# ── Field enrichment ─────────────────────────────────────────

def enrich_parsed_fields(parsed: dict, raw_text: str) -> dict:
    """Fill missing parsed fields with inferred values or 未知."""
    from app.services.recruitment.resume_field_verify import resolve_verified_education

    result = dict(parsed or {})

    if not normalize_text_field(result.get("name"), default=""):
        result["name"] = UNKNOWN

    gender = result.get("gender")
    result["gender"] = gender if gender in ("男", "女") else UNKNOWN

    result["age"] = resolve_age(result, raw_text)

    result["education"] = resolve_verified_education(result, raw_text)
    result["experience"] = normalize_experience_resume(
        result.get("experience"),
        has_work_history=bool(result.get("workHistory")),
    )
    result["phone"] = normalize_text_field(result.get("phone"))
    result["email"] = normalize_text_field(result.get("email"))
    result["ethnicity"] = normalize_text_field(result.get("ethnicity")) or DEFAULT_ETHNICITY

    native_place = normalize_text_field(result.get("nativePlace"), default="")
    if not native_place:
        native_place = infer_native_place_from_text(raw_text) or UNKNOWN
    result["nativePlace"] = native_place

    return result


def normalize_experience_resume(value, *, has_work_history: bool = False) -> str:
    text = normalize_text_field(value, default="")
    if has_work_history:
        return text if text else UNKNOWN
    if not text or text in ("未知", "应届", "无", "暂无", "无工作经验", "0", "0年"):
        return IN_SCHOOL
    return text


def augment_gender_from_portrait(parsed: dict, resume_file: str) -> dict:
    """When text parsing cannot determine gender, infer it from resume headshot."""
    if parsed.get("gender") in ("男", "女"):
        return parsed

    with minio_storage.resolved_local_path(resume_file or "") as resolved:
        if not resolved:
            return parsed
        from app.services.recruitment.portrait_gender import infer_gender_from_resume_file
        portrait_gender = infer_gender_from_resume_file(resolved, client=None)
        if portrait_gender in ("男", "女"):
            parsed["gender"] = portrait_gender
    return parsed


# ── LLM JSON extraction ─────────────────────────────────────

from app.utils.json_utils import extract_json_from_text as _extract_json_from_llm


# ── LLM Resume Parsing ──────────────────────────────────────

async def parse_resume_with_llm(text: str, position_name: str = "") -> Tuple[dict, Optional[str]]:
    """Use LLM to parse resume text into structured data."""
    from app.services.ai import llm_chat

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
        raw = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        content = _extract_json_from_llm(raw)
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("LLM returned invalid JSON on first attempt: %s", e)
        try:
            fix_prompt = (
                f"你刚才返回了无效的 JSON。请修复后重新输出，只返回纯 JSON，不要任何其他文字。\n\n"
                f"错误：{e}\n\n"
                f"【原始任务】：{prompt}"
            )
            raw2 = await llm_chat(
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
                max_tokens=4096,
            )
            content2 = _extract_json_from_llm(raw2)
            parsed = json.loads(content2)
            logger.info("LLM JSON retry succeeded")
        except json.JSONDecodeError as e2:
            logger.error("LLM still returned invalid JSON after retry: %s", e2)
            return {}, f"AI 返回格式错误（重试后仍失败）: {e2}"
        except Exception as e2:
            logger.exception("LLM retry failed with unexpected error")
            return {}, f"AI 重试失败: {e2}"
    except Exception as e:
        logger.exception("LLM parse error")
        return {}, f"AI 解析失败: {e}"

    parsed = enrich_parsed_fields(parsed, text)
    if parsed.get("name") == UNKNOWN:
        return parsed, "AI 未能识别姓名，请检查简历格式"
    return parsed, None
