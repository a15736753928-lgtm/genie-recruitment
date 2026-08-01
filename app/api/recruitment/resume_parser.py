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


async def extract_text_from_file(file_path: str) -> Tuple[str, Optional[str]]:
    """Extract text from PDF/image — 本地 OCR 优先 + MIMO 多模态视觉兜底; DOCX falls back.

    混合 OCR+LLM（2026-08-01 启用）：PDF/图片先走本地 RapidOCR 抽文字（快、免费、
    字符保真），OCR 输出过短或不可用时回退到 MIMO 多模态视觉，准确率不降。
    """
    with minio_storage.resolved_local_path(file_path) as resolved:
        if not resolved:
            return "", f"简历文件不存在: {file_path}"

        ext = os.path.splitext(resolved)[1].lower()
        try:
            if ext == ".pdf":
                # 1) 文字层优先（文本型 PDF 毫秒级；扫描件 get_text 为空才继续走 OCR/视觉）
                try:
                    import fitz
                    doc = fitz.open(resolved)
                    try:
                        layer_text = "\n".join(page.get_text() for page in doc)
                    finally:
                        doc.close()
                    if len((layer_text or "").strip()) >= get_settings().ocr_fallback_threshold:
                        return layer_text.strip(), None
                except Exception as e:
                    logger.warning("PDF 文字层抽取失败（走 OCR 兜底）: %s", e)
                # 2) OCR（扫描件）
                ocr_text = await _extract_ocr_first(resolved)
                if ocr_text is not None:
                    return ocr_text, None
                # 3) MIMO 多模态视觉兜底
                from app.services.ai.vision import extract_text_from_vision
                return await extract_text_from_vision(resolved)
            if ext in (".docx",):
                from docx import Document
                doc = Document(resolved)
                text = "\n".join(p.text for p in doc.paragraphs).strip()
                if not text:
                    return "", "Word 文档未提取到文本"
                return text, None
            if ext == ".doc":
                return "", "暂不支持旧版 .doc 格式，请转换为 .docx 或 PDF 后重新上传"
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                ocr_text = await _extract_ocr_first(resolved)
                if ocr_text is not None:
                    return ocr_text, None
                from app.services.ai.vision import extract_text_from_vision
                return await extract_text_from_vision(resolved)
            if ext in (".txt", ".md"):
                with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip(), None
            return "", f"不支持的文件格式: {ext}"
        except ImportError as e:
            return "", f"缺少文件解析依赖: {e}"
        except Exception as e:
            logger.exception("Failed to extract text from %s", resolved)
            return "", f"文件解析失败: {e}"


# ── 混合 OCR+LLM 辅助 ──────────────────────────────────────

async def _extract_ocr_first(file_path: str) -> Optional[str]:
    """本地 RapidOCR 优先抽取；OCR 不可用或文本过短返回 None（调用方回退多模态视觉）。"""
    from app.config import get_settings
    s = get_settings()
    if not s.ocr_enabled:
        return None
    from app.services.ai import ocr as ocr_service
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            ocr_text = await asyncio.to_thread(ocr_service.ocr_pdf, file_path)
        else:
            with open(file_path, "rb") as fh:
                ocr_text = await asyncio.to_thread(ocr_service.ocr_image_bytes, fh.read())
    except Exception as e:
        logger.warning("本地 OCR 抽取失败，回退多模态: %s", e)
        return None
    ocr_text = (ocr_text or "").strip()
    if len(ocr_text) < s.ocr_fallback_threshold:
        logger.info("本地 OCR 文本过短（%d 字 < 阈值 %d），回退多模态视觉",
                    len(ocr_text), s.ocr_fallback_threshold)
        return None
    return ocr_text


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
    if gender not in ("男", "女"):
        # OCR 文本里的「性别：男/女」本地正则兜底，避免无谓触发 MIMO 视觉识别
        m = re.search(r"性别[:：\s]*([男女MmFfVv])", raw_text or "")
        if m:
            gender = {"男": "男", "M": "男", "m": "男",
                      "女": "女", "F": "女", "f": "女", "V": "女", "v": "女"}.get(m.group(1))
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


async def augment_gender_from_portrait(parsed: dict, resume_file: str) -> dict:
    """When text parsing cannot determine gender, infer it from resume headshot (multimodal)."""
    if parsed.get("gender") in ("男", "女"):
        return parsed

    with minio_storage.resolved_local_path(resume_file or "") as resolved:
        if not resolved:
            return parsed
        from app.services.ai.vision import infer_gender_from_vision
        portrait_gender = await infer_gender_from_vision(resolved)
        if portrait_gender in ("男", "女"):
            parsed["gender"] = portrait_gender
    return parsed


# ── LLM JSON extraction ─────────────────────────────────────

from app.utils.json_utils import extract_json_from_text as _extract_json_from_llm
from app.utils.llm_json import extract_json_object as _extract_json_obj


# ── LLM Resume Parsing ──────────────────────────────────────

# ── 并行解析：3 个独立 LLM 调用分别提取不同类型的信息 ────────
# 每个 prompt 更聚焦、输出更短、不会截断，还能并行执行降低总耗时。

async def _parse_profile(text: str, position_hint: str) -> dict:
    """LLM 调用 1：提取基本档案（个人信息 + 教育/工作/项目经历）"""
    from app.services.ai import llm_chat

    prompt = f"""你是一个专业的简历解析器。请从简历文本中提取个人基本信息和经历，返回纯JSON。
{position_hint}

【简历文本】
{text[:12000]}

返回如下JSON（找不到的填"未知"，应届经验填"应届"）：
{{
    "name": "姓名",
    "gender": "男/女/未知",
    "age": null,
    "birthDate": "出生日期如2003-12-12，无则null",
    "education": "最高学历：大专/本科/硕士/博士/其他/未知",
    "experience": "工作年限如7年；无工作经历填在校中",
    "phone": "手机号，未知填未知",
    "email": "邮箱，未知填未知",
    "ethnicity": "民族，未提及填汉族",
    "nativePlace": "籍贯或现居地，未知填未知",
    "skills": ["技能1", "技能2"],
    "educationHistory": [{{"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}}],
    "workHistory": [{{"company": "公司", "role": "职位", "period": "时间段", "description": "工作描述"}}],
    "projectHistory": [{{"name": "项目名", "role": "角色", "period": "时间段", "description": "项目描述"}}]
}}

只返回JSON。严格要求：只能提取原文明确出现的信息，不得臆造。age留null，识别到出生日期填birthDate。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=4096)
    return _extract_json_obj(raw) or {}


async def _parse_extra(text: str) -> dict:
    """LLM 调用 2：提取扩展字段（行业经验/管理经验/到岗时间/薪资/作品/证书）"""
    from app.services.ai import llm_chat

    prompt = f"""从以下简历文本中提取6项特定信息，返回纯JSON。

【简历文本】
{text[:12000]}

返回如下JSON（没有则填"未知"或返回空数组）：
{{
    "industryExperience": "行业经验描述，如'3年AI行业经验'，未知填未知",
    "managementExperience": "管理经验描述，如'带过5人团队'，未知填未知",
    "availableDate": "到岗时间，如'随时到岗'/'一周内'，未知填未知",
    "salaryExpectation": "薪资要求，如'25-30K'/'面议'，未知填未知",
    "portfolio": ["作品、GitHub、博客等链接或名称"],
    "certificates": ["职业证书、资格认证名称"]
}}

只返回JSON。严格只提取原文明确出现的信息，不得推测。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=2048)
    return _extract_json_obj(raw) or {}


async def _parse_analysis(text: str, position_hint: str) -> dict:
    """LLM 调用 3：AI 综合评估（评分 + 关键词 + 优劣势分析）"""
    from app.services.ai import llm_chat

    prompt = f"""你是资深HR分析师。请分析以下简历并给出综合评估，返回纯JSON。
{position_hint}

【简历文本】
{text[:12000]}

返回如下JSON：
{{
    "overallScore": 0到100的整数,
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
    "summary": "综合评价摘要（50-100字）",
    "positionMatch": "岗位匹配度分析",
    "experienceInsight": "经验洞察",
    "highlights": ["亮点1", "亮点2"],
    "risks": ["风险点1"],
    "recommendation": "推荐建议"
}}

只返回JSON。注意：不要输出dimensions字段。keywords恰好5个，不足用技能补。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=2048)
    return _extract_json_obj(raw) or {}


# ── 细拆解析（2026-08-01）：把 profile/analysis 两个大 JSON 拆成多个聚焦小 JSON，
#    每个输出几百 token（vs 4096），8 路并行总耗时=最慢一路，且聚焦任务更稳 ──

async def _parse_basic(text: str, position_hint: str) -> dict:
    """细拆 1：基本信息（不含经历数组）"""
    from app.services.ai import llm_chat

    prompt = f"""你是专业的简历解析器。请从简历文本中提取个人基本信息，返回纯JSON。
{position_hint}

【简历文本】
{text[:12000]}

返回如下JSON（找不到的填"未知"，应届经验填"应届"）：
{{
    "name": "姓名",
    "gender": "男/女/未知",
    "age": null,
    "birthDate": "出生日期如2003-12-12，无则null",
    "education": "最高学历：大专/本科/硕士/博士/其他/未知",
    "experience": "工作年限如7年；无工作经历填在校中",
    "phone": "手机号，未知填未知",
    "email": "邮箱，未知填未知",
    "ethnicity": "民族，未提及填汉族",
    "nativePlace": "籍贯或现居地，未知填未知"
}}

只返回JSON。严格要求：只能提取原文明确出现的信息，不得臆造。age留null，识别到出生日期填birthDate。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_education(text: str) -> dict:
    """细拆 2：教育经历"""
    from app.services.ai import llm_chat

    prompt = f"""从简历文本中提取教育经历列表，返回纯JSON。
【简历文本】
{text[:12000]}

返回如下JSON（没有则返回空数组）：
{{"educationHistory": [{{"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}}]}}

只返回JSON。严格只提取原文明确出现的信息。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_work(text: str) -> dict:
    """细拆 3：工作经历"""
    from app.services.ai import llm_chat

    prompt = f"""从简历文本中提取工作经历列表，返回纯JSON。
【简历文本】
{text[:12000]}

返回如下JSON（没有则返回空数组）：
{{"workHistory": [{{"company": "公司", "role": "职位", "period": "时间段", "description": "工作描述"}}]}}

只返回JSON。严格只提取原文明确出现的信息。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_project(text: str) -> dict:
    """细拆 4：项目经历"""
    from app.services.ai import llm_chat

    prompt = f"""从简历文本中提取项目经历列表，返回纯JSON。
【简历文本】
{text[:12000]}

返回如下JSON（没有则返回空数组）：
{{"projectHistory": [{{"name": "项目名", "role": "角色", "period": "时间段", "description": "项目描述"}}]}}

只返回JSON。严格只提取原文明确出现的信息。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_skills(text: str) -> dict:
    """细拆 5：专业技能"""
    from app.services.ai import llm_chat

    prompt = f"""从简历文本中提取专业技能列表，返回纯JSON。
【简历文本】
{text[:12000]}

返回如下JSON（没有则返回空数组）：
{{"skills": ["技能1", "技能2"]}}

只返回JSON。严格只提取原文明确出现的技能。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_score(text: str, position_hint: str) -> dict:
    """细拆 6：AI 评分 + 关键词"""
    from app.services.ai import llm_chat

    prompt = f"""你是资深HR分析师。请分析简历并给出评分和关键词，返回纯JSON。
{position_hint}

【简历文本】
{text[:12000]}

返回如下JSON：
{{
    "overallScore": 0到100的整数,
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}}

keywords恰好5个，不足用技能补。只返回JSON。不要输出其他字段。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return _extract_json_obj(raw) or {}


async def _parse_insight(text: str, position_hint: str) -> dict:
    """细拆 7：AI 洞察（摘要/匹配/经验/亮点/风险/建议）"""
    from app.services.ai import llm_chat

    prompt = f"""你是资深HR分析师。请分析简历并给出洞察，返回纯JSON。
{position_hint}

【简历文本】
{text[:12000]}

返回如下JSON：
{{
    "summary": "综合评价摘要（50-100字）",
    "positionMatch": "岗位匹配度分析",
    "experienceInsight": "经验洞察",
    "highlights": ["亮点1", "亮点2"],
    "risks": ["风险点1"],
    "recommendation": "推荐建议"
}}

只返回JSON。不要输出overallScore或keywords字段。"""
    raw = await llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1536)
    return _extract_json_obj(raw) or {}


async def _safe_parse(coro, label: str) -> dict:
    """安全执行单个 LLM 解析任务，失败返回空 dict 并记录日志。"""
    try:
        return await coro
    except Exception as e:
        logger.warning("简历并行解析 [%s] 失败: %s", label, e)
        return {}


async def parse_resume_with_llm(text: str, position_name: str = "") -> Tuple[dict, Optional[str]]:
    """Use LLM to parse resume text into structured data (3 parallel calls)."""
    import asyncio

    position_hint = f"\n目标应聘岗位：{position_name}" if position_name else ""

    # 三个解析任务并行执行
    profile_task, extra_task, analysis_task = await asyncio.gather(
        _safe_parse(_parse_profile(text, position_hint), "基本档案"),
        _safe_parse(_parse_extra(text), "扩展信息"),
        _safe_parse(_parse_analysis(text, position_hint), "AI评估"),
    )

    # 合并结果
    parsed = {**profile_task, **extra_task, "analysis": analysis_task}

    parsed = enrich_parsed_fields(parsed, text)
    if parsed.get("name") == UNKNOWN:
        return parsed, "AI 未能识别姓名，请检查简历格式"
    return parsed, None
