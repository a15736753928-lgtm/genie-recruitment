"""判断上传文档是否为个人求职简历。"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

from app.config import get_settings
from app.services.ai import get_llm_client

logger = logging.getLogger(__name__)
settings = get_settings()

UNKNOWN = "未知"

RESUME_SIGNAL_KEYWORDS = (
    "个人简历",
    "求职简历",
    "简历",
    "工作经历",
    "工作履历",
    "教育经历",
    "教育背景",
    "项目经验",
    "项目经历",
    "实习经历",
    "求职意向",
    "期望岗位",
    "应聘岗位",
    "联系电话",
    "专业技能",
    "自我评价",
)

NON_RESUME_KEYWORDS = (
    "甲方",
    "乙方",
    "合同编号",
    "发票代码",
    "发票号码",
    "税号",
    "参考文献",
    "abstract",
    "关键词：",
    "会议纪要",
    "采购订单",
    "营业执照",
)


def _has_plausible_name(name: str) -> bool:
    cleaned = (name or "").strip()
    if not cleaned or cleaned == UNKNOWN:
        return False
    cjk = re.sub(r"[a-zA-Z\s·.\-]", "", cleaned)
    return len(cjk) >= 2


def heuristic_parsed_is_resume(parsed: dict) -> Tuple[bool, str]:
    """根据解析结果判断是否像一份个人求职简历。"""
    if not parsed:
        return False, "未能从文档中提取简历信息"

    name = (parsed.get("name") or "").strip()
    education = (parsed.get("education") or "").strip()
    skills = parsed.get("skills") or []
    work_history = parsed.get("workHistory") or []
    education_history = parsed.get("educationHistory") or []
    experience = (parsed.get("experience") or "").strip()

    has_real_name = _has_plausible_name(name)
    has_real_edu = education and education not in (UNKNOWN, "其他")
    has_edu_history = bool(education_history)
    has_work_history = bool(work_history)
    has_real_skills = bool(skills) and any(s and str(s).strip() not in ("", UNKNOWN) for s in skills)
    has_real_exp = experience and experience not in (UNKNOWN, "应届", "在校中", "无", "暂无")

    if has_real_name and (
        has_real_edu
        or has_edu_history
        or has_work_history
        or has_real_skills
        or has_real_exp
    ):
        return True, ""

    if has_work_history or has_edu_history:
        return True, ""

    return False, "未识别到姓名、教育或工作经历等简历关键信息"


def heuristic_text_is_resume(text: str) -> Tuple[bool, str]:
    """根据文本关键词做轻量判断。"""
    content = (text or "").strip()
    if len(content) < 60:
        return False, "文档内容过短，不像简历"

    lower = content.lower()
    resume_hits = sum(1 for keyword in RESUME_SIGNAL_KEYWORDS if keyword in content)
    non_resume_hits = sum(
        1 for keyword in NON_RESUME_KEYWORDS if keyword in content or keyword.lower() in lower
    )

    if non_resume_hits >= 2 and resume_hits == 0:
        return False, "文档内容更像合同、发票或其他非简历文件"
    if resume_hits >= 2:
        return True, ""
    if resume_hits == 1 and non_resume_hits == 0:
        return True, ""
    if non_resume_hits >= 1 and resume_hits == 0:
        return False, "文档内容不像个人求职简历"

    return False, "未识别到工作经历、教育背景等简历常见内容"


async def classify_with_llm(text: str) -> Tuple[bool, str, str]:
    """使用 LLM 判断文档类型。Returns (is_resume, reason, document_type)."""
    prompt = f"""你是文档分类助手。请判断下列文本是否属于「个人求职简历 / CV」（含应届生简历）。

不属于简历的例子：劳动合同、商业合同、发票、论文、产品说明书、公司介绍、新闻稿、会议纪要、面试对话记录、空白页或乱码。

【文档文本】
{text[:8000]}

只返回 JSON，不要其他文字：
{{
  "isResume": true或false,
  "reason": "一句话说明判断依据",
  "documentType": "简历/合同/发票/论文/其他"
}}"""

    response = await get_llm_client().chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
        extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考提速
    )
    from app.utils.llm_json import extract_json_object
    payload = extract_json_object((response.choices[0].message.content or "") or "")
    if not isinstance(payload, dict):
        raise ValueError("模型未返回可解析的 JSON")
    is_resume = bool(payload.get("isResume"))
    reason = str(payload.get("reason") or "").strip()
    document_type = str(payload.get("documentType") or "其他").strip()
    return is_resume, reason, document_type


async def validate_is_resume(
    text: str,
    *,
    parsed: Optional[dict] = None,
    use_llm: bool = True,
) -> Tuple[bool, str, str]:
    """Returns (is_resume, reject_reason, document_type)."""
    content = (text or "").strip()
    if not content:
        return False, "文档内容为空", "未知"

    document_type = "未知"

    if use_llm:
        try:
            is_resume, reason, document_type = await classify_with_llm(content)
            if not is_resume:
                return False, reason or f"识别为{document_type}，不是个人求职简历", document_type
        except Exception as exc:
            logger.warning("简历类型 LLM 识别失败，回退规则判断: %s", exc)

    if parsed:
        ok, reason = heuristic_parsed_is_resume(parsed)
        if ok:
            return True, "", document_type if document_type != "未知" else "简历"
        if use_llm:
            return False, reason, document_type

    ok, reason = heuristic_text_is_resume(content)
    if ok:
        return True, "", document_type if document_type != "未知" else "简历"
    return False, reason, document_type
