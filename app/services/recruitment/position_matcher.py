"""岗位匹配 Agent —— 根据简历内容 + 当前在招岗位，AI 自动判断应聘岗位。

上传简历时不再强制要求用户先选择应聘岗位，而是由本 Agent 读取所有在招岗位
（含 JD 要求、技术栈等结构化字段），结合简历文本，由 LLM 选出最匹配的岗位。

返回：
    (position_id, position_name, reason)  —— 命中时
    (None, None, reason)                  —— 无在招岗位 / LLM 无法判断时
"""
from __future__ import annotations

from app.prompts import render_prompt

import json
import logging
import re
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.recruitment import Position
from app.services.ai import get_llm_client
from app.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)
settings = get_settings()


def _serialize_position_for_matching(p: Position) -> dict:
    """把岗位压成 LLM 可读的精简结构（只保留对匹配有用的字段）。"""
    return {
        "id": str(p.id),
        "name": p.name,
        "department": p.department,
        "responsibilities": (p.jd_responsibilities or "")[:600],
        "requirements": (p.jd_requirements or "")[:600],
        "preferred": (p.jd_preferred or "")[:400],
        "techStack": (p.jd_tech_stack or "")[:400],
        "educationRequirement": p.education_requirement,
        "experienceRequirement": p.experience_requirement,
    }


async def _load_open_positions(db: AsyncSession) -> List[Position]:
    result = await db.execute(select(Position).order_by(Position.created_at))
    return list(result.scalars().all())


_INTENT_RE = re.compile(r"(?:求职意向|期望岗位|应聘岗位|意向岗位|目标岗位)[：:\s]*([^\n\r，,。；;·]{1,30})")


def _match_by_intent_rule(text: str, positions: List[Position]) -> Optional[Position]:
    """从简历「求职意向/期望岗位」明文快速命中岗位，省一次 LLM 调用。未命中返回 None。"""
    m = _INTENT_RE.search(text or "")
    if not m:
        return None
    intent = m.group(1).strip()
    if not intent:
        return None
    intent_lower = intent.lower()
    for p in positions:
        pname = (p.name or "").strip()
        if not pname:
            continue
        p_lower = pname.lower()
        # 岗位名与意向字段互相包含，或共享 ≥2 个汉字词元
        if p_lower in intent_lower or intent_lower in p_lower:
            return p
        overlap = sum(1 for ch in pname if ch in intent and '一' <= ch <= '鿿')
        if overlap >= 2:
            return p
    return None


async def match_position_for_resume(
    db: AsyncSession,
    resume_text: str,
) -> Tuple[Optional[str], Optional[str], str]:
    """根据简历文本，从在招岗位中选出最匹配的岗位。

    Returns:
        (position_id, position_name, reason)
        - 命中：position_id / position_name 非空，reason 为 LLM 给出的简短理由
        - 未命中：position_id / position_name 为 None，reason 解释原因
    """
    text = (resume_text or "").strip()
    if not text:
        return None, None, "简历文本为空，无法匹配岗位"

    positions = await _load_open_positions(db)
    if not positions:
        return None, None, "当前没有在招岗位，无法自动匹配"

    # 规则先行：从简历「求职意向/期望岗位」明文快速命中，省一次 LLM 调用
    rule_hit = _match_by_intent_rule(text, positions)
    if rule_hit is not None:
        return str(rule_hit.id), rule_hit.name, "按简历求职意向字段匹配"

    # 精简后的岗位清单，喂给 LLM
    positions_brief = [_serialize_position_for_matching(p) for p in positions]

    prompt = (
        render_prompt('recruitment/position_matcher.md', {'json_dumps_positions_brief_ensure_ascii_': json.dumps(positions_brief, ensure_ascii=False), 'resume_text_8000': resume_text[:8000]}, 'Prompt 1')
    )

    try:
        client = get_llm_client()
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考提速
        )
        content = (resp.choices[0].message.content or "").strip()
        data = extract_json_object(content)
        if data is None:
            raise ValueError("模型未返回可解析的 JSON")
    except json.JSONDecodeError as e:
        logger.warning("position_matcher 返回非法 JSON: %s", e)
        return None, None, f"AI 返回格式错误: {e}"
    except Exception as e:
        logger.warning("position_matcher LLM 调用失败: %s", e)
        return None, None, f"AI 岗位匹配失败: {e}"

    raw_id = data.get("positionId")
    raw_name = data.get("positionName")
    reason = str(data.get("reason") or "").strip() or "AI 未给出理由"

    if not raw_id or raw_id == "null":
        return None, None, reason or "AI 判定无匹配岗位"

    # 校验 id 确实在在招岗位列表中，防止 LLM 编造
    valid_ids = {str(p.id) for p in positions}
    valid_map = {str(p.id): p for p in positions}
    if str(raw_id) not in valid_ids:
        logger.warning(
            "position_matcher 返回的 positionId=%s 不在在招岗位列表中", raw_id
        )
        return None, None, f"AI 返回的岗位不在在招列表中: {raw_id}"

    matched = valid_map[str(raw_id)]
    return str(matched.id), matched.name, reason
