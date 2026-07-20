"""岗位匹配 Agent —— 根据简历内容 + 当前在招岗位，AI 自动判断应聘岗位。

上传简历时不再强制要求用户先选择应聘岗位，而是由本 Agent 读取所有在招岗位
（含 JD 要求、技术栈等结构化字段），结合简历文本，由 LLM 选出最匹配的岗位。

返回：
    (position_id, position_name, reason)  —— 命中时
    (None, None, reason)                  —— 无在招岗位 / LLM 无法判断时
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.recruitment import Position
from app.services.ai import get_llm_client

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

    # 精简后的岗位清单，喂给 LLM
    positions_brief = [_serialize_position_for_matching(p) for p in positions]

    prompt = (
        "你是一个招聘助理 Agent，任务是根据候选人简历内容，从公司当前在招的岗位列表中"
        "选出**最匹配**的一个岗位。\n\n"
        "## 在招岗位列表\n"
        f"{json.dumps(positions_brief, ensure_ascii=False)}\n\n"
        "## 候选人简历文本\n"
        f"{resume_text[:8000]}\n\n"
        "## 判断规则\n"
        "1. 优先看简历中明确写出的「求职意向 / 期望岗位 / 应聘岗位」字段；\n"
        "2. 其次结合工作经历、项目经历、专业技能与岗位 JD（职责 / 要求 / 技术栈）做语义匹配；\n"
        "3. 必须从上面岗位列表的 id 中选一个，不要凭空捏造；\n"
        "4. 若简历与所有岗位都不匹配（例如岗位列表为空，或简历内容明显属于完全不同的方向），"
        "请将 positionId 返回为 null，并在 reason 中说明原因。\n\n"
        "## 输出格式\n"
        "只返回纯 JSON，不要任何额外文字或 markdown 代码块：\n"
        '{"positionId": "<岗位id 或 null>", "positionName": "<岗位名 或 null>", "reason": "<一句话理由>"}'
    )

    try:
        client = get_llm_client()
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
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
