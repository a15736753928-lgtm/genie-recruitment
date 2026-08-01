"""反问环节评分 Agent —— 专门对候选人「向面试官提问」片段评分的子 Agent。

设计原则（与 app.agent.scoring_agent 一致）：
- 一次只处理一个片段：接收反问原文 + 岗位上下文，输出 4 个维度的分数。
- 固定 4 个评分维度：问题深度 / 团队业务理解 / 职业规划清晰度 / 沟通。
- 输出极简：只输出综合分 + 4 个维度分（及维度名），不输出长篇点评，稳定且省 token。
- 无状态：每次调用独立，可安全并发。
- 失败返回兜底分数，调用方据此落库，保证前端始终有 AI 参考评分。

反问环节考察的是候选人的思考深度、对团队/业务的关注、自身规划的清晰度与沟通表达，
与问答环节的技术深度维度不同，故独立成一个 Agent，自带本组维度常量。
"""

from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()


# ── 反问环节评分维度（固定 4 个）──────────────────────────────────
REVERSE_DIMENSIONS: List[str] = ["问题深度", "团队业务理解", "职业规划清晰度", "沟通"]

# 每个维度的评分量表（rubric），供 Agent 保持评分一致性。
DIMENSION_RUBRIC = {
    "问题深度": "提出的问题是否触及岗位/技术/业务的核心，是否有思考深度，而非泛泛而谈或可轻易查到。",
    "团队业务理解": "问题是否体现出对目标团队、业务方向、协作方式的关注与初步理解。",
    "职业规划清晰度": "是否通过提问展现出清晰的职业规划与成长诉求，与岗位发展路径是否匹配。",
    "沟通": "提问是否表达得体、逻辑清楚、节奏合适，能否与面试官形成有效互动。",
}


def _build_system_prompt(position_name: str) -> str:
    rubric_lines = "\n".join(f"- {name}：{desc}" for name, desc in DIMENSION_RUBRIC.items())
    return f"""你是一位资深的面试评分专家（反问环节评分 Agent），正在为「{position_name}」岗位的候选人面试的「反问环节」评分。

反问环节指面试结尾候选人向面试官提问的部分。

你的职责：
1. 仔细阅读候选人在反问环节提出的问题原文，并**结合该岗位的实际要求（岗位职责/任职要求/技术栈）**判断提问质量。
2. 从以下 4 个维度独立打分，每维 0-100 分：
{rubric_lines}

评分原则：
- 必须结合岗位要求进行评分，不要脱离岗位凭感觉打分。
- 「问题深度」：提问是否切中该岗位的核心职责、技术难点或业务挑战，而非泛泛而谈。
- 「团队业务理解」：提问是否体现出对岗位所在团队业务方向、协作方式、技术栈的关注与初步理解。
- 「职业规划清晰度」：是否通过提问展现出与该岗位发展路径匹配的清晰规划。
- 严格基于候选人提问内容本身评分，不要臆造候选人未提及的信息。
- 提问空泛、跑题、只问薪资福利等表面问题时该维度低分（<50）。
- 提问有深度、关注团队业务、体现清晰规划且沟通得体时该维度高分（>=80）。
- 综合分 score 为 4 个维度的加权综合（问题深度与团队业务理解权重略高），不是简单平均。
- 若候选人未提问或内容为空，4 个维度均给 0 分，综合分 0。

输出格式：必须是纯 JSON 对象，不要包含任何 markdown 或解释文字，结构如下：
{{
  "score": <整数 0-100>,
  "dimensions": [
    {{"name": "问题深度", "score": <整数>}},
    {{"name": "团队业务理解", "score": <整数>}},
    {{"name": "职业规划清晰度", "score": <整数>}},
    {{"name": "沟通", "score": <整数>}}
  ]
}}

维度顺序必须与上面一致，只输出这 4 个维度的分数，不要输出其他内容。只返回 JSON。"""


def _build_user_prompt(reverse_text: str, position_name: str, position_requirements: str) -> str:
    return f"""请对以下反问环节片段进行评分。

【岗位】{position_name or '未知'}
【岗位要求】
{position_requirements or '（无岗位要求信息）'}

【候选人向面试官提出的问题】
{reverse_text or '（无反问）'}

请结合岗位要求，按系统提示的维度和格式输出评分 JSON。"""


def _fallback_result(fallback_score: int) -> dict:
    """构造兜底评分结果（调用方据此落库，保证前端始终能看到 AI 参考评分）。"""
    return {
        "score": fallback_score,
        "dimensions": [{"name": name, "score": fallback_score} for name in REVERSE_DIMENSIONS],
    }


def _parse_scoring_json(content: str) -> dict:
    """从 LLM 返回中解析 JSON，容忍 ```json``` 包裹，并规整为固定 4 维。"""
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    data = json.loads(text.strip())
    if not isinstance(data, dict) or data.get("score") is None:
        raise ValueError(f"invalid scoring payload: {content[:200]}")
    dims = data.get("dimensions") or []
    if not isinstance(dims, list):
        dims = []
    by_name = {d.get("name"): d for d in dims if isinstance(d, dict)}
    data["dimensions"] = [
        {"name": name, "score": int(by_name.get(name, {}).get("score", 0) or 0)}
        for name in REVERSE_DIMENSIONS
    ]
    return data


async def score_reverse_question(
    reverse_text: str,
    position_name: str = "未知岗位",
    fallback_score: int = 75,
    position_requirements: str = "",
) -> dict:
    """反问环节评分 Agent 主入口：对反问片段评分，返回结构化 dict。

    评分需结合岗位要求（position_requirements，含职责/任职要求/技术栈），
    判断候选人提问是否贴合岗位实际。

    返回：{"score": int, "dimensions": [{"name", "score"}, ...]}（固定 4 维）。
    LLM 调用失败时返回兜底结果（调用方仍可落库展示）。
    """
    if not reverse_text or not reverse_text.strip():
        # 没有反问内容，直接给 0 分（不浪费 LLM 调用）
        return _fallback_result(0)

    from app.services.ai import get_llm_client
    client = get_llm_client()
    system_prompt = _build_system_prompt(position_name)
    user_prompt = _build_user_prompt(reverse_text, position_name, position_requirements or "")

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        return _parse_scoring_json(content)
    except Exception as exc:
        print(f"[reverse_question_agent] 评分失败，使用兜底分数 {fallback_score}：{exc}")
        return _fallback_result(fallback_score)
