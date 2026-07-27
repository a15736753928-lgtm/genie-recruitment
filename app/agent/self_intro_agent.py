"""自我介绍评分 Agent —— 专门对候选人面试开场「自我介绍」片段评分的子 Agent。

设计原则（与 app.agent.scoring_agent 一致）：
- 一次只处理一个片段：接收自我介绍原文 + 岗位上下文，输出 4 个维度的分数。
- 固定 4 个评分维度：内容完整度 / 表达清晰度 / 岗位匹配度 / 亮点真实性。
- 输出极简：只输出综合分 + 4 个维度分（及维度名），不输出长篇点评，稳定且省 token。
- 无状态：每次调用独立，可安全并发。
- 失败返回兜底分数，调用方据此落库，保证前端始终有 AI 参考评分。

与问答评分 Agent 的区别：维度不同（自我介绍更看重陈述的完整度、表达、岗位契合与
亮点可信度，而非技术深度），故独立成一个 Agent，自带本组维度常量。
"""

from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()


# ── 自我介绍环节评分维度（固定 4 个）──────────────────────────────
SELF_INTRO_DIMENSIONS: List[str] = ["内容完整度", "表达清晰度", "岗位匹配度", "亮点真实性"]

# 每个维度的评分量表（rubric），供 Agent 保持评分一致性。
DIMENSION_RUBRIC = {
    "内容完整度": "是否覆盖个人背景、教育经历、工作/项目经历、求职意向等关键信息，结构是否完整。",
    "表达清晰度": "语言是否流畅、条理是否清楚、重点是否突出，能否在有限时间内把自我介绍讲明白。",
    "岗位匹配度": "陈述的经历、技能与目标岗位「{position}」的要求契合程度，是否体现出岗位相关性。",
    "亮点真实性": "是否有突出的、可信的成就或亮点，细节是否经得起追问，而非空泛堆砌或夸大。",
}


def _build_system_prompt(position_name: str) -> str:
    rubric_lines = "\n".join(
        f"- {name}：{desc.format(position=position_name) if '{position}' in desc else desc}"
        for name, desc in DIMENSION_RUBRIC.items()
    )
    return f"""你是一位资深的面试评分专家（自我介绍评分 Agent），正在为「{position_name}」岗位的候选人面试的「自我介绍」环节评分。

你的职责：
1. 仔细阅读候选人在面试开场所做的自我介绍原文，并**对照其简历**核实内容。
2. 从以下 4 个维度独立打分，每维 0-100 分：
{rubric_lines}

评分原则：
- 必须结合候选人简历进行评分，不要脱离简历凭感觉打分。
- 「内容完整度」：自我介绍是否覆盖简历中的关键经历（教育、工作/项目、技能、求职意向）；遗漏简历重要信息则扣分。
- 「亮点真实性」：自我介绍中陈述的亮点、成就、数据是否与简历一致、有据可查；夸大、与简历矛盾或简历无法佐证的亮点该维度低分。
- 「岗位匹配度」：结合简历经历判断其与目标岗位「{position_name}」要求的契合程度。
- 严格基于自我介绍与简历内容评分，不要臆造候选人未提及的信息。
- 自我介绍空泛、跑题、信息缺失严重时该维度低分（<50）。
- 自我介绍完整、清晰、与岗位高度契合、亮点可信时该维度高分（>=80）。
- 综合分 score 为 4 个维度的加权综合（内容完整度与岗位匹配度权重略高），不是简单平均。
- 若候选人未做自我介绍或内容为空，4 个维度均给 0 分，综合分 0。

输出格式：必须是纯 JSON 对象，不要包含任何 markdown 或解释文字，结构如下：
{{
  "score": <整数 0-100>,
  "dimensions": [
    {{"name": "内容完整度", "score": <整数>}},
    {{"name": "表达清晰度", "score": <整数>}},
    {{"name": "岗位匹配度", "score": <整数>}},
    {{"name": "亮点真实性", "score": <整数>}}
  ]
}}

维度顺序必须与上面一致，只输出这 4 个维度的分数，不要输出其他内容。只返回 JSON。"""


def _build_user_prompt(self_intro_text: str, position_name: str, resume_text: str) -> str:
    return f"""请对以下自我介绍片段进行评分。

【岗位】{position_name or '未知'}
【候选人简历摘要】
{resume_text or '（无简历）'}

【候选人自我介绍】
{self_intro_text or '（无自我介绍）'}

请对照简历，按系统提示的维度和格式输出评分 JSON。"""


def _fallback_result(fallback_score: int) -> dict:
    """构造兜底评分结果（调用方据此落库，保证前端始终能看到 AI 参考评分）。"""
    return {
        "score": fallback_score,
        "dimensions": [{"name": name, "score": fallback_score} for name in SELF_INTRO_DIMENSIONS],
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
        for name in SELF_INTRO_DIMENSIONS
    ]
    return data


async def score_self_intro(
    self_intro_text: str,
    position_name: str = "未知岗位",
    fallback_score: int = 75,
    resume_text: str = "",
) -> dict:
    """自我介绍评分 Agent 主入口：对自我介绍片段评分，返回结构化 dict。

    评分需结合候选人简历（resume_text），对照核实自我介绍的真实性与完整度。

    返回：{"score": int, "dimensions": [{"name", "score"}, ...]}（固定 4 维）。
    LLM 调用失败时返回兜底结果（调用方仍可落库展示）。
    """
    if not self_intro_text or not self_intro_text.strip():
        # 没有自我介绍内容，直接给 0 分（不浪费 LLM 调用）
        return _fallback_result(0)

    from app.services.ai import get_llm_client
    client = get_llm_client()
    system_prompt = _build_system_prompt(position_name)
    # 简历文本可能较长，截断避免超长 prompt
    resume_excerpt = (resume_text or "")[:3000]
    user_prompt = _build_user_prompt(self_intro_text, position_name, resume_excerpt)

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
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
        print(f"[self_intro_agent] 评分失败，使用兜底分数 {fallback_score}：{exc}")
        return _fallback_result(fallback_score)
