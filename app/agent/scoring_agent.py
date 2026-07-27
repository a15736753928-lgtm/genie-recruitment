"""面试评分 Agent —— 专门对候选人单个面试回答进行评分的子 Agent。

设计原则（按需求）：
- 一次只处理一个问题：接收单个问题 + 回答原文，输出 3 个维度的分数。
- 固定 3 个评分维度：表达能力、逻辑思维、技术深度。这是「单题评分」专用维度，
  与面试轮次评定的 R1_DIMS/R2_DIMS（app/api/talent/phase1_interview.py）是两回事，不要混用。
- 输出极简：只输出这 3 个数字（及维度名），不输出长篇点评，稳定且省 token。
- 无状态：每次调用独立，可安全并发。批量评分时用 asyncio.gather 并发调用多个 Agent 实例。
- 失败返回兜底分数，调用方据此落库，保证前端 AI 参考评分栏始终有内容。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()


# ── 单题评分维度（固定 3 个）────────────────────────────────────
SCORING_DIMENSIONS: List[str] = ["表达能力", "逻辑思维", "技术深度"]

# 每个维度的评分量表（rubric），供 Agent 保持评分一致性。
DIMENSION_RUBRIC = {
    "表达能力": "表述是否清晰、结构化、重点突出；能否用准确语言把问题讲明白。",
    "逻辑思维": "分析问题是否条理清楚、推理严谨；思路是否有层次、是否能抓住关键。",
    "技术深度": "对相关技术原理、底层机制、最佳实践的理解深度；能否讲清 why 而非 only how。",
}


@dataclass
class ScoringResult:
    """评分 Agent 的结构化输出。"""
    score: int                  # 综合分 0-100（3 维加权综合）
    dimensions: List[dict]      # [{"name": "...", "score": int}, ...] 固定 3 项


def _build_system_prompt(position_name: str) -> str:
    """构造评分 Agent 的系统提示词。"""
    rubric_lines = "\n".join(f"- {name}：{desc}" for name, desc in DIMENSION_RUBRIC.items())
    return f"""你是一位资深的技术面试评分专家（面试评分 Agent），正在为「{position_name}」岗位的候选人面试回答评分。

你的职责：
1. 仔细阅读面试官提出的问题和候选人的回答原文。
2. 从以下 3 个维度独立打分，每维 0-100 分：
{rubric_lines}

评分原则：
- 严格基于回答内容本身评分，不要臆造候选人未提及的信息。
- 回答空泛、跑题、有明显错误时该维度低分（<50）。
- 回答准确、有深度、有条理时该维度高分（>=80）。
- 综合分 score 为 3 个维度的加权综合（技术深度与逻辑思维权重略高），不是简单平均。
- 若候选人未作答或回答为空，3 个维度均给 0 分，综合分 0。

输出格式：必须是纯 JSON 对象，不要包含任何 markdown 或解释文字，结构如下：
{{
  "score": <整数 0-100>,
  "dimensions": [
    {{"name": "表达能力", "score": <整数>}},
    {{"name": "逻辑思维", "score": <整数>}},
    {{"name": "技术深度", "score": <整数>}}
  ]
}}

维度顺序必须与上面一致，只输出这 3 个维度的分数，不要输出其他内容。只返回 JSON。"""


def _build_user_prompt(question_content: str, category: str, difficulty: str, answer: str) -> str:
    return f"""请对以下面试问答进行评分。

【题目】{question_content}
【分类】{category or '综合'}
【难度】{difficulty or 'medium'}
【候选人回答】
{answer or '（无回答）'}

请按系统提示的维度和格式输出评分 JSON。"""


def _fallback_result(fallback_score: int) -> dict:
    """构造兜底评分结果（调用方据此落库，保证前端始终能看到 AI 参考评分）。"""
    return {
        "score": fallback_score,
        "dimensions": [{"name": name, "score": fallback_score} for name in SCORING_DIMENSIONS],
    }


def _parse_scoring_json(content: str) -> dict:
    """从 LLM 返回中解析 JSON，容忍 ```json``` 包裹，并规整为固定 3 维。"""
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
        for name in SCORING_DIMENSIONS
    ]
    return data


async def score_answer(
    question_content: str,
    answer: str,
    position_name: str = "未知岗位",
    category: str = "",
    difficulty: str = "medium",
    fallback_score: int = 75,
) -> dict:
    """评分 Agent 主入口：对单个问答评分，返回结构化 dict。

    一次只处理一个问题。无状态，可安全并发调用。
    返回：{"score": int, "dimensions": [{"name", "score"}, ...]}（固定 3 维）。
    LLM 调用失败时返回兜底结果（调用方仍可落库展示）。
    """
    from app.services.ai import get_llm_client
    client = get_llm_client()
    system_prompt = _build_system_prompt(position_name)
    user_prompt = _build_user_prompt(question_content, category, difficulty, answer)

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
        print(f"[scoring_agent] 评分失败，使用兜底分数 {fallback_score}：{exc}")
        return _fallback_result(fallback_score)


async def score_answers_batch(
    items: List[dict],
    position_name: str = "未知岗位",
    fallback_score: int = 75,
    concurrency: int = 5,
) -> List[dict]:
    """并发评分多个问答——把评分 Agent 复制多份并行调用。

    Args:
        items: [{"question_content", "answer", "category", "difficulty"}, ...]
        position_name: 岗位名，作为评分上下文
        fallback_score: 兜底分数
        concurrency: 最大并发数，避免一次性打爆 LLM 限流

    Returns:
        与 items 等长的结果列表，顺序与输入一致。
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _score_one(item: dict) -> dict:
        async with semaphore:
            return await score_answer(
                question_content=item.get("question_content", ""),
                answer=item.get("answer", ""),
                position_name=position_name,
                category=item.get("category", ""),
                difficulty=item.get("difficulty", "medium"),
                fallback_score=fallback_score,
            )

    return await asyncio.gather(*[_score_one(it) for it in items])
