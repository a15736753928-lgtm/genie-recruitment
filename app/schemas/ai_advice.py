"""AIAdvice —— 统一 AI 输出契约 (§2.2)。

字段全部 snake_case(全系统唯一例外,不做 camelCase 转换)。
evidence 不可为空列表 —— 违反即 ValidationError,上层捕获 -> code:422 + exceptions_queue。

用法:
    from app.schemas.ai_advice import AIAdvice, parse_ai_advice
    advice = AIAdvice(**raw_llm_output)      # 自动校验
    stored_json = advice.model_dump()        # 存入 JSON 列
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class AIAdvice(BaseModel):
    """AI 建议通用契约 —— 所有 AI 出口(简历评分/面试分析/录用建议/…)均用此结构。"""

    result: str
    score: int | None = None             # 0–100,无分值场景可为 None
    confidence: float                    # 0.0–1.0
    evidence: list[str]                  # ≥1 条;空 -> ValidationError
    strengths: list[str] = []
    risks: list[str] = []
    missing_information: list[str] = []
    recommended_action: str
    requires_human_confirmation: bool = True

    @field_validator("evidence")
    @classmethod
    def evidence_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("AI 建议必须包含至少一条证据(evidence 不可为空)")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"置信度必须在 0.0–1.0 之间,实际: {v}")
        return v

    @field_validator("score")
    @classmethod
    def score_range(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 100):
            raise ValueError(f"score 必须在 0–100 之间,实际: {v}")
        return v


def parse_ai_advice(raw: dict) -> AIAdvice:
    """从 LLM 原始输出 dict 解析 AIAdvice;解析失败抛 ValueError(上层 -> 422)。"""
    return AIAdvice.model_validate(raw)
