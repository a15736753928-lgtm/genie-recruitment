"""AI 建议的 Pydantic 模型，用于验证和序列化。"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class AIAdvice(BaseModel):
    """结构化 AI 建议——存为 ORM JSON 列（snake_case）。"""

    result: str = ""
    score: Optional[int] = None
    confidence: float = 0.7
    evidence: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    requires_human_confirmation: bool = False

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return max(0, min(100, v))

    @field_validator("evidence", "strengths", "risks", "missing_information", mode="before")
    @classmethod
    def _ensure_list(cls, v: object) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v] if v else []


def parse_ai_advice(raw: dict) -> AIAdvice:
    """解析并验证 AI 建议 dict，返回 AIAdvice 实例。"""
    return AIAdvice(**raw)
