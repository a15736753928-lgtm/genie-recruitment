"""Shared helpers for resume dimension sub-agents.

Each sub-agent lives in its own folder under ``resume_scoring/`` and exposes
``async def score(text, parsed, position_name) -> Optional[int]`` returning a
0-100 integer. The LLM-based agents share a single prompt template and a robust
integer-score extractor here.

For efficiency, ``score_all()`` now batches all 4 LLM-rubric dimensions into
a single API call via ``_llm_score_batched()``, eliminating 3 redundant
round-trips per resume.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.config import get_settings
from app.services.ai import get_llm_client
from app.prompts import load_prompt, render_prompt

logger = logging.getLogger(__name__)
settings = get_settings()

# 固定的 5 个维度及其展示顺序。
DIMENSIONS = ("教育背景", "项目经验", "工作经验", "专业技能", "实习经历")

# 4 个 LLM 评分维度 → 对应 prompt 目录名
_LLM_DIMENSIONS = {
    "项目经验": "project_experience",
    "工作经验": "work_experience",
    "专业技能": "professional_skill",
    "实习经历": "internship_experience",
}


def _load_prompt(folder: str) -> str:
    """加载统一管理的维度评分标准。"""
    return load_prompt(f"resume_scoring/{folder}.md")


async def _llm_score(rubric: str, resume_text: str, position_name: str) -> Optional[int]:
    """Call the LLM with a rubric + resume and parse a single 0-100 integer."""
    if not resume_text or not resume_text.strip():
        return None
    position_line = f"\n目标应聘岗位：{position_name}" if position_name else ""
    prompt = render_prompt("resume_scoring/single_score.md", {
        "rubric": rubric,
        "position_line": position_line,
        "resume_text": resume_text[:8000],
    })
    try:
        client = get_llm_client()
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
        )
        content = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d{1,3}", content)
        if not m:
            logger.warning("dimension agent returned non-numeric: %r", content)
            return None
        score = max(0, min(100, int(m.group())))
        return score
    except Exception as exc:
        logger.warning("dimension agent LLM call failed: %s", exc)
        return None


async def _llm_score_batched(
    resume_text: str,
    position_name: str,
    rubrics: Optional[dict[str, str]] = None,
) -> dict[str, Optional[int]]:
    """Score all 4 LLM dimensions in a single API call.

    Sends the same resume text once with all rubrics, receives one JSON
    response with all 4 scores.  Cuts 3 redundant LLM round-trips vs the
    old per-dimension approach.

    Args:
        resume_text: Full resume text (truncated to 8000 chars).
        position_name: Target position name for context.
        rubrics: Optional {dimension_name: rubric_text} dict.  If omitted,
                 loads all 4 from ``prompt.md`` files.

    Returns:
        {dimension_name: 0-100 score or None} for all 4 LLM dimensions.
    """
    if not resume_text or not resume_text.strip():
        return {name: None for name in _LLM_DIMENSIONS}

    if rubrics is None:
        rubrics = {name: _load_prompt(folder) for name, folder in _LLM_DIMENSIONS.items()}

    # ── Build combined prompt ──
    parts = [load_prompt("resume_scoring/batched.md", "Intro")]
    for name, rubric in rubrics.items():
        parts.append(f"\n## {name}\n{rubric}")

    if position_name:
        parts.append(f"\n目标应聘岗位：{position_name}")

    parts.append(f"\n【简历文本】\n{resume_text[:8000]}")

    field_names = "、".join(rubrics.keys())
    parts.append(render_prompt("resume_scoring/batched.md", {
        "field_names": field_names,
    }, "Output"))

    prompt = "\n".join(parts)

    # ── Two attempts: first call may occasionally return empty on overload ──
    for attempt in range(2):
        try:
            client = get_llm_client()
            resp = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": load_prompt("resume_scoring/batched.md", "System")},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                result = _parse_batched_scores(content, rubrics)
                if any(v is not None for v in result.values()):
                    return result
            logger.warning(
                "batched scoring attempt %d returned empty or unparseable: %r",
                attempt + 1, content[:100],
            )
        except Exception as exc:
            logger.warning("batched dimension LLM call attempt %d failed: %s", attempt + 1, exc)

    logger.error("batched scoring both attempts failed, all dimensions fallback to 70")
    return {name: None for name in rubrics}


def _parse_batched_scores(raw: str, rubrics: dict[str, str]) -> dict[str, Optional[int]]:
    """Parse the LLM's JSON (or near-JSON) response into per-dimension scores."""
    result: dict[str, Optional[int]] = {name: None for name in rubrics}

    # ── Attempt 1: 统一的 JSON 抽取（含围栏剥离、括号配对、尾随逗号/裸换行修复）──
    from app.utils.llm_json import extract_json_object
    scores = extract_json_object(raw)
    if isinstance(scores, dict):
        for name in rubrics:
            val = scores.get(name)
            if isinstance(val, (int, float)):
                result[name] = max(0, min(100, int(val)))
        if any(v is not None for v in result.values()):
            return result

    # ── Attempt 2: per-dimension regex fallback ──
    for name in rubrics:
        if result[name] is not None:
            continue
        pattern = re.escape(name) + r"\D*(\d{1,3})"
        m = re.search(pattern, raw)
        if m:
            result[name] = max(0, min(100, int(m.group(1))))

    if not any(v is not None for v in result.values()):
        logger.warning("batched dimension agent returned unparseable: %r", raw[:300])

    return result
