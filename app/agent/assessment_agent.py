"""面试综合评定 Agent —— 基于转写文本、问答评分与简历上下文，生成全方位分析报告。"""

from __future__ import annotations

from app.prompts import render_prompt

import json
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings

settings = get_settings()

LEVEL_LABELS = [
    (85, "优"),
    (75, "较高"),
    (60, "中等"),
    (0, "偏低"),
]


def score_level(score: int | float | None) -> str:
    if score is None:
        return "待评"
    s = int(score)
    for threshold, label in LEVEL_LABELS:
        if s >= threshold:
            return label
    return "偏低"


def _build_system_prompt(position_name: str, round_label: str) -> str:
    return render_prompt('interview/assessment_agent.md', {'position_name': position_name, 'round_label': round_label}, 'Prompt 1')


def _parse_json_object(content: str) -> dict:
    from app.utils.llm_json import extract_json_object
    parsed = extract_json_object(content)
    if parsed is None:
        raise ValueError("模型未返回可解析的 JSON 对象")
    return parsed


def _normalize_dimension_list(items: Any, defaults: list[str]) -> list[dict]:
    if not isinstance(items, list):
        items = []
    by_name = {d.get("name"): d for d in items if isinstance(d, dict) and d.get("name")}
    result = []
    for name in defaults:
        raw = by_name.get(name, {})
        score = int(raw.get("score") or 0)
        level = raw.get("level") or score_level(score)
        result.append({"name": name, "score": score, "level": level})
    return result


def _normalize_report(data: dict, fallback_score: int = 70) -> dict:
    overall = int(data.get("overallScore") or fallback_score)
    rec = data.get("recommendation") or "待定"
    if rec not in ("强烈推荐", "推荐", "待定", "不推荐"):
        rec = "待定"

    external_defaults = [
        "专业能力", "沟通表达", "基本素质", "逻辑思维",
        "形象气质", "语言描述品质", "岗位匹配度",
    ]
    internal_defaults = ["工作意向", "个性特质", "行为能力", "认知能力"]

    advantages = data.get("advantages") or {}
    disadvantages = data.get("disadvantages") or {}
    risk = data.get("riskMonitoring") or {}

    def _risk_item(key: str, default_status: str) -> dict:
        item = risk.get(key) if isinstance(risk, dict) else {}
        if not isinstance(item, dict):
            item = {}
        return {
            "status": str(item.get("status") or default_status),
            "detail": str(item.get("detail") or "暂无足够信息"),
        }

    detail_sections = data.get("detailSections") or []
    if not isinstance(detail_sections, list):
        detail_sections = []

    section_defaults = [
        ("work_intent", "工作意向"),
        ("professional", "专业能力"),
        ("behavioral", "行为能力+个性潜质"),
    ]
    by_key = {s.get("key"): s for s in detail_sections if isinstance(s, dict)}
    normalized_sections = []
    for key, title in section_defaults:
        sec = by_key.get(key) or {}
        sub_skills = sec.get("subSkills") or []
        if not isinstance(sub_skills, list):
            sub_skills = []
        evidence = sec.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        normalized_sections.append({
            "key": key,
            "title": str(sec.get("title") or title),
            "score": int(sec.get("score") or overall),
            "maxScore": 100,
            "summary": str(sec.get("summary") or "暂无详细分析"),
            "subSkills": [
                {
                    "name": str(s.get("name") or "综合"),
                    "score": int(s.get("score") or 0),
                    "analysis": str(s.get("analysis") or ""),
                }
                for s in sub_skills[:6]
                if isinstance(s, dict)
            ],
            "evidence": [
                {
                    "label": str(e.get("label") or "依据"),
                    "content": str(e.get("content") or ""),
                }
                for e in evidence[:4]
                if isinstance(e, dict)
            ],
        })

    return {
        "overallScore": overall,
        "recommendation": rec,
        "recommendationReason": str(data.get("recommendationReason") or "基于面试表现的综合评估"),
        "externalDimensions": _normalize_dimension_list(
            data.get("externalDimensions"), external_defaults
        ),
        "internalDimensions": _normalize_dimension_list(
            data.get("internalDimensions"), internal_defaults
        ),
        "advantages": {
            "tags": list(advantages.get("tags") or [])[:6] if isinstance(advantages, dict) else [],
            "summary": str(advantages.get("summary") or "") if isinstance(advantages, dict) else "",
        },
        "disadvantages": {
            "tags": list(disadvantages.get("tags") or [])[:6] if isinstance(disadvantages, dict) else [],
            "summary": str(disadvantages.get("summary") or "") if isinstance(disadvantages, dict) else "",
        },
        "riskMonitoring": {
            "identityCheck": _risk_item("identityCheck", "待核实"),
            "psychologicalRisk": _risk_item("psychologicalRisk", "待评估"),
            "backgroundCheck": _risk_item("backgroundCheck", "待核实"),
            "processMonitoring": _risk_item("processMonitoring", "正常"),
            "resumeAlerts": _risk_item("resumeAlerts", "待核实"),
        },
        "detailSections": normalized_sections,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_report(fallback_score: int = 70) -> dict:
    base_dims = lambda names: [
        {"name": n, "score": fallback_score, "level": score_level(fallback_score)}
        for n in names
    ]
    return _normalize_report({
        "overallScore": fallback_score,
        "recommendation": "待定",
        "recommendationReason": "AI 综合分析暂不可用，请结合分项评分人工判断",
        "externalDimensions": base_dims([
            "专业能力", "沟通表达", "基本素质", "逻辑思维",
            "形象气质", "语言描述品质", "岗位匹配度",
        ]),
        "internalDimensions": base_dims(["工作意向", "个性特质", "行为能力", "认知能力"]),
        "advantages": {"tags": [], "summary": ""},
        "disadvantages": {"tags": [], "summary": ""},
        "riskMonitoring": {},
        "detailSections": [],
    }, fallback_score)


def _build_user_prompt(context: dict) -> str:
    candidate = context.get("candidate") or {}
    position = context.get("position") or {}
    qa_items = context.get("qaItems") or []
    segments = context.get("segments") or []
    resume = context.get("resumeAnalysis") or {}

    qa_lines = []
    for item in qa_items[:20]:
        qa_lines.append(
            f"- [{item.get('category', '综合')}] Q: {item.get('question', '')}\n"
            f"  A: {(item.get('answer') or '（无）')[:500]}\n"
            f"  AI评分: {item.get('aiScore', '—')} | 维度: {item.get('aiDimensions', [])}"
        )

    seg_lines = []
    for seg in segments:
        seg_lines.append(
            f"- {seg.get('segmentType')}: { (seg.get('content') or '')[:800]}\n"
            f"  AI评分: {seg.get('aiScore', '—')} | 维度: {seg.get('aiDimensions', [])}"
        )

    return render_prompt('interview/assessment_agent.md', {'candidate_name': candidate.get('name', ''), 'candidate_education': candidate.get('education', ''), 'candidate_experience': candidate.get('experience', ''), 'candidate_skills': ', '.join(candidate.get('skills') or []), 'report_position_name': position.get('name', ''), 'position_requirements': (position.get('requirements') or '')[:1500], 'resume_score': resume.get('overallScore', '—'), 'resume_summary': (resume.get('summary') or '')[:800], 'resume_highlights': resume.get('highlights') or [], 'resume_risks': resume.get('risks') or [], 'qa_count': len(qa_items), 'qa_lines': chr(10).join(qa_lines) if qa_lines else '（无抽取问答）', 'segment_lines': chr(10).join(seg_lines) if seg_lines else '（无）', 'transcript_excerpt': (context.get('transcriptExcerpt') or '')[:12000]}, 'Prompt 2')


async def generate_assessment_report(context: dict, fallback_score: int = 70) -> dict:
    """生成全方位面试评定报告。"""
    from app.services.ai import get_llm_client

    position_name = (context.get("position") or {}).get("name") or "未知岗位"
    round_label = "二面" if context.get("round") == "second" else "一面"
    client = get_llm_client()

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            extra_body={"thinking": {"type": "disabled"}},  # deepseek-v4-flash 关闭思考
            messages=[
                {"role": "system", "content": _build_system_prompt(position_name, round_label)},
                {"role": "user", "content": _build_user_prompt(context)},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        data = _parse_json_object(content)
        avg_qa = context.get("avgQaScore")
        fb = int(avg_qa) if avg_qa is not None else fallback_score
        return _normalize_report(data, fb)
    except Exception as exc:
        print(f"[assessment_agent] 综合评定失败，使用兜底报告：{exc}")
        avg_qa = context.get("avgQaScore")
        fb = int(avg_qa) if avg_qa is not None else fallback_score
        return _fallback_report(fb)
