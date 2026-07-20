"""面试综合评定 Agent —— 基于转写文本、问答评分与简历上下文，生成全方位分析报告。"""

from __future__ import annotations

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
    return f"""你是一位资深 HR 与技术面试综合评估专家，正在为「{position_name}」岗位的{round_label}候选人撰写全方位面试评定报告。

你的任务不是逐题打分（已有分项评分数据），而是：
1. 综合面试转写、问答表现、自我介绍/反问环节与简历背景，给出 holistic 评估；
2. 从「外在行为」与「内在潜质」两个视角给出多维度分数与定性等级；
3. 提炼优势/劣势标签与叙述性总结；
4. 给出风险监控结论（基于简历与面试过程可观察信号，无法核实的项标注为「待核实」）；
5. 按「工作意向」「专业能力」「行为能力+个性潜质」三个板块做深度分析，含证据引用与子技能拆解。

评分原则：
- 严格基于提供的材料，不臆造未出现的事实；
- 分数 0-100，与已有 AI 分项评分大体一致，可微调综合判断；
- recommendation 只能是：强烈推荐、推荐、待定、不推荐；
- 所有分析使用中文，专业、客观、可执行。

输出必须是纯 JSON 对象（不要 markdown），结构如下：
{{
  "overallScore": <整数 0-100>,
  "recommendation": "<强烈推荐|推荐|待定|不推荐>",
  "recommendationReason": "<100字以内结论理由>",
  "externalDimensions": [
    {{"name": "专业能力", "score": <整数>, "level": "<优|较高|中等|偏低>"}},
    {{"name": "沟通表达", "score": <整数>, "level": "<...>"}},
    {{"name": "基本素质", "score": <整数>, "level": "<...>"}},
    {{"name": "逻辑思维", "score": <整数>, "level": "<...>"}},
    {{"name": "形象气质", "score": <整数>, "level": "<...>"}},
    {{"name": "语言描述品质", "score": <整数>, "level": "<...>"}},
    {{"name": "岗位匹配度", "score": <整数>, "level": "<...>"}}
  ],
  "internalDimensions": [
    {{"name": "工作意向", "score": <整数>, "level": "<...>"}},
    {{"name": "个性特质", "score": <整数>, "level": "<...>"}},
    {{"name": "行为能力", "score": <整数>, "level": "<...>"}},
    {{"name": "认知能力", "score": <整数>, "level": "<...>"}}
  ],
  "advantages": {{
    "tags": ["<标签1>", "<标签2>", "<标签3>"],
    "summary": "<150字以内优势综述>"
  }},
  "disadvantages": {{
    "tags": ["<标签1>", "<标签2>"],
    "summary": "<150字以内劣势与改进建议>"
  }},
  "riskMonitoring": {{
    "identityCheck": {{"status": "<正常|待核实|异常>", "detail": "<简述>"}},
    "psychologicalRisk": {{"status": "<低|中|高|待评估>", "detail": "<简述>"}},
    "backgroundCheck": {{"status": "<正常|待核实|异常>", "detail": "<简述>"}},
    "processMonitoring": {{"status": "<正常|待关注|异常>", "detail": "<简述>"}},
    "resumeAlerts": {{"status": "<无异常|待核实|有预警>", "detail": "<简述>"}}
  }},
  "detailSections": [
    {{
      "key": "work_intent",
      "title": "工作意向",
      "score": <整数>,
      "maxScore": 100,
      "summary": "<200字以内分析>",
      "evidence": [{{"label": "关键表述", "content": "<引用或概括>"}}]
    }},
    {{
      "key": "professional",
      "title": "专业能力",
      "score": <整数>,
      "maxScore": 100,
      "summary": "<200字以内分析>",
      "subSkills": [
        {{"name": "<技能/技术点>", "score": <整数>, "analysis": "<80字以内>"}}
      ],
      "evidence": [{{"label": "典型问答", "content": "<引用或概括>"}}]
    }},
    {{
      "key": "behavioral",
      "title": "行为能力+个性潜质",
      "score": <整数>,
      "maxScore": 100,
      "summary": "<200字以内分析>",
      "subSkills": [
        {{"name": "<子维度如影响力>", "score": <整数>, "analysis": "<80字以内>"}}
      ]
    }}
  ]
}}

externalDimensions 固定 7 项、internalDimensions 固定 4 项、detailSections 固定 3 项，顺序不可变。"""


def _parse_json_object(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


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

    return f"""请为以下候选人撰写全方位面试评定报告。

【候选人】
姓名: {candidate.get('name', '')}
学历: {candidate.get('education', '')}
经验: {candidate.get('experience', '')}
技能: {', '.join(candidate.get('skills') or [])}

【岗位】
名称: {position.get('name', '')}
要求: {(position.get('requirements') or '')[:1500]}

【简历 AI 摘要】
综合分: {resume.get('overallScore', '—')}
摘要: {(resume.get('summary') or '')[:800]}
亮点: {resume.get('highlights') or []}
风险: {resume.get('risks') or []}

【问答表现】（共 {len(qa_items)} 题）
{chr(10).join(qa_lines) if qa_lines else '（无抽取问答）'}

【特殊环节】
{chr(10).join(seg_lines) if seg_lines else '（无）'}

【转写摘要】（前 12000 字）
{(context.get('transcriptExcerpt') or '')[:12000]}
"""


async def generate_assessment_report(context: dict, fallback_score: int = 70) -> dict:
    """生成全方位面试评定报告。"""
    from app.services.ai import get_llm_client

    position_name = (context.get("position") or {}).get("name") or "未知岗位"
    round_label = "二面" if context.get("round") == "second" else "一面"
    client = get_llm_client()

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
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
