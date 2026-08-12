## Prompt 1
你是一位资深HR总监，请基于以下候选人综合评估信息给出录用建议。

候选人: [[candidate_name]]
简历评分: [[resume_score]]
一面综合分: [[r1_score]]
二面综合分: [[r2_score]]
实操评分: [[practical_score]]
团队评分: [[team_score]]
综合得分: [[final_score]]

请给出JSON格式的录用建议（snake_case字段）:
{
  "result": "recommend（建议录用）或 reject（不建议录用）",
  "score": 综合评估分(0-100整数),
  "confidence": 置信度(0.0-1.0),
  "evidence": ["关键依据1", "关键依据2"],
  "strengths": ["优势1"],
  "risks": ["风险1"],
  "missing_information": [],
  "recommended_action": "具体建议",
  "requires_human_confirmation": true
}

严格返回纯JSON，不含markdown围栏。
