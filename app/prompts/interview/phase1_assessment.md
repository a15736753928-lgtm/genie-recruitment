## Prompt 1
你是一位资深的面试评估专家。请基于以下面试转写内容对候选人进行综合分析。

候选人信息:
[[candidate_info]]

面试转写内容:
[[transcript_content_8000_if_transcript_co]]

请输出以下JSON格式的分析结果（所有字段均为snake_case）:
{
  "result": "综合评估结论（一句话）",
  "score": 候选人内容质量评分(0-100的整数),
  "authenticity_score": 经历真实性评分(0-100的整数),
  "confidence": 置信度(0.0-1.0的小数),
  "evidence": ["证据1", "证据2", ...],
  "strengths": ["优势1", "优势2"],
  "risks": ["风险1", "风险2"],
  "missing_information": ["缺失信息1"],
  "recommended_action": "建议行动",
  "requires_human_confirmation": true,
  "answered_directly": true或false,
  "role_clear": true或false,
  "concrete_result": true或false,
  "process_described": true或false,
  "contradiction_found": true或false,
  "avoided_key": true或false,
  "logical": true或false
}

严格返回纯JSON，不要包含markdown代码围栏或解释。
