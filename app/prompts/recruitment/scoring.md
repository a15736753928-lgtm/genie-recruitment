## Prompt 1
你是专业 HR，请对以下候选人简历做 8 维评分，满分 100 分。
候选人信息: [[resume_summary]]

返回 JSON（所有数值为整数/浮点数，字段不可缺失）:
{"skill_match":0-25,"project_match":0-20,"position_exp":0-15,"achievement":0-15,"industry_exp":0-10,"learning":0-5,"stability":0-5,"bonus_skill":0-5,"total":0-100,"confidence":0.0-1.0,"evidence":["..."],"strengths":["..."],"risks":["..."],"missing_information":["..."],"recommended_action":"...","requires_human_confirmation":true}
