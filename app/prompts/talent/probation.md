## Prompt 1
作为HR试用期评估专家,请对以下员工进行试用期表现评估。

员工:[[employee_name]]
部门:[[department]]
入职日期:[[onboard_date]]
试用期截止:[[probation_end_date]]

任务数据:[[tasks_json]]

请从以下维度评估并返回JSON:
1. 项目表现(60分满分)
2. 技术能力(20分满分)
3. 团队协作(20分满分)

返回格式:{"projectPerformance": 分数, "techCapability": 分数, "collaboration": 分数, "score": 综合总分, "result": "converted/extended/rejected", "comment": "评估意见"}
只返回JSON。
