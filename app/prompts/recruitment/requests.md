## Prompt 1
你是资深 HR 顾问。根据以下招聘需求，生成结构化的岗位说明书。

## 招聘需求
[[base_info]]

## 返回 JSON（只返回 JSON，不要其他内容）
{
  "basicInfo": {
    "positionName": "岗位名称",
    "department": "所属部门",
    "headcount": 1,
    "reportTo": "汇报对象",
    "salaryRange": "薪资范围",
    "workLocation": "工作地点",
    "employmentType": "正式员工",
    "level": "岗位等级"
  },
  "mission": "岗位使命和核心价值（100字以上）",
  "responsibilities": ["主要工作职责1（含量化指标）", "职责2"],
  "qualifications": {
    "required": ["必须具备的条件1", "条件2"],
    "preferred": ["优先条件1", "加分项2"]
  },
  "permissions": ["岗位工作权限1", "权限2"],
  "collaborations": {
    "internal": ["内部协作部门/角色1", "协作2"],
    "external": ["外部协作对象1"]
  },
  "workEnvironment": {
    "officeType": "办公室/远程/混合",
    "workingHours": "工作时间",
    "overtime": "加班情况",
    "travel": "出差要求"
  },
  "kpi": ["核心绩效考核指标1", "指标2", "指标3"],
  "careerPath": "职业发展通道描述"
}

## Prompt 2
你是资深 HR 顾问。根据以下招聘需求，生成岗位能力模型。

## 招聘需求
[[base_info]]

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {"dimension": "专业技能", "weight": 30, "description": "评估标准描述"},
  {"dimension": "项目经验", "weight": 20, "description": "评估标准描述"},
  {"dimension": "任务交付能力", "weight": 15, "description": "评估标准描述"},
  {"dimension": "问题解决能力", "weight": 15, "description": "评估标准描述"},
  {"dimension": "学习能力", "weight": 10, "description": "评估标准描述"},
  {"dimension": "沟通协作与责任感", "weight": 10, "description": "评估标准描述"}
]

## Prompt 3
你是资深 HR 顾问。根据以下招聘需求，生成简历评分规则。

## 招聘需求
[[base_info]]

## 返回 JSON（只返回 JSON，不要其他内容）
{
  "skillMatch": 25,
  "projectMatch": 20,
  "positionExp": 15,
  "achievement": 15,
  "industryExp": 10,
  "learning": 5,
  "stability": 5,
  "bonusSkill": 5
}

## Prompt 4
你是资深 HR 顾问。根据以下招聘需求，生成第一轮面试（初试）的评估维度。

## 招聘需求
[[base_info]]

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {"dimension": "经历真实性", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]},
  {"dimension": "专业基础", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]},
  {"dimension": "项目经验", "weight": 20, "description": "评估要点描述", "sampleQuestions": ["问题1"]},
  {"dimension": "学习能力", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]},
  {"dimension": "沟通表达", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]},
  {"dimension": "稳定性与动机", "weight": 10, "description": "评估要点描述", "sampleQuestions": ["问题1"]}
]

## Prompt 5
你是资深技术面试官。根据以下招聘需求，生成第二轮面试（复试）的评估维度。

## 招聘需求
[[base_info]]

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {"dimension": "系统设计能力", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]},
  {"dimension": "技术深度", "weight": 25, "description": "评估要点描述", "sampleQuestions": ["问题1", "问题2"]},
  {"dimension": "问题解决思路", "weight": 20, "description": "评估要点描述", "sampleQuestions": ["问题1"]},
  {"dimension": "技术视野", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]},
  {"dimension": "领导力/影响力", "weight": 15, "description": "评估要点描述", "sampleQuestions": ["问题1"]}
]

## Prompt 6
你是资深 HR 顾问。根据以下招聘需求，生成试用期考核框架。

## 招聘需求
[[base_info]]
- 试用期目标: [[req_probation_goal]]
- 淘汰条件: [[req_elimination_criteria]]

## 返回 JSON（只返回 JSON，不要其他内容）
{
  "phases": [
    {"name": "第1周", "duration": "1周", "goals": ["熟悉环境和团队", "了解项目背景"], "evaluationMethod": "导师评价"},
    {"name": "第2-4周", "duration": "3周", "goals": ["独立完成小型任务", "熟悉开发流程"], "evaluationMethod": "任务完成度评估"},
    {"name": "第2-3个月", "duration": "2个月", "goals": ["独立负责模块开发", "达到试用期目标"], "evaluationMethod": "转正评审"}
  ]
}

## Prompt 7
你是资深 HR 顾问。根据以下招聘需求，生成入职培训计划。

## 招聘需求
[[base_info]]

## 返回 JSON 数组（只返回 JSON，不要其他内容）
[
  {"title": "公司文化与价值观", "duration": "半天", "content": "企业使命、愿景、核心价值观", "objectives": ["了解公司文化", "认同核心价值观"], "method": "讲座"},
  {"title": "产品与业务介绍", "duration": "1天", "content": "公司产品线、业务模式、客户群体", "objectives": ["理解产品定位", "了解业务流程"], "method": "讲解+演示"},
  {"title": "技术架构概览", "duration": "1天", "content": "技术栈、系统架构、开发规范", "objectives": ["熟悉技术栈", "了解代码规范"], "method": "技术分享"},
  {"title": "开发工具与流程", "duration": "半天", "content": "Git、CI/CD、项目管理工具", "objectives": ["掌握开发工具", "熟悉发布流程"], "method": "实操"},
  {"title": "团队协作规范", "duration": "半天", "content": "代码评审、文档规范、沟通机制", "objectives": ["了解协作流程", "掌握规范要求"], "method": "讲解"}
]
