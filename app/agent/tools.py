"""Agent Tool Registry — all tools the AI Agent can call."""
from typing import List

from app.agent.tool_desc import tool_desc


TOOL_REGISTRY = {
    # Recruitment tools
    "list_resumes": {
        "name": "list_resumes",
        "description": tool_desc(
            "分页查询候选人列表",
            "按姓名/岗位/状态筛选、列出人选；查某岗位其他人选时用 positionName",
            "查总数(用get_operations_dashboard)、查JD、改状态、出题",
            "「看看Agent工程师岗位的其他候选人」→ list_resumes(positionName=Agent工程师)",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positionId": {"type": "string", "description": "岗位ID，不传表示全部"},
                "positionName": {"type": "string", "description": "岗位名称（如 Agent工程师），会自动解析为 positionId"},
                "statuses": {"type": "string", "description": "状态列表，逗号分隔"},
                "keyword": {"type": "string", "description": "搜索关键词（姓名/技能/岗位名）"},
                "sortBy": {"type": "string", "enum": ["score", "uploadTime"], "description": "排序字段"},
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
        },
    },
    "get_resume": {
        "name": "get_resume",
        "description": tool_desc(
            "读取单个候选人详情",
            "看完整简历/匹配分/技能/经历；view=full 看全量",
            "岗位题库(get_position_questions)、候选人题单(get_questions)、改状态",
            "「查看吴佳熙完整简历」→ get_resume(id, view=full)",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人 UUID（必须是 list_resumes 返回的真实 id）"},
                "view": {
                    "type": "string",
                    "description": "summary|core|detail|contact|screening|file|full，默认 detail",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "精确字段列表，优先于 view",
                },
                "purpose": {"type": "string", "description": "本轮用户意图摘要，用于自动选字段"},
            },
            "required": ["id"],
        },
    },
    "list_positions": {
        "name": "list_positions",
        "description": "获取所有岗位列表（id + 名称 + 部门）。按岗位名查 JD 时：先用本工具找 id，再调 get_position；不要 list_resumes。",
        "parameters": {"type": "object", "properties": {}},
    },
    "update_resume": {
        "name": "update_resume",
        "description": tool_desc(
            "更新候选人状态或字段（写操作）",
            "通过初筛/安排一面/一面未通过/淘汰/改电话等",
            "查详情(get_resume)、出题(generate_questions)、搜知识库(rag_search)",
            "「一面未通过」→ update_resume(id, status=rejected)",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
                "status": {
                    "type": "string",
                    "enum": ["new", "parsed", "pending_screen", "invited", "round1", "round2", "pending_offer", "hired", "talent_pool", "rejected"],
                    "description": "pending_screen=待筛选, invited=初筛通过待安排面试, round1=一面中, round2=二面中, pending_offer=待发Offer, hired=已录用, rejected=未通过/淘汰, talent_pool=人才池",
                },
                "fields": {"type": "object", "description": "要更新的其他字段"},
            },
            "required": ["id"],
        },
    },
    "upload_resume": {
        "name": "upload_resume",
        "description": "上传简历文件并触发AI解析。fileKey 必须是前端已上传到 MinIO 的 object key（通过「添加资料」上传后获得）。",
        "parameters": {
            "type": "object",
            "properties": {
                "fileKey": {"type": "string", "description": "MinIO object key，如 resumes/xxx.pdf"},
                "fileName": {"type": "string", "description": "原始文件名"},
                "positionId": {"type": "string", "description": "应聘岗位ID"},
            },
            "required": ["fileKey", "fileName", "positionId"],
        },
    },
    "batch_parse_resumes": {
        "name": "batch_parse_resumes",
        "description": "批量重新解析多个候选人的简历（AI 重新提取结构化信息）。",
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}, "description": "候选人ID列表"},
            },
            "required": ["ids"],
        },
    },
    "reanalyze_resume": {
        "name": "reanalyze_resume",
        "description": "对单个候选人重新进行 AI 简历解析与评分。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
            },
            "required": ["id"],
        },
    },
    "delete_resume": {
        "name": "delete_resume",
        "description": "删除候选人及其简历文件。不可逆操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
            },
            "required": ["id"],
        },
    },

    # Position tools (write)
    "get_position": {
        "name": "get_position",
        "description": (
            "获取岗位信息。按意图选字段，默认 core（含学历/经验/年龄/薪资独立字段 + JD 正文）。"
            "view: summary|core|jd|requirements|edit|criteria|probation_plan|full；"
            "或 fields 指定字段。改学历必须先看到 educationRequirement 字段再 update_position。"
            "用户说「查看XX岗位JD」时用此工具；不要同时查候选人。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "岗位 UUID（来自 list_positions）"},
                "view": {
                    "type": "string",
                    "description": "summary|core|jd|requirements|edit|criteria|probation_plan|full",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "精确字段列表（如 [\"educationRequirement\",\"salaryRange\"]），优先于 view",
                },
                "purpose": {"type": "string", "description": "本轮用户意图摘要，用于自动选字段"},
            },
            "required": ["id"],
        },
    },
    "create_position": {
        "name": "create_position",
        "description": "创建新招聘岗位，含 JD、部门等。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "岗位名称"},
                "department": {"type": "string", "description": "部门"},
                "jdResponsibilities": {"type": "string", "description": "岗位职责"},
                "jdRequirements": {"type": "string", "description": "任职要求"},
                "jdPreferred": {"type": "string", "description": "加分项"},
                "jdTechStack": {"type": "string", "description": "技术栈"},
            },
            "required": ["name"],
        },
    },
    "update_position": {
        "name": "update_position",
        "description": (
            "更新岗位信息。重要：学历/经验/年龄/薪资有专门的独立字段，"
            "修改这些属性时必须改对应字段，禁止把它们当文字写进 jdRequirements 正文！\n"
            "- educationRequirement: 学历要求（如「本科」「硕士」「博士」「不限」）\n"
            "- experienceRequirement: 经验要求（如「3-5年」「应届」）\n"
            "- ageRequirement: 年龄要求（如「25-35岁」）\n"
            "- salaryRange: 薪资范围（如「10-20K」）\n"
            "其余字段：name/department/jdResponsibilities/jdRequirements(职责/要求正文)/"
            "jdPreferred/jdTechStack/screeningCriteria/interviewCriteriaR1/interviewCriteriaR2 等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "岗位ID"},
                "fields": {"type": "object", "description": (
                    "要更新的字段。改学历用 educationRequirement，改经验用 experienceRequirement，"
                    "改年龄用 ageRequirement，改薪资用 salaryRange——不要写进 jdRequirements。"
                )},
            },
            "required": ["id", "fields"],
        },
    },
    "delete_position": {
        "name": "delete_position",
        "description": "删除岗位。若岗位下有候选人会拒绝删除。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "岗位ID"},
            },
            "required": ["id"],
        },
    },
    "get_position_questions": {
        "name": "get_position_questions",
        "description": tool_desc(
            "读取岗位通用题库（非某个候选人的题）",
            "用户明确要看「岗位题库/岗位标准题」",
            "查候选人简历(get_resume)、查某人题单(get_questions)、完整简历",
            "参数 positionId+round；勿在用户要看简历时调用",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positionId": {"type": "string"},
                "round": {"type": "string", "enum": ["first", "second"], "description": "一面/二面"},
            },
            "required": ["positionId", "round"],
        },
    },
    "save_position_questions": {
        "name": "save_position_questions",
        "description": "保存岗位题库题目（覆盖该轮次原有题目）。",
        "parameters": {
            "type": "object",
            "properties": {
                "positionId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
                "questions": {"type": "array", "items": {"type": "object"}, "description": "题目数组，每项含 index/content/category/difficulty"},
            },
            "required": ["positionId", "round", "questions"],
        },
    },

    # Interview tools
    "get_questions": {
        "name": "get_questions",
        "description": tool_desc(
            "读取候选人预生成面试题（出题环节）",
            "查看/展示某候选人的题单；参数 candidateId+round",
            "岗位通用题库(get_position_questions)、查简历(get_resume)、改状态",
            "「看吴佳熙一面题目」→ get_questions(candidateId, round=一面)",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
            },
            "required": ["candidateId", "round"],
        },
    },
    "generate_questions": {
        "name": "generate_questions",
        "description": tool_desc(
            "重新生成候选人面试题（覆盖旧题）",
            "用户明确要求「生成/换一批/重新出题」",
            "改状态、面试评定、仅查看题目(get_questions)",
            "「给张三重新出一面题」",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
            },
            "required": ["candidateId", "round"],
        },
    },
    "get_evaluation": {
        "name": "get_evaluation",
        "description": tool_desc(
            "读取面试评定结果（从上传录音/转写抽取）",
            "开始面试评定、看评分/问答/报告",
            "预生成面试题(get_questions/generate_questions)、改状态",
            "「开始一面评定」→ get_evaluation",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "first/second 或 一面/二面，默认 first",
                    "enum": ["first", "second", "一面", "二面"],
                },
                "transcriptId": {"type": "string", "description": "可选，指定某次上传记录 ID"},
            },
            "required": ["candidateId"],
        },
    },
    "ai_score_question": {
        "name": "ai_score_question",
        "description": "使用AI对面试回答进行评分。",
        "parameters": {
            "type": "object",
            "properties": {
                "questionId": {"type": "string"},
                "answer": {"type": "string", "description": "候选人的回答"},
            },
            "required": ["questionId"],
        },
    },
    "get_leaderboard": {
        "name": "get_leaderboard",
        "description": "获取一面/二面排行榜。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["first_result", "second_result"]},
                "limit": {"type": "integer", "description": "返回名次数量，默认20"},
            },
            "required": ["category"],
        },
    },
    "save_questions": {
        "name": "save_questions",
        "description": "保存候选人某轮次的面试题目（覆盖原有题目）。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
                "questions": {"type": "array", "items": {"type": "object"}, "description": "题目数组，每项含 index/content/category/difficulty"},
            },
            "required": ["candidateId", "round", "questions"],
        },
    },
    "replace_question": {
        "name": "replace_question",
        "description": "替换候选人面试题单中的某一道题（AI 重新生成一道）。",
        "parameters": {
            "type": "object",
            "properties": {
                "questionId": {"type": "string", "description": "要替换的题目ID"},
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
                "prompt": {"type": "string", "description": "额外要求（可选）"},
                "category": {"type": "string", "description": "指定分类（可选）"},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"], "description": "指定难度（可选）"},
            },
            "required": ["questionId", "candidateId", "round"],
        },
    },
    "save_evaluation": {
        "name": "save_evaluation",
        "description": "保存 HR 对候选人面试的回答评分（每题含 hrScore/answer/dimensions）。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first/second 或 一面/二面",
                    "enum": ["first", "second", "一面", "二面"],
                },
                "scores": {"type": "array", "items": {"type": "object"}, "description": "每项含 questionId/hrScore/answer/dimensions"},
            },
            "required": ["candidateId", "round", "scores"],
        },
    },
    "submit_evaluation": {
        "name": "submit_evaluation",
        "description": (
            "提交候选人某一轮面试评定（把已评分题目置为 scored）。"
            "⚠️ 有副作用：会按加权总分（Q&A 80% + 自我介绍 10% + 反问 10%）与合格分数线"
            "**自动推进或淘汰候选人**——一面通过→round2，二面通过→pending_offer，"
            "未达线→rejected（淘汰）。属于破坏性操作，执行前必须先向用户说明并取得确认。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {
                    "type": "string",
                    "description": "面试轮次：first=一面 / second=二面，默认 first。传错轮次会推进到错误状态。",
                    "enum": ["first", "second"],
                },
            },
            "required": ["candidateId"],
        },
    },
    "get_rankings": {
        "name": "get_rankings",
        "description": "获取同岗位候选人排名（按简历匹配分排序）。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string", "description": "当前候选人ID"},
            },
            "required": ["candidateId"],
        },
    },

    # Probation/Performance
    "list_probation": {
        "name": "list_probation",
        "description": "查询试用期员工列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "all", "pending_onboard", "training", "probation",
                        "pending_confirmation", "formal", "transferred", "resigned",
                    ],
                    "description": (
                        "员工状态：pending_onboard=待入职, training=培训中, probation=试用期中, "
                        "pending_confirmation=待转正审批, formal=已转正, transferred=已调岗, "
                        "resigned=已离职；all=不筛选"
                    ),
                },
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
        },
    },
    "get_probation_stats": {
        "name": "get_probation_stats",
        "description": "获取试用期员工统计（总数/考核中/通过/未通过）。",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_probation_employee": {
        "name": "get_probation_employee",
        "description": (
            "获取试用期员工信息。按意图选字段，默认 core（入职/试用截止日、导师、进度、AI 结论）。"
            "view: summary|core|tasks|full；或 fields 指定字段。"
            "周考细节看 tasks（任务自带 weekNumber），转正结论看 core 的 aiScore/aiResult/overallScore。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "员工ID"},
                "view": {
                    "type": "string",
                    "enum": ["summary", "core", "tasks", "full"],
                    "description": "summary|core|tasks|full",
                },
                "fields": {"type": "array", "items": {"type": "string"}, "description": "精确字段列表"},
                "purpose": {"type": "string", "description": "本轮用户意图摘要"},
            },
            "required": ["id"],
        },
    },
    "create_probation_employee": {
        "name": "create_probation_employee",
        "description": "新增试用期员工（可由候选人转化或直接录入）。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "candidateId": {"type": "string", "description": "关联候选人ID（可选）"},
                "positionId": {"type": "string"},
                "gender": {"type": "string"},
                "age": {"type": "integer"},
                "department": {"type": "string"},
                "joinDate": {"type": "string", "description": "入职日期 YYYY-MM-DD"},
                "probationEnd": {"type": "string", "description": "试用期截止日期 YYYY-MM-DD"},
                "mentorName": {"type": "string", "description": "导师姓名"},
            },
            "required": ["name"],
        },
    },
    "create_probation_task": {
        "name": "create_probation_task",
        "description": "为试用期员工新增任务。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "weekNumber": {"type": "integer", "description": "第几周"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "deadline": {"type": "string", "description": "截止日期 YYYY-MM-DD（可选）"},
            },
            "required": ["employeeId", "weekNumber", "title"],
        },
    },
    "update_probation_task": {
        "name": "update_probation_task",
        "description": (
            "更新试用期任务（状态/标题/描述/评审意见）。"
            "status 合法值只有：draft / pending_confirm / in_progress / pending_review / "
            "passed / rework / closed —— 完成用 passed，**不要用 completed/done/finished**，"
            "非法值或非法迁移会被状态机拒绝（409）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string"},
                "fields": {"type": "object", "description": (
                    "title/description/reviewNotes/objective/deliverables 等；"
                    "status 只能取 draft/pending_confirm/in_progress/pending_review/passed/rework/closed"
                )},
            },
            "required": ["taskId", "fields"],
        },
    },
    "ai_evaluate_probation": {
        "name": "ai_evaluate_probation",
        "description": "AI 自动评估试用期员工表现（基于任务完成情况给出项目/技术/协作分数与转正建议）。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
            },
            "required": ["employeeId"],
        },
    },
    "update_probation_status": {
        "name": "update_probation_status",
        "description": "手动更新试用期员工状态。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "pending_onboard", "training", "probation",
                        "pending_confirmation", "formal", "transferred", "resigned",
                    ],
                    "description": (
                        "目标状态：pending_onboard=待入职, training=培训中, probation=试用期中, "
                        "pending_confirmation=待转正审批, formal=已转正, transferred=已调岗, "
                        "resigned=已离职。非法迁移会被状态机拒绝（409）。"
                    ),
                },
            },
            "required": ["employeeId", "status"],
        },
    },
    "manual_review_probation": {
        "name": "manual_review_probation",
        "description": "手动录入试用期 AI 评估分数与结论（覆盖 AI 自动评估结果）。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "aiScore": {"type": "number"},
                "aiResult": {"type": "string", "description": "converted/extended/rejected"},
            },
            "required": ["employeeId"],
        },
    },

    # Performance tools
    "list_performance": {
        "name": "list_performance",
        "description": "查询绩效数据。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "description": "如：2026-Q3"},
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
            "required": ["quarter"],
        },
    },
    "get_performance_stats": {
        "name": "get_performance_stats",
        "description": "获取某季度绩效统计（参与人数/平均分/优秀数/待改进数）。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "description": "如 2026-Q3"},
            },
            "required": ["quarter"],
        },
    },
    "get_department_performance": {
        "name": "get_department_performance",
        "description": "获取某季度各部门平均绩效得分。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string"},
            },
            "required": ["quarter"],
        },
    },
    "get_grade_distribution": {
        "name": "get_grade_distribution",
        "description": "获取某季度绩效等级分布（S/A/B+/B/C）。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string"},
            },
            "required": ["quarter"],
        },
    },
    "get_bonus_info": {
        "name": "get_bonus_info",
        "description": "获取某季度奖金池信息（总额/已分配/待分配）。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string"},
            },
            "required": ["quarter"],
        },
    },
    "get_quarter_trends": {
        "name": "get_quarter_trends",
        "description": "获取最近4个季度绩效趋势。",
        "parameters": {"type": "object", "properties": {}},
    },
    "initiate_appraisal": {
        "name": "initiate_appraisal",
        "description": "为指定季度发起绩效考核，批量创建员工绩效记录。",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "description": "如 2026-Q3"},
                "employeeIds": {"type": "array", "items": {"type": "string"}, "description": "参与考核的员工ID列表"},
            },
            "required": ["quarter", "employeeIds"],
        },
    },
    "update_bonus": {
        "name": "update_bonus",
        "description": "更新员工某季度的奖金数额。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "bonus": {"type": "number", "description": "奖金金额"},
            },
            "required": ["employeeId", "bonus"],
        },
    },

    # Knowledge / RAG
    "rag_search": {
        "name": "rag_search",
        "description": tool_desc(
            "语义检索知识库",
            "查制度/规范/技术文档/公司政策",
            "改候选人状态、查简历列表、统计人数",
            "「公司加班制度是什么」",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询文本"},
                "topK": {"type": "integer", "description": "返回条数，默认5"},
                "categoryKey": {"type": "string", "description": "分类过滤"},
            },
            "required": ["query"],
        },
    },
    "list_knowledge": {
        "name": "list_knowledge",
        "description": "查询知识库素材列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "categoryKey": {"type": "string"},
                "keyword": {"type": "string"},
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
        },
    },
    "get_knowledge_stats": {
        "name": "get_knowledge_stats",
        "description": "获取知识库素材统计（总数/本月新增）。",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_knowledge_categories": {
        "name": "get_knowledge_categories",
        "description": "获取知识库分类树。",
        "parameters": {"type": "object", "properties": {}},
    },
    "upload_knowledge_file": {
        "name": "upload_knowledge_file",
        "description": (
            "确认知识库素材文件已就绪（本工具**不做任何上传**：文件在用户点「添加资料」时"
            "就已经进了 MinIO，这里只是回显 object key 并提示进入下一步）。"
            "调用后必须接着调 create_knowledge_item 才算真正入库。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fileKey": {"type": "string", "description": "MinIO object key（如 knowledge/xxx.pdf）"},
                "fileName": {"type": "string", "description": "原始文件名"},
            },
            "required": ["fileKey", "fileName"],
        },
    },
    "create_knowledge_item": {
        "name": "create_knowledge_item",
        "description": "创建知识库素材记录（可关联已上传文件，自动入库向量化）。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "素材名称"},
                "category": {"type": "string", "description": "分类key"},
                "categoryPath": {"type": "string", "description": "分类路径"},
                "type": {"type": "string", "description": "类型：interview/policy/tech/other"},
                "fileId": {"type": "string", "description": "已上传文件的 object key（可选）"},
            },
            "required": ["name"],
        },
    },
    "update_knowledge_item": {
        "name": "update_knowledge_item",
        "description": "更新知识库素材（名称/分类/类型）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "fields": {"type": "object", "description": "name/category/categoryPath/type"},
            },
            "required": ["id", "fields"],
        },
    },
    "delete_knowledge_item": {
        "name": "delete_knowledge_item",
        "description": "删除知识库素材（同时删除向量和文件）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    "recall_test": {
        "name": "recall_test",
        "description": "知识库召回测试（检索并返回命中片段+来源）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },

    # RAG advanced (knowledge bases & documents)
    "list_knowledge_bases": {
        "name": "list_knowledge_bases",
        "description": "列出 RAG 知识库（高级向量库）。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
            },
        },
    },
    "create_knowledge_base": {
        "name": "create_knowledge_base",
        "description": "创建 RAG 知识库。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    "update_knowledge_base": {
        "name": "update_knowledge_base",
        "description": "更新 RAG 知识库名称/描述。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "fields": {"type": "object", "description": "name/description"},
            },
            "required": ["id", "fields"],
        },
    },
    "delete_knowledge_base": {
        "name": "delete_knowledge_base",
        "description": "删除 RAG 知识库及其所有文档和向量。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    "upload_document": {
        "name": "upload_document",
        "description": "向 RAG 知识库上传文档并启动异步入库（向量化）。fileKey 由前端「添加资料」上传后获得。",
        "parameters": {
            "type": "object",
            "properties": {
                "kbId": {"type": "string", "description": "知识库ID"},
                "fileKey": {"type": "string", "description": "MinIO object key"},
                "fileName": {"type": "string", "description": "原始文件名"},
            },
            "required": ["kbId", "fileKey", "fileName"],
        },
    },
    "list_documents": {
        "name": "list_documents",
        "description": "列出 RAG 知识库下的文档。",
        "parameters": {
            "type": "object",
            "properties": {
                "kbId": {"type": "string"},
                "status": {"type": "string", "description": "按入库状态过滤，留空表示不过滤"},
                "limit": {"type": "integer", "description": "返回数量限制，默认20"},
            },
        },
    },
    "delete_document": {
        "name": "delete_document",
        "description": "删除 RAG 文档及其向量分块。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "文档ID"},
            },
            "required": ["id"],
        },
    },

    # Dashboard
    "get_operations_dashboard": {
        "name": "get_operations_dashboard",
        "description": tool_desc(
            "招聘运营统计汇总",
            "有多少候选人/简历、状态分布、平均匹配分",
            "查单个简历详情、列出每人人选",
            "「现在有多少候选人」",
        ),
        "parameters": {"type": "object", "properties": {}},
    },

    # System
    "get_settings": {
        "name": "get_settings",
        "description": (
            "获取系统设置。按意图选字段，默认 core。"
            "view: summary|core|scoring|ai|knowledge|notify|full；或 fields 指定 key。"
            "修改前先 get 再 update_settings，key 必须与返回的 [字段: xxx] 一致。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["summary", "core", "scoring", "ai", "knowledge", "notify", "full"],
                    "description": "summary|core|scoring|ai|knowledge|notify|full",
                },
                "fields": {"type": "array", "items": {"type": "string"}, "description": "精确设置 key 列表"},
                "purpose": {"type": "string", "description": "本轮用户意图摘要"},
            },
        },
    },
    "update_settings": {
        "name": "update_settings",
        "description": "更新系统设置。fields 的 key 必须来自 get_settings 返回的字段名（如 minMatchScore/passScoreThreshold/probationDays）。",
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {"type": "object", "description": "要更新的设置字段 key→value"},
            },
            "required": ["fields"],
        },
    },
}

# Direct-database tools were removed: raw SQL writes bypass business logic
# (status transitions, downstream list updates) and leave the system in an
# inconsistent state. Ad-hoc needs should get a dedicated, curated tool.
GENERAL_TOOL_NAMES = tuple(TOOL_REGISTRY.keys())


# ── 工具→权限点映射（对话工具层权限隔离）─────────────────────
# 价值：get_tools_for_agent 据此过滤"模型可见的工具集"（第一道），
#       tool_executor.execute_tool_call 据此在 dispatch 前二次校验（第二道，防御纵深）。
# 规则：
#   - 写/敏感工具映射到 app/core/permissions.py 已有权限点；
#   - 涉及业务数据查看的敏感读工具（简历/绩效/试用期/系统设置）也挂权限；
#   - 公共只读工具（list_positions/get_position/rag_search/知识库读/运营看板等）
#     显式登记为 PUBLIC_TOOL_PERMISSION，所有登录用户可见。
#   - admin 凭 system:manage 通配放行（CurrentUser.has() 内含 WILDCARD）。
# 规则（fail-closed）：所有工具必须在此登记；未登记的工具默认不可见/不可调，
# 避免新增工具忘登记就自动变成全员可用。
# 同步新增工具时：写/敏感/涉业务数据的一律登记对应权限点；纯公共只读登记
# PUBLIC_TOOL_PERMISSION。
PUBLIC_TOOL_PERMISSION = "__public__"

TOOL_PERMISSIONS: dict[str, str] = {
    # ── 简历（读+写，ceo/hr/interviewer 可见；manager 无 resume:view）──
    "list_resumes": "resume:view",
    "get_resume": "resume:view",
    "update_resume": "resume:decide",
    "upload_resume": "resume:decide",
    "batch_parse_resumes": "resume:decide",
    "reanalyze_resume": "resume:decide",
    "delete_resume": "resume:decide",
    "get_rankings": "resume:view",
    # ── 岗位增删改（读公开：list_positions/get_position 已登记 PUBLIC）──
    "create_position": "position:manage",
    "update_position": "position:manage",
    "delete_position": "position:manage",
    # ── 面试出题/评分 ──
    "save_position_questions": "interview:manage",
    "generate_questions": "interview:manage",
    "save_questions": "interview:manage",
    "replace_question": "interview:manage",
    "get_position_questions": "interview:score",
    "get_questions": "interview:score",
    "get_evaluation": "interview:score",
    "ai_score_question": "interview:score",
    "save_evaluation": "interview:score",
    "submit_evaluation": "interview:score",
    "get_leaderboard": "interview:score",
    # ── 试用期（管理类；employee 有 probation:submit 但无对应自助工具）──
    "list_probation": "probation:manage",
    "get_probation_stats": "probation:manage",
    "get_probation_employee": "probation:manage",
    "create_probation_employee": "probation:manage",
    "create_probation_task": "probation:manage",
    "update_probation_task": "probation:manage",
    "ai_evaluate_probation": "probation:manage",
    "update_probation_status": "probation:manage",
    "manual_review_probation": "probation:manage",
    # ── 绩效（敏感数据，talent:view / salary:view）──
    "list_performance": "talent:view",
    "get_performance_stats": "talent:view",
    "get_department_performance": "talent:view",
    "get_grade_distribution": "talent:view",
    "get_quarter_trends": "talent:view",
    "initiate_appraisal": "talent:manage",
    "get_bonus_info": "salary:view",
    "update_bonus": "salary:manage",
    # ── 知识库写管理（admin 专属；读 list_knowledge/rag_search 等未列入=公共）──
    "upload_knowledge_file": "system:manage",
    "create_knowledge_item": "system:manage",
    "update_knowledge_item": "system:manage",
    "delete_knowledge_item": "system:manage",
    "create_knowledge_base": "system:manage",
    "update_knowledge_base": "system:manage",
    "delete_knowledge_base": "system:manage",
    "upload_document": "system:manage",
    "delete_document": "system:manage",
    # ── 系统设置 ──
    "get_settings": "system:manage",
    "update_settings": "system:manage",
    # ── 系统/调试 ──
    "recall_test": "system:manage",
    # ── 公共只读工具（显式标记，所有登录用户可见）──
    "list_positions": PUBLIC_TOOL_PERMISSION,
    "get_position": PUBLIC_TOOL_PERMISSION,
    "list_knowledge": PUBLIC_TOOL_PERMISSION,
    "list_knowledge_bases": PUBLIC_TOOL_PERMISSION,
    "list_documents": PUBLIC_TOOL_PERMISSION,
    "get_knowledge_categories": PUBLIC_TOOL_PERMISSION,
    "get_knowledge_stats": PUBLIC_TOOL_PERMISSION,
    "rag_search": PUBLIC_TOOL_PERMISSION,
    "get_operations_dashboard": PUBLIC_TOOL_PERMISSION,
}


def get_tools_for_agent(agent_id: str = "genie", current_user=None) -> List[dict]:
    """Get the list of tool definitions for a specific agent type.

    系统只剩一个全能 agent（"genie"），它拿到全部工具。旧的 4 个分组
    （recruit/interview/training/performance）连同它们过时的工具清单已删除
    —— 那份清单早已与 TOOL_REGISTRY 脱节，且没有任何调用方还在传这些 id。
    任何未知 agent_id 一律退回全量工具（与 genie 相同），保证不会因为
    传错 id 就把工具集悄悄裁掉。

    权限隔离：传入 current_user 时，按其 permissions 过滤掉无权调用的工具
    （admin 凭 system:manage 通配放行，has() 内含 WILDCARD）。未传时退回全量，
    仅用于 quality_guard 的只读验证路径与内部调用——这些路径不应被权限收窄影响。
    """
    tool_defs = [TOOL_REGISTRY[name] for name in GENERAL_TOOL_NAMES]
    if current_user is None:
        return tool_defs
    # fail-closed：未登记权限的工具默认不可见；显式公共只读（PUBLIC_TOOL_PERMISSION）
    # 或持有对应权限点的工具才对用户可见。
    visible = []
    for td in tool_defs:
        perm = TOOL_PERMISSIONS.get(td["name"])
        if perm is None:
            continue
        if perm == PUBLIC_TOOL_PERMISSION or current_user.has(perm):
            visible.append(td)
    return visible


# ── LangChain / LangGraph Tool Conversion ────────────────

import inspect
from typing import get_type_hints
from pydantic import create_model, Field
from app.database import async_session_factory


_JSON_TYPE_MAP = {
    "string": (str, None),
    "integer": (int, None),
    "number": (float, None),
    "boolean": (bool, None),
    "object": (dict, None),
    "array": (list, None),
}


def _build_pydantic_model(tool_name: str, params_schema: dict) -> type:
    """Convert a JSON Schema parameters dict into a dynamic Pydantic model."""
    fields = {}
    properties = params_schema.get("properties", {})
    required = params_schema.get("required", [])

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        python_type, _ = _JSON_TYPE_MAP.get(json_type, (str, None))
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else None
        fields[prop_name] = (python_type, Field(default, description=description))

    model_name = f"{tool_name}_params"
    # If no fields, create a model with no required fields (empty input)
    if not fields:
        fields["dummy"] = (str | None, Field(None, description="No parameters needed"))

    return create_model(model_name, **fields)


async def _execute_tool_sync(tool_name: str, *, current_user=None, **kwargs) -> str:
    """Execute a tool with an auto-created DB session (for LangGraph context).

    Uses lazy import to avoid circular dependency with app.routers.ai_agent.

    IMPORTANT: The session cleanup is shielded from ``asyncio.CancelledError``
    so that when the SSE streaming client disconnects mid-tool-call, the
    asyncpg connection is always returned to the pool instead of being left
    dangling for the garbage collector.

    current_user：由 tool_func 闭包透传。在此 set 到 contextvar，供 handler 内
    get_current_user_for_tools() 做写操作二次权限校验（防御纵深）。即便可见集
    过滤被绕过（新增工具忘挂 require_permission），关键写工具仍能挡住。
    走 quality_guard 验证路径时不传 current_user → contextvar 为 None → 校验跳过。
    """
    from app.api.ai.tool_executor import (
        execute_tool_call,
        set_current_user_for_tools,
        _current_user_ctx,
    )  # lazy import
    import asyncio as _asyncio

    # Remove None values (unset optional params)
    params = {k: v for k, v in kwargs.items() if v is not None}
    # Remove the dummy field if present
    params.pop("dummy", None)

    # Destructive-operation confirmation is handled conversationally via the
    # system prompt (rules.txt) — the model confirms with the user before
    # calling a delete_* tool. No runtime permission engine.

    db = async_session_factory()
    # 注入当前用户到 contextvar（shield 调用前 set，同 task 的 await 链路可见）
    _ctx_token = set_current_user_for_tools(current_user)
    try:
        result = await _asyncio.shield(execute_tool_call(tool_name, params, db))
        await db.commit()
        return result
    except BaseException:
        await db.rollback()
        raise
    finally:
        # 还原 contextvar，避免跨请求串号
        _current_user_ctx.reset(_ctx_token)
        # Shield close() so the connection ALWAYS goes back to the pool,
        # even when CancelledError fires during the finally block itself.
        await _asyncio.shield(db.close())


def create_langchain_tools(agent_id: str = "genie", current_user=None) -> list:
    """Convert the tool registry into LangChain StructuredTool objects.

    Returns a list of tools compatible with LangGraph's ToolNode.
    current_user 用于按权限过滤可见工具集 + 注入 contextvar 供 handler 二次校验。
    """
    from langchain_core.tools import StructuredTool

    tool_defs = get_tools_for_agent(agent_id, current_user)
    lc_tools = []

    for td in tool_defs:
        tool_name = td["name"]
        description = td["description"]
        params_schema = td.get("parameters", {})

        # Build Pydantic args model from JSON Schema
        args_model = _build_pydantic_model(tool_name, params_schema)

        # Create the async tool function bound to this tool_name.
        # current_user 作为默认参数捕获进闭包，透传给 _execute_tool_sync，
        # 后者 set 到 contextvar 供 handler 内 get_current_user_for_tools() 读取。
        async def tool_func(tool_name=tool_name, current_user=current_user, **kwargs) -> str:
            return await _execute_tool_sync(tool_name, current_user=current_user, **kwargs)

        # Attach proper signature for LangChain
        tool_func.__name__ = tool_name
        tool_func.__doc__ = description

        structured_tool = StructuredTool.from_function(
            name=tool_name,
            description=description,
            args_schema=args_model,
            coroutine=tool_func,
        )
        lc_tools.append(structured_tool)

    return lc_tools


