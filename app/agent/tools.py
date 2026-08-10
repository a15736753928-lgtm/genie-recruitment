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
            "「查看丁洁简历」→ get_resume(id)（返回含 resumeFileUrl，回复中自动附「查看原版简历」链接）；「看完整简历」→ view=full",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人 UUID（必须是 list_resumes 返回的真实 id）"},
                "view": {
                    "type": "string",
                    "description": "summary|core|detail|contact|screening|file|full，默认 detail。只要查看候选人简历，返回都会带 resumeFileUrl（用于在回复中给「查看原版简历」可点击链接）；只看文件信息可用 file",
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
            "查详情(get_resume)、出题(generate_questions)",
            "「一面未通过」→ update_resume(id, status=rejected)",
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
                "status": {
                    "type": "string",
                    "enum": ["job_hunting", "round1", "round2", "pending_offer", "rejected", "talent_pool"],
                    "description": "求职中=job_hunting, 一面中=round1, 二面中=round2, 待发Offer=pending_offer, 未通过=rejected, 已失效(入人才池)=talent_pool。注意：hired 不能由 AI 直接设置（仅 Offer 审批通过自动生成）；round1/round2/pending_offer/rejected/talent_pool 均可从求职中迁入",
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
            "生成/重新生成候选人面试题（覆盖旧题）",
            "用户要求「出题/生成面试题/换一批/重新出题」；候选人状态是否满足由系统判断，直接调用本工具即可",
            "仅查看题目(get_questions)、面试评定、改状态；候选人状态不满足出题条件时，系统会自动弹出确认卡征求用户决策——不要自行调 update_resume 改状态、不要用自然语言向用户询问「是否出题/是否先改状态」，直接调用本工具，收到「[系统] 用户确认」指令后按指示执行",
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

    # ── 知识库/RAG 工具已断开（2026-08-01）────────────────────────────
    # rag_search / list_knowledge / create_knowledge_item / recall_test /
    # list_knowledge_bases / upload_document 等 16 个知识库工具的注册已移除。
    # 定义仍保留在 git 历史与本文件旧版本；如后期决定恢复，从 git 历史
    # 还原 TOOL_REGISTRY 对应键 + TOOL_PERMISSIONS 登记 + graph.py 权重 +
    # tool_handlers/knowledge.py 注册即可。

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
    # ── 知识库/RAG 工具的权限登记已随工具摘除（2026-08-01）──
    # ── 系统设置 ──
    "get_settings": "system:manage",
    "update_settings": "system:manage",
    # ── 公共只读工具（显式标记，所有登录用户可见）──
    "list_positions": PUBLIC_TOOL_PERMISSION,
    "get_position": PUBLIC_TOOL_PERMISSION,
    "get_operations_dashboard": PUBLIC_TOOL_PERMISSION,
}


# ── 工具分组 → 能力描述（AI 能力清单的唯一真源）─────────────────
# 价值：build_capability_lines 据此把「当前用户可见的工具集」渲染成 AI 能正确
#       回答的能力清单——AI 宣称的每项能力都有对应工具支撑，不再像旧方案那样
#       按角色权限点宣称一堆对话里根本不存在的模块功能（录用/期权/培训/任务等）。
# 规则：
#   - 每个登录用户可见的工具必须且只能归入一个能力段；段名即 AI 对外宣称的能力。
#   - 同组工具的 TOOL_PERMISSIONS 权限点必须一致（_validate_capability_groups
#     启动时强制校验），避免命中低权限读工具却宣称该组的写能力（如只有 salary:view
#     的用户不能因此宣称"可调整薪资"）。
#   - 新增工具：在 TOOL_PERMISSIONS 登记权限的同时，必须把工具名归入对应能力段，
#     否则启动即报错（fail-closed，杜绝清单漂移）。
_CAPABILITY_GROUPS: tuple[tuple[str, frozenset[str], str], ...] = (
    # (能力段名, 组内工具名集合, 能力描述)
    ("岗位查询", frozenset({"list_positions", "get_position"}),
     "浏览与查询招聘岗位"),
    ("运营看板", frozenset({"get_operations_dashboard"}),
     "查看招聘运营统计汇总（候选人/简历数量、状态分布、平均匹配分）"),
    ("简历查看", frozenset({"list_resumes", "get_resume", "get_rankings"}),
     "查询/查看候选人简历与匹配分，按岗位/状态/关键词找人，查看同岗位排名"),
    ("简历处置", frozenset({"update_resume", "upload_resume", "batch_parse_resumes",
                            "reanalyze_resume", "delete_resume"}),
     "上传、删除、重新解析简历，并按初筛结果更新候选人状态"),
    ("岗位管理", frozenset({"create_position", "update_position", "delete_position"}),
     "创建/更新/删除招聘岗位，管理 JD 与岗位题库"),
    ("面试出题", frozenset({"save_position_questions", "generate_questions",
                            "save_questions", "replace_question"}),
     "生成、保存、替换一面二面面试题目（出题会覆盖旧题）"),
    ("面试评定", frozenset({"get_position_questions", "get_questions", "get_evaluation",
                            "ai_score_question", "save_evaluation", "submit_evaluation",
                            "get_leaderboard"}),
     "查看候选人题单与岗位题库，录入候选人回答、AI 评分，查看排行榜并提交面试结论"),
    ("试用期管理", frozenset({"list_probation", "get_probation_stats", "get_probation_employee",
                              "create_probation_employee", "create_probation_task",
                              "update_probation_task", "ai_evaluate_probation",
                              "update_probation_status", "manual_review_probation"}),
     "查看试用期员工与周任务，新增/跟踪周任务，AI 自动或手动评估"),
    ("绩效查看", frozenset({"list_performance", "get_performance_stats",
                            "get_department_performance", "get_grade_distribution",
                            "get_quarter_trends"}),
     "查看绩效统计、部门绩效、等级分布与季度趋势"),
    ("发起考核", frozenset({"initiate_appraisal"}),
     "发起季度绩效考核"),
    ("薪资查看", frozenset({"get_bonus_info"}),
     "查看员工薪资与奖金"),
    ("薪资调整", frozenset({"update_bonus"}),
     "调整员工薪资与奖金"),
    ("系统管理", frozenset({"get_settings", "update_settings"}),
     "查看与修改系统配置"),
)


def _validate_capability_groups() -> None:
    """启动校验：能力分组必须覆盖 TOOL_PERMISSIONS 全部工具，且同组权限一致。

    fail-closed：新增工具忘记归组、或误把不同权限的工具塞进同一能力段，
    都会在启动时直接报错，杜绝能力清单与可见工具集漂移。
    """
    covered: set[str] = set()
    for label, names, _desc in _CAPABILITY_GROUPS:
        unknown = names - set(GENERAL_TOOL_NAMES)
        if unknown:
            raise ValueError(
                f"能力分组[{label}]引用了未注册工具: {sorted(unknown)}"
            )
        perms = {TOOL_PERMISSIONS.get(n) for n in names}
        if len(perms) > 1:
            raise ValueError(
                f"能力分组[{label}]内工具权限不一致: {sorted(perms)}"
            )
        covered |= names
    ungrouped = set(TOOL_PERMISSIONS) - covered
    if ungrouped:
        raise ValueError(
            "以下工具未归入任何能力分组（请在 tools.py 的 _CAPABILITY_GROUPS 登记）: "
            f"{sorted(ungrouped)}"
        )


_validate_capability_groups()


def build_capability_lines(visible_tool_names: set[str] | None = None) -> list[str]:
    """按当前用户可见工具名集合生成 AI 能力清单。

    visible_tool_names 为 None → 渲染全部能力段（未接权限的内部调用兜底，等价 admin）；
    否则仅渲染「组内任一工具可见」的能力段——AI 宣称的能力严格等于它能调的工具，
    与 get_tools_for_agent 的 fail-closed 过滤同源。空集 → 无能力段（由调用方补基础句）。
    """
    if visible_tool_names is None:
        visible_tool_names = set(GENERAL_TOOL_NAMES)
    visible = set(visible_tool_names)
    return [
        f"- **{label}**：{desc}"
        for label, names, desc in _CAPABILITY_GROUPS
        if names & visible
    ]


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


