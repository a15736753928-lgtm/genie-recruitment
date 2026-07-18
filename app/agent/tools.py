"""Agent Tool Registry — all tools the AI Agent can call."""
import json
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session_factory


async def _get_db() -> AsyncSession:
    async with async_session_factory() as session:
        return session


# ── Tool Definitions ────────────────────────────────────

TOOL_REGISTRY = {
    # Recruitment tools
    "list_resumes": {
        "name": "list_resumes",
        "description": "查询候选人列表。可按岗位、状态、关键词筛选，支持排序和分页。",
        "parameters": {
            "type": "object",
            "properties": {
                "positionId": {"type": "string", "description": "岗位ID，不传表示全部"},
                "statuses": {"type": "string", "description": "状态列表，逗号分隔"},
                "keyword": {"type": "string", "description": "搜索关键词"},
                "sortBy": {"type": "string", "enum": ["score", "uploadTime"], "description": "排序字段"},
                "limit": {"type": "integer", "description": "返回数量限制，默认10"},
            },
        },
    },
    "get_resume": {
        "name": "get_resume",
        "description": "获取单个候选人的详细信息，包括AI分析结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
            },
            "required": ["id"],
        },
    },
    "list_positions": {
        "name": "list_positions",
        "description": "获取所有岗位列表。",
        "parameters": {"type": "object", "properties": {}},
    },
    "update_resume": {
        "name": "update_resume",
        "description": "更新候选人信息或状态。可更新基本信息、技能、工作经历等。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "候选人ID"},
                "status": {"type": "string", "description": "新状态：job_hunting/passed/first_interview/..."},
                "fields": {"type": "object", "description": "要更新的字段"},
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
        "description": "获取单个岗位的详细信息（含 JD、筛选标准、面试标准等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "岗位ID"},
            },
            "required": ["id"],
        },
    },
    "create_position": {
        "name": "create_position",
        "description": "创建新招聘岗位，含 JD、部门、章节数等。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "岗位名称"},
                "department": {"type": "string", "description": "部门"},
                "chapterNumber": {"type": "integer", "description": "章节数（可选）"},
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
        "description": "更新岗位信息，包括 JD、筛选标准、面试标准、试用期项目要求等所有配置。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "岗位ID"},
                "fields": {"type": "object", "description": "要更新的字段（name/department/jdResponsibilities/jdRequirements/screeningCriteria/interviewCriteriaR1 等）"},
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
        "description": "获取岗位题库中某轮次的题目列表。",
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
                "round": {"type": "string", "enum": ["first", "second"]},
                "questions": {"type": "array", "items": {"type": "object"}, "description": "题目数组，每项含 index/content/category/difficulty"},
            },
            "required": ["positionId", "round", "questions"],
        },
    },

    # Interview tools
    "get_questions": {
        "name": "get_questions",
        "description": "获取或自动生成面试题单。若题单不存在会自动调用AI生成。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string", "enum": ["first", "second"]},
            },
            "required": ["candidateId", "round"],
        },
    },
    "generate_questions": {
        "name": "generate_questions",
        "description": "为候选人重新生成面试题目。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string", "enum": ["first", "second"]},
            },
            "required": ["candidateId", "round"],
        },
    },
    "get_evaluation": {
        "name": "get_evaluation",
        "description": "获取候选人的面试评分详情。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
                "round": {"type": "string"},
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
                "round": {"type": "string", "enum": ["first", "second"]},
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
                "round": {"type": "string", "enum": ["first", "second"]},
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
                "round": {"type": "string", "enum": ["first", "second"]},
                "scores": {"type": "array", "items": {"type": "object"}, "description": "每项含 questionId/hrScore/answer/dimensions"},
            },
            "required": ["candidateId", "round", "scores"],
        },
    },
    "submit_evaluation": {
        "name": "submit_evaluation",
        "description": "提交候选人面试评定（将已评分题目状态置为 scored）。",
        "parameters": {
            "type": "object",
            "properties": {
                "candidateId": {"type": "string"},
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
                "status": {"type": "string", "description": "assessing/passed/failed"},
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
        "description": "获取单个试用期员工详情（含任务、周报、转正评估）。",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "员工ID"},
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
    "save_week1_assessment": {
        "name": "save_week1_assessment",
        "description": "保存试用期第一周评估（4个维度打分：完成度/准确度/问题解决/规范度，总分>=70通过）。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "dimensionCompletion": {"type": "integer", "description": "完成度 0-100"},
                "dimensionFidelity": {"type": "integer", "description": "准确度 0-100"},
                "dimensionProblemSolving": {"type": "integer", "description": "问题解决 0-100"},
                "dimensionStandards": {"type": "integer", "description": "规范度 0-100"},
                "deductionReasons": {"type": "object", "description": "扣分原因（可选）"},
                "assessorSignature": {"type": "string", "description": "评估人签名（可选）"},
                "deptHeadSignature": {"type": "string", "description": "部门负责人签名（可选）"},
            },
            "required": ["employeeId", "dimensionCompletion", "dimensionFidelity", "dimensionProblemSolving", "dimensionStandards"],
        },
    },
    "save_conversion": {
        "name": "save_conversion",
        "description": "保存转正评估（项目表现60%+技术能力20%+团队协作20%，>=80转正/70-79延长/<70不通过）。",
        "parameters": {
            "type": "object",
            "properties": {
                "employeeId": {"type": "string"},
                "projectPerformanceScore": {"type": "integer", "description": "项目表现 0-100"},
                "techCapabilityScore": {"type": "integer", "description": "技术能力 0-100"},
                "collaborationScore": {"type": "integer", "description": "团队协作 0-100"},
                "mentorComments": {"type": "string", "description": "导师评语（可选）"},
                "mentorSignature": {"type": "string"},
                "deptHeadSignature": {"type": "string"},
                "hrSignature": {"type": "string"},
            },
            "required": ["employeeId", "projectPerformanceScore", "techCapabilityScore", "collaborationScore"],
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
        "description": "更新试用期任务（状态/标题/描述/评审意见）。",
        "parameters": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string"},
                "fields": {"type": "object", "description": "title/status/description/reviewNotes/weekNumber/deadline"},
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
                "status": {"type": "string", "description": "assessing/passed/failed"},
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
        "description": "在知识库中检索相关内容（RAG）。用于查面试题、公司制度、技术规范等。",
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
        "description": "上传知识库素材文件到 MinIO，返回 object key。fileKey 必须由前端「添加资料」上传后获得。",
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
        "description": "获取运营看板数据。包含概览统计、风险、漏斗等。",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_dashboard_overview": {
        "name": "get_dashboard_overview",
        "description": "获取数据看板概览（各模块统计汇总）。",
        "parameters": {"type": "object", "properties": {}},
    },

    # System
    "get_settings": {
        "name": "get_settings",
        "description": "获取系统设置。",
        "parameters": {"type": "object", "properties": {}},
    },
    "update_settings": {
        "name": "update_settings",
        "description": "更新系统设置（如公司名、自动解析开关、合格分数线、试用期天数等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {"type": "object", "description": "要更新的设置字段，如 companyName/autoParseResume/minMatchScore 等"},
            },
            "required": ["fields"],
        },
    },
}


def get_tools_for_agent(agent_id: str = "recruit") -> List[dict]:
    """Get the list of tool definitions for a specific agent type.

    The system now defaults to a single omnipotent agent ("genie") that has
    access to ALL tools. The legacy 4-agent IDs (recruit/interview/training/
    performance) are kept for backward compatibility and still return a
    scoped subset, but the frontend is expected to use the "genie" agent.
    """
    # The omnipotent agent gets every tool in the registry.
    if agent_id in ("genie", "all", "omnipotent"):
        return list(TOOL_REGISTRY.values())

    # Legacy scoped subsets (kept for backward compat / @-mention routing).
    recruit_tools = [
        "list_resumes", "get_resume", "list_positions", "get_position",
        "update_resume", "upload_resume", "batch_parse_resumes",
        "reanalyze_resume", "delete_resume", "rag_search",
    ]
    interview_tools = [
        "get_questions", "save_questions", "generate_questions", "replace_question",
        "get_evaluation", "save_evaluation", "submit_evaluation",
        "ai_score_question", "get_leaderboard", "get_rankings", "rag_search",
    ]
    training_tools = [
        "list_probation", "get_probation_stats", "get_probation_employee",
        "create_probation_employee", "save_week1_assessment", "save_conversion",
        "create_probation_task", "update_probation_task",
        "ai_evaluate_probation", "update_probation_status", "manual_review_probation",
        "rag_search",
    ]
    performance_tools = [
        "list_performance", "get_performance_stats", "get_department_performance",
        "get_grade_distribution", "get_bonus_info", "get_quarter_trends",
        "initiate_appraisal", "update_bonus", "rag_search",
    ]

    tool_map = {
        "recruit": recruit_tools,
        "interview": interview_tools,
        "training": training_tools,
        "performance": performance_tools,
    }

    # All agents get some common tools
    common = [
        "get_operations_dashboard", "get_dashboard_overview",
        "get_settings", "update_settings",
        "list_knowledge", "get_knowledge_stats", "get_knowledge_categories",
        "create_knowledge_item", "update_knowledge_item", "delete_knowledge_item",
        "recall_test", "upload_knowledge_file",
        "list_knowledge_bases", "create_knowledge_base", "update_knowledge_base",
        "delete_knowledge_base", "upload_document", "list_documents", "delete_document",
    ]
    names = tool_map.get(agent_id, recruit_tools) + common

    return [TOOL_REGISTRY[name] for name in names if name in TOOL_REGISTRY]


# ── LangChain / LangGraph Tool Conversion ────────────────

import inspect
from typing import get_type_hints
from pydantic import create_model, Field


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


async def _execute_tool_sync(tool_name: str, **kwargs) -> str:
    """Execute a tool with an auto-created DB session (for LangGraph context).

    Uses lazy import to avoid circular dependency with app.routers.ai_agent.
    """
    from app.routers.ai_agent import execute_tool_call  # lazy import

    # Remove None values (unset optional params)
    params = {k: v for k, v in kwargs.items() if v is not None}
    # Remove the dummy field if present
    params.pop("dummy", None)

    async with async_session_factory() as db:
        return await execute_tool_call(tool_name, params, db)


def create_langchain_tools(agent_id: str = "recruit") -> list:
    """Convert the tool registry into LangChain StructuredTool objects.

    Returns a list of tools compatible with LangGraph's ToolNode.
    """
    from langchain_core.tools import StructuredTool

    tool_defs = get_tools_for_agent(agent_id)
    lc_tools = []

    for td in tool_defs:
        tool_name = td["name"]
        description = td["description"]
        params_schema = td.get("parameters", {})

        # Build Pydantic args model from JSON Schema
        args_model = _build_pydantic_model(tool_name, params_schema)

        # Create the async tool function bound to this tool_name
        async def tool_func(tool_name=tool_name, **kwargs) -> str:
            return await _execute_tool_sync(tool_name, **kwargs)

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
