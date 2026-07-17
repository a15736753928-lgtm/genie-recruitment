"""
V3 Migration: Add COMMENT ON COLUMN for all tables.
Fixes: actual column names + individual transactions per statement.
"""
import asyncio
from sqlalchemy import text
from app.database import engine

COMMENTS = [
    # ══════════════════════════════════════════════════════
    # positions
    # ══════════════════════════════════════════════════════
    ("positions", "id", "岗位主键(UUID)"),
    ("positions", "name", "岗位名称(唯一)"),
    ("positions", "chapter_number", "手册章节号(第5-10章对应6个核心岗位)"),
    ("positions", "department", "所属部门"),
    ("positions", "jd_content", "岗位描述(旧字段,兼容保留)"),
    ("positions", "jd_responsibilities", "岗位职责"),
    ("positions", "jd_requirements", "任职要求"),
    ("positions", "jd_preferred", "优先条件/加分项"),
    ("positions", "jd_tech_stack", "核心技术栈"),
    ("positions", "screening_criteria", "简历筛选双维度评分标准(JSONB): 岗位匹配度60% + 简历内容质量40%"),
    ("positions", "interview_criteria_r1", "第一轮面试评分表(JSONB): 10维度x10分, >=80进二面, 70-79入人才池"),
    ("positions", "interview_criteria_r2", "第二轮面试评分表(JSONB): 10维度x10分, >=85录用"),
    ("positions", "week1_project_requirement", "试用期第一周项目复现要求(JSONB)"),
    ("positions", "weeks_2_4_plan", "试用期第2-4周工作规划(JSONB)"),
    ("positions", "later_week_scoring", "后续周统一考核标准(JSONB): 交付质量40%+交付效率30%+技术能力15%+业务沟通15%"),
    ("positions", "conversion_criteria", "转正考核标准(JSONB): 项目表现60%+技术能力20%+团队协作20%"),
    ("positions", "created_at", "创建时间"),
    ("positions", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # position_questions
    # ══════════════════════════════════════════════════════
    ("position_questions", "id", "题目主键(UUID)"),
    ("position_questions", "position_id", "所属岗位(FK->positions)"),
    ("position_questions", "round", "面试轮次(first/second)"),
    ("position_questions", "index_num", "题目序号"),
    ("position_questions", "content", "题目内容"),
    ("position_questions", "category", "题目分类(如: 岗位基础专业知识/岗位常规操作与排查)"),
    ("position_questions", "difficulty", "难度(easy/medium/hard)"),
    ("position_questions", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # candidates
    # ══════════════════════════════════════════════════════
    ("candidates", "id", "候选人主键(UUID)"),
    ("candidates", "name", "姓名"),
    ("candidates", "gender", "性别(男/女/未知)"),
    ("candidates", "age", "年龄"),
    ("candidates", "education", "最高学历"),
    ("candidates", "experience", "工作年限"),
    ("candidates", "ethnicity", "民族(默认汉族)"),
    ("candidates", "native_place", "籍贯或现居地"),
    ("candidates", "phone", "手机号"),
    ("candidates", "email", "邮箱"),
    ("candidates", "position_id", "应聘岗位(FK->positions)"),
    ("candidates", "score", "综合得分(AI初筛总分)"),
    ("candidates", "status", "状态(job_hunting求职中/passed通过/first_interview一面/second_interview二面/failed淘汰/expired过期)"),
    ("candidates", "resume_file", "简历文件路径"),
    ("candidates", "upload_time", "简历上传日期"),
    ("candidates", "screening_ai_score", "AI初筛双维度评分(0-100)"),
    ("candidates", "screening_manual_confirmed", "是否经人工复核确认"),
    ("candidates", "screening_confirmed_by", "复核人姓名"),
    ("candidates", "interviewer", "面试官"),
    ("candidates", "interview_round", "当前面试轮次(first/second/null)"),
    ("candidates", "created_at", "创建时间"),
    ("candidates", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # candidate_skills
    # ══════════════════════════════════════════════════════
    ("candidate_skills", "candidate_id", "候选人ID(FK->candidates)"),
    ("candidate_skills", "skill", "技能名称"),

    # ══════════════════════════════════════════════════════
    # candidate_educations
    # ══════════════════════════════════════════════════════
    ("candidate_educations", "id", "记录主键(UUID)"),
    ("candidate_educations", "candidate_id", "候选人ID(FK->candidates)"),
    ("candidate_educations", "school", "学校名称"),
    ("candidate_educations", "degree", "学位"),
    ("candidate_educations", "major", "专业"),
    ("candidate_educations", "period", "就读时间段"),

    # ══════════════════════════════════════════════════════
    # candidate_work_experiences
    # ══════════════════════════════════════════════════════
    ("candidate_work_experiences", "id", "记录主键(UUID)"),
    ("candidate_work_experiences", "candidate_id", "候选人ID(FK->candidates)"),
    ("candidate_work_experiences", "company", "公司名称"),
    ("candidate_work_experiences", "role", "职位"),
    ("candidate_work_experiences", "period", "工作时间段"),
    ("candidate_work_experiences", "description", "工作描述"),

    # ══════════════════════════════════════════════════════
    # candidate_project_experiences
    # ══════════════════════════════════════════════════════
    ("candidate_project_experiences", "id", "记录主键(UUID)"),
    ("candidate_project_experiences", "candidate_id", "候选人ID(FK->candidates)"),
    ("candidate_project_experiences", "name", "项目名称"),
    ("candidate_project_experiences", "role", "项目角色"),
    ("candidate_project_experiences", "period", "项目时间段"),
    ("candidate_project_experiences", "description", "项目描述"),

    # ══════════════════════════════════════════════════════
    # candidate_ai_analyses
    # ══════════════════════════════════════════════════════
    ("candidate_ai_analyses", "candidate_id", "候选人ID(FK->candidates, 一对一)"),
    ("candidate_ai_analyses", "overall_score", "AI综合评分(0-100)"),
    ("candidate_ai_analyses", "summary", "综合评价摘要"),
    ("candidate_ai_analyses", "position_match", "岗位匹配度分析"),
    ("candidate_ai_analyses", "experience_insight", "经验洞察"),
    ("candidate_ai_analyses", "recommendation", "推荐建议"),
    ("candidate_ai_analyses", "keywords", "关键词列表(JSON)"),
    ("candidate_ai_analyses", "highlights", "亮点列表(JSON)"),
    ("candidate_ai_analyses", "risks", "风险点列表(JSON)"),
    ("candidate_ai_analyses", "dimensions", "各维度评分(JSON: [{name, score}, ...])"),
    ("candidate_ai_analyses", "analyzed_at", "分析时间"),

    # ══════════════════════════════════════════════════════
    # talent_pool
    # ══════════════════════════════════════════════════════
    ("talent_pool", "id", "记录主键(UUID)"),
    ("talent_pool", "candidate_id", "候选人ID(FK->candidates)"),
    ("talent_pool", "position_id", "岗位ID(FK->positions)"),
    ("talent_pool", "score", "面试得分"),
    ("talent_pool", "source_round", "来源轮次(first/second)"),
    ("talent_pool", "added_at", "入池时间"),
    ("talent_pool", "notes", "备注"),

    # ══════════════════════════════════════════════════════
    # interview_questions
    # ══════════════════════════════════════════════════════
    ("interview_questions", "id", "题目主键(UUID)"),
    ("interview_questions", "candidate_id", "候选人ID(FK->candidates)"),
    ("interview_questions", "round", "面试轮次(first/second)"),
    ("interview_questions", "index_num", "题目序号"),
    ("interview_questions", "content", "题目内容"),
    ("interview_questions", "category", "题目分类"),
    ("interview_questions", "difficulty", "难度(easy/medium/hard)"),
    ("interview_questions", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # interview_evaluations
    # ══════════════════════════════════════════════════════
    ("interview_evaluations", "id", "评分记录主键(UUID)"),
    ("interview_evaluations", "candidate_id", "候选人ID(FK->candidates)"),
    ("interview_evaluations", "round", "面试轮次(first/second)"),
    ("interview_evaluations", "question_id", "关联题目(FK->interview_questions)"),
    ("interview_evaluations", "answer", "候选人回答文本"),
    ("interview_evaluations", "ai_score", "AI评分"),
    ("interview_evaluations", "ai_dimensions", "AI各维度评分(JSON)"),
    ("interview_evaluations", "hr_score", "面试官(HR)评分"),
    ("interview_evaluations", "hr_dimensions", "面试官各维度评分(JSON)"),
    ("interview_evaluations", "status", "评分状态(pending/scoring/scored)"),
    ("interview_evaluations", "transcript", "转写文本"),
    ("interview_evaluations", "audio_uploaded", "是否上传录音"),
    ("interview_evaluations", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # interview_transcripts
    # ══════════════════════════════════════════════════════
    ("interview_transcripts", "id", "记录主键(UUID)"),
    ("interview_transcripts", "candidate_id", "候选人ID(FK->candidates)"),
    ("interview_transcripts", "round", "面试轮次(first/second)"),
    ("interview_transcripts", "content", "转写/上传文本内容"),
    ("interview_transcripts", "source", "来源(upload文件上传/transcribe语音转写)"),
    ("interview_transcripts", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # employees
    # ══════════════════════════════════════════════════════
    ("employees", "id", "员工主键(UUID)"),
    ("employees", "candidate_id", "来源候选人ID(FK->candidates)"),
    ("employees", "position_id", "岗位ID(FK->positions)"),
    ("employees", "name", "姓名"),
    ("employees", "gender", "性别"),
    ("employees", "age", "年龄"),
    ("employees", "department", "部门"),
    ("employees", "join_date", "入职日期"),
    ("employees", "probation_end", "试用期截止日期"),
    ("employees", "status", "试用期状态(assessing考核中/passed通过/failed淘汰)"),
    ("employees", "ai_score", "AI评估综合分"),
    ("employees", "ai_result", "AI评估结论"),
    ("employees", "mentor_name", "带教负责人姓名"),
    ("employees", "mentor_id", "带教负责人ID(FK->users)"),
    ("employees", "week1_score", "第一周项目复现考核得分(0-100, >=70通过)"),
    ("employees", "week1_passed", "第一周是否通过"),
    ("employees", "conversion_score", "转正考核加权总分"),
    ("employees", "conversion_decision", "转正决定(converted转正/extended延期/rejected辞退)"),
    ("employees", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # probation_tasks
    # ══════════════════════════════════════════════════════
    ("probation_tasks", "id", "任务主键(UUID)"),
    ("probation_tasks", "employee_id", "员工ID(FK->employees)"),
    ("probation_tasks", "title", "任务标题"),
    ("probation_tasks", "week_number", "所属周次(1/2/3/4)"),
    ("probation_tasks", "description", "任务描述"),
    ("probation_tasks", "status", "任务状态(pending待开始/in_progress进行中/completed已完成)"),
    ("probation_tasks", "deadline", "截止日期"),
    ("probation_tasks", "review_notes", "评审备注"),
    ("probation_tasks", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # probation_week1_assessments
    # ══════════════════════════════════════════════════════
    ("probation_week1_assessments", "id", "考核记录主键(UUID)"),
    ("probation_week1_assessments", "employee_id", "员工ID(FK->employees, 一对一)"),
    ("probation_week1_assessments", "dimension_completion", "项目复现完整度(满分30)"),
    ("probation_week1_assessments", "dimension_fidelity", "代码/方案还原度(满分25)"),
    ("probation_week1_assessments", "dimension_problem_solving", "独立解决问题能力(满分25)"),
    ("probation_week1_assessments", "dimension_standards", "规范性与总结(满分20)"),
    ("probation_week1_assessments", "total_score", "总分(满分100, >=70通过)"),
    ("probation_week1_assessments", "deduction_reasons", "扣分原因(JSON)"),
    ("probation_week1_assessments", "assessor_signature", "考核人(带教负责人)签字"),
    ("probation_week1_assessments", "dept_head_signature", "部门负责人签字"),
    ("probation_week1_assessments", "assessor_date", "考核日期"),
    ("probation_week1_assessments", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # probation_conversions
    # ══════════════════════════════════════════════════════
    ("probation_conversions", "id", "转正考核主键(UUID)"),
    ("probation_conversions", "employee_id", "员工ID(FK->employees, 一对一)"),
    ("probation_conversions", "project_performance_score", "后三周真实项目表现得分(满分100)"),
    ("probation_conversions", "project_performance_weight", "项目表现权重(默认0.60)"),
    ("probation_conversions", "tech_capability_score", "技术能力与业务产出得分(满分100)"),
    ("probation_conversions", "tech_capability_weight", "技术能力权重(默认0.20)"),
    ("probation_conversions", "collaboration_score", "团队协作与综合素养得分(满分100)"),
    ("probation_conversions", "collaboration_weight", "团队协作权重(默认0.20)"),
    ("probation_conversions", "total_score", "加权总分(>=80转正, 70-79延期, <70辞退)"),
    ("probation_conversions", "decision", "转正决定(converted/extended/rejected)"),
    ("probation_conversions", "mentor_comments", "带教负责人综合评价"),
    ("probation_conversions", "mentor_signature", "带教负责人签字"),
    ("probation_conversions", "mentor_date", "带教负责人签字日期"),
    ("probation_conversions", "dept_head_signature", "部门负责人签字"),
    ("probation_conversions", "dept_head_date", "部门负责人签字日期"),
    ("probation_conversions", "hr_signature", "HR复核签字"),
    ("probation_conversions", "hr_date", "HR复核日期"),
    ("probation_conversions", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # performance_quarters
    # ══════════════════════════════════════════════════════
    ("performance_quarters", "quarter", "季度标识(如2026-Q3)"),
    ("performance_quarters", "status", "季度状态(draft草稿/active进行中/closed已关闭)"),
    ("performance_quarters", "bonus_pool", "奖金池总额"),
    ("performance_quarters", "distributed", "已分配金额"),
    ("performance_quarters", "initiated_at", "启动时间"),

    # ══════════════════════════════════════════════════════
    # performance_records
    # ══════════════════════════════════════════════════════
    ("performance_records", "id", "记录主键(UUID)"),
    ("performance_records", "employee_id", "员工ID(FK->employees)"),
    ("performance_records", "quarter", "所属季度"),
    ("performance_records", "tasks_completed", "完成任务数"),
    ("performance_records", "quality", "质量评分"),
    ("performance_records", "speed", "效率评分"),
    ("performance_records", "compliance", "合规评分"),
    ("performance_records", "total_score", "总分"),
    ("performance_records", "grade", "绩效等级(S/A/B+/B/C)"),
    ("performance_records", "bonus", "奖金金额"),
    ("performance_records", "rank", "排名"),

    # ══════════════════════════════════════════════════════
    # users
    # ══════════════════════════════════════════════════════
    ("users", "id", "用户主键(UUID)"),
    ("users", "username", "用户名(唯一)"),
    ("users", "password_hash", "密码哈希(bcrypt)"),
    ("users", "display_name", "显示名称"),
    ("users", "email", "邮箱"),
    ("users", "role", "角色(hr_admin管理员/hr_recruiter招聘专员/interviewer面试官/dept_manager部门经理)"),
    ("users", "department", "所属部门"),
    ("users", "is_active", "是否启用"),
    ("users", "created_at", "创建时间"),
    ("users", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # system_settings
    # ══════════════════════════════════════════════════════
    ("system_settings", "key", "设置键(global=全局设置)"),
    ("system_settings", "value", "设置值(JSONB)"),
    ("system_settings", "updated_at", "最后更新时间"),
    ("system_settings", "updated_by", "更新人ID(FK->users)"),

    # ══════════════════════════════════════════════════════
    # knowledge_categories
    # ══════════════════════════════════════════════════════
    ("knowledge_categories", "key", "分类键(唯一标识)"),
    ("knowledge_categories", "title", "分类名称"),
    ("knowledge_categories", "parent_key", "上级分类键(可为空)"),
    ("knowledge_categories", "sort_order", "排序"),

    # ══════════════════════════════════════════════════════
    # knowledge_bases
    # ══════════════════════════════════════════════════════
    ("knowledge_bases", "id", "知识库主键"),
    ("knowledge_bases", "name", "知识库名称"),
    ("knowledge_bases", "description", "知识库描述"),
    ("knowledge_bases", "doc_count", "文档数量"),
    ("knowledge_bases", "chunk_count", "分块数量"),
    ("knowledge_bases", "created_at", "创建时间"),
    ("knowledge_bases", "updated_at", "最后更新时间"),
    ("knowledge_bases", "owner_id", "所有者ID(FK->users)"),

    # ══════════════════════════════════════════════════════
    # knowledge_documents (actual columns)
    # ══════════════════════════════════════════════════════
    ("knowledge_documents", "id", "文档主键"),
    ("knowledge_documents", "kb_id", "所属知识库ID(FK->knowledge_bases)"),
    ("knowledge_documents", "file_name", "文件名"),
    ("knowledge_documents", "file_size", "文件大小(字节)"),
    ("knowledge_documents", "file_type", "文件类型"),
    ("knowledge_documents", "chunk_count", "分块数量"),
    ("knowledge_documents", "object_key", "对象存储键"),
    ("knowledge_documents", "file_hash", "文件哈希"),
    ("knowledge_documents", "status", "处理状态"),
    ("knowledge_documents", "uploaded_at", "上传时间"),
    ("knowledge_documents", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # knowledge_chunks (actual columns)
    # ══════════════════════════════════════════════════════
    ("knowledge_chunks", "id", "分块主键"),
    ("knowledge_chunks", "doc_id", "所属文档ID(FK->knowledge_documents)"),
    ("knowledge_chunks", "kb_id", "所属知识库ID(FK->knowledge_bases)"),
    ("knowledge_chunks", "chunk_index", "分块序号"),
    ("knowledge_chunks", "chunk_text", "分块文本内容"),
    ("knowledge_chunks", "milvus_pk", "Milvus向量库主键"),
    ("knowledge_chunks", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # knowledge_items
    # ══════════════════════════════════════════════════════
    ("knowledge_items", "id", "知识条目主键(UUID)"),
    ("knowledge_items", "name", "条目名称"),
    ("knowledge_items", "category_key", "分类键(FK->knowledge_categories)"),
    ("knowledge_items", "category_path", "分类路径"),
    ("knowledge_items", "type", "类型(interview/standard/rule/data)"),
    ("knowledge_items", "file_path", "文件路径"),
    ("knowledge_items", "content", "文本内容"),
    ("knowledge_items", "recall_count", "召回次数"),
    ("knowledge_items", "training_status", "训练状态"),
    ("knowledge_items", "milvus_ids", "Milvus向量ID列表(JSON)"),
    ("knowledge_items", "created_at", "创建时间"),
    ("knowledge_items", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # knowledge_training_jobs
    # ══════════════════════════════════════════════════════
    ("knowledge_training_jobs", "id", "训练任务主键(UUID)"),
    ("knowledge_training_jobs", "status", "训练状态"),
    ("knowledge_training_jobs", "total_chunks", "总分块数"),
    ("knowledge_training_jobs", "started_at", "开始时间"),
    ("knowledge_training_jobs", "finished_at", "结束时间"),

    # ══════════════════════════════════════════════════════
    # ingestion_tasks
    # ══════════════════════════════════════════════════════
    ("ingestion_tasks", "id", "摄入任务主键"),
    ("ingestion_tasks", "doc_id", "文档ID(FK->knowledge_documents)"),
    ("ingestion_tasks", "kb_id", "知识库ID(FK->knowledge_bases)"),
    ("ingestion_tasks", "file_name", "文件名"),
    ("ingestion_tasks", "file_size", "文件大小(字节)"),
    ("ingestion_tasks", "status", "处理状态"),
    ("ingestion_tasks", "progress", "进度(0-100)"),
    ("ingestion_tasks", "message", "状态消息"),
    ("ingestion_tasks", "created_at", "创建时间"),
    ("ingestion_tasks", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # agent_sessions
    # ══════════════════════════════════════════════════════
    ("agent_sessions", "id", "会话主键(UUID)"),
    ("agent_sessions", "user_id", "用户ID(FK->users)"),
    ("agent_sessions", "title", "会话标题"),
    ("agent_sessions", "agent_id", "Agent类型标识"),
    ("agent_sessions", "created_at", "创建时间"),
    ("agent_sessions", "updated_at", "最后更新时间"),

    # ══════════════════════════════════════════════════════
    # agent_messages
    # ══════════════════════════════════════════════════════
    ("agent_messages", "id", "消息主键(UUID)"),
    ("agent_messages", "session_id", "所属会话(FK->agent_sessions)"),
    ("agent_messages", "role", "角色(user/assistant)"),
    ("agent_messages", "content", "消息内容"),
    ("agent_messages", "thinking", "思考过程"),
    ("agent_messages", "tool_blocks", "工具调用记录(JSON)"),
    ("agent_messages", "handoffs", "Agent交接记录(JSON)"),
    ("agent_messages", "created_at", "创建时间"),

    # ══════════════════════════════════════════════════════
    # agent_tasks
    # ══════════════════════════════════════════════════════
    ("agent_tasks", "id", "任务主键(UUID)"),
    ("agent_tasks", "session_id", "关联会话(FK->agent_sessions)"),
    ("agent_tasks", "title", "任务标题"),
    ("agent_tasks", "description", "任务描述"),
    ("agent_tasks", "progress", "进度(0-100)"),
    ("agent_tasks", "status", "任务状态"),
    ("agent_tasks", "started_at", "开始时间"),
    ("agent_tasks", "finished_at", "完成时间"),

    # ══════════════════════════════════════════════════════
    # agent_materials
    # ══════════════════════════════════════════════════════
    ("agent_materials", "id", "素材主键(UUID)"),
    ("agent_materials", "session_id", "关联会话(FK->agent_sessions)"),
    ("agent_materials", "user_id", "上传用户ID(FK->users)"),
    ("agent_materials", "name", "素材名称"),
    ("agent_materials", "type", "素材类型(file/resume/jd/material/knowledge)"),
    ("agent_materials", "knowledge_id", "关联知识库条目ID"),
    ("agent_materials", "file_path", "文件路径"),
    ("agent_materials", "uploaded_at", "上传时间"),

    # ══════════════════════════════════════════════════════
    # audit_logs
    # ══════════════════════════════════════════════════════
    ("audit_logs", "id", "审计日志主键(UUID)"),
    ("audit_logs", "actor_id", "操作人ID(FK->users)"),
    ("audit_logs", "actor_name", "操作人姓名"),
    ("audit_logs", "action", "操作描述"),
    ("audit_logs", "section", "操作模块"),
    ("audit_logs", "metadata", "附加数据(JSON)"),
    ("audit_logs", "created_at", "操作时间"),
]


async def run():
    """Execute each COMMENT in its own connection to avoid transaction aborts."""
    from app.database import async_session_factory
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import text as _text

    success = 0
    skipped = 0
    for table, column, comment in COMMENTS:
        try:
            async with async_session_factory() as session:
                escaped = comment.replace("'", "''")
                sql = f"COMMENT ON COLUMN {table}.{column} IS '{escaped}'"
                await session.execute(_text(sql))
                await session.commit()
                success += 1
        except Exception as e:
            # Print first line of error only
            msg = str(e).split('\n')[0][:120]
            print(f"  SKIP {table}.{column}: {msg}")
            skipped += 1

    print(f"\nDone: {success} comments added, {skipped} skipped.")


if __name__ == "__main__":
    asyncio.run(run())
