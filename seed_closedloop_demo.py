"""闭环测试数据 —— 让几个虚构候选人/员工真实走完整条业务链路。

候选人 → 简历评分 → 面试(r1/r2) → offer 审批 → 员工 → 试用期(计划/周评/带教) →
转正评审 → 人才画像 → 工作任务/积分/奖惩 → 晋级 → 期权。

覆盖 6 条不同阶段的故事线，让每个页面都有真实、内部一致的数据可看：
  1. 陈明 —— 旗舰全流程：candidate(hired) → employee(formal) → 晋级(approved) → 期权(approved)
  2. 赵敏 —— 已在职正式员工（无候选人履历），晋级/期权都停在"待审批"，方便你在 UI 里亲手点通过
  3. 林娜 —— 候选人已录用，试用期进行中（第2周）
  4. 黄强 —— 候选人二面进行中
  5. 周婷 —— 候选人 offer 待审批
  6. 吴帆 —— 候选人已淘汰（进入人才库）

不会重复运行插入两遍：以 candidates.name / employees.name 做存在性检查。

用法：
    python seed_closedloop_demo.py
"""
import asyncio
import sys
import uuid
from datetime import date, datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("seed_closedloop_demo.py", 1)[0])

from sqlalchemy import select

from app.database import async_session_factory
from app.models.recruitment import (
    Position, Department, Candidate, CandidateSkill, CandidateEducation,
    CandidateWorkExperience, CandidateProjectExperience, CandidateAIAnalysis,
)
from app.models.interview import (
    InterviewQuestion, InterviewEvaluation, InterviewTranscript, InterviewSegmentEvaluation,
)
from app.models.phase1 import (
    ResumeScore, Interview, InterviewerScore, AIInterviewReport, OfferApproval,
    RecruitmentRequest,
)
from app.models.probation import Employee, ProbationTask
from app.models.phase2 import (
    ProbationPlan, ProbationWeekReview, ConfirmationReview, MentorRecord,
)
from app.models.performance import PerformanceRecord, PerformanceQuarter
from app.models.phase3 import WorkTask, TaskAcceptance, PointRecord, RewardPenaltyRecord, Appeal
from app.models.phase4 import (
    TalentProfile, AbilityTag, PromotionRecord, EquityRecord, Project, ProjectAssignment,
)
from app.models.system import StateTransition, ExceptionQueue
from app.models.auth import User, UserRole
from app.core.security import hash_password

NOW = datetime.utcnow()
TODAY = date.today()


def days_ago(n: int) -> datetime:
    return NOW - timedelta(days=n)


def date_days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


async def _refs(db):
    """拉取已有种子数据（部门/岗位/用户）的名称→ID 映射。"""
    positions = {p.name: p for p in (await db.execute(select(Position))).scalars().all()}
    departments = {d.name: d for d in (await db.execute(select(Department))).scalars().all()}
    users = {u.username: u for u in (await db.execute(select(User))).scalars().all()}
    projects = {p.name: p for p in (await db.execute(select(Project))).scalars().all()}
    return positions, departments, users, projects


async def already_seeded(db) -> bool:
    row = await db.execute(select(Candidate).where(Candidate.name == "陈明"))
    return row.scalar_one_or_none() is not None


# ══════════════════════════════════════════════════════════════
# 故事线 1：陈明 —— 旗舰全流程
# ══════════════════════════════════════════════════════════════

async def seed_chen_ming(db, positions, users, projects):
    pos = positions["全栈开发工程师"]
    hr, ivr, mgr, mtr, ceo, pld = users["hr01"], users["ivr01"], users["mgr01"], users["mtr01"], users["ceo01"], users["pld01"]

    candidate = Candidate(
        name="陈明", gender="男", age=27, education="本科", experience="4年",
        ethnicity="汉族", native_place="浙江杭州", phone="13800001001",
        email="chenming@example.com", position_id=pos.id, score=88,
        status="hired", resume_file="resumes/demo-chenming.pdf",
        interviewer="王强", interview_round="r2",
        upload_time=date_days_ago(70), created_at=days_ago(70), updated_at=days_ago(50),
    )
    db.add(candidate)
    await db.flush()

    db.add_all([
        CandidateSkill(candidate_id=candidate.id, skill=s)
        for s in ["React", "TypeScript", "Node.js", "PostgreSQL", "Docker"]
    ])
    db.add(CandidateEducation(
        candidate_id=candidate.id, school="浙江大学", degree="本科",
        major="计算机科学与技术", period="2018-2022",
    ))
    db.add(CandidateWorkExperience(
        candidate_id=candidate.id, company="某电商科技公司", role="全栈开发工程师",
        period="2022-2026", description="负责交易中台前后端开发，主导商品详情页重构，页面加载耗时下降40%。",
    ))
    db.add(CandidateProjectExperience(
        candidate_id=candidate.id, name="交易中台重构项目", role="核心开发",
        period="2024-2025", description="将单体应用拆分为微服务，引入 React + Node.js + PostgreSQL 技术栈，支撑日均千万级订单。",
    ))
    db.add(CandidateAIAnalysis(
        candidate_id=candidate.id, overall_score=88,
        summary="陈明拥有4年全栈开发经验，技术栈扎实，有大型电商中台的实战经验，简历信息完整、可验证性强。",
        position_match="技能栈（React/TypeScript/Node.js/PostgreSQL）与全栈开发工程师岗位高度匹配。",
        experience_insight="4年经验里有明确的项目主导经历和量化成果（加载耗时下降40%），可信度高。",
        recommendation="建议优先安排面试，重点考察系统设计能力与团队协作经验。",
        keywords=["React", "TypeScript", "Node.js", "PostgreSQL", "微服务", "全栈"],
        highlights=["主导交易中台重构，性能提升40%", "4年全栈经验，技术栈与岗位高度契合"],
        risks=["缺少大规模并发场景的量化数据佐证"],
        dimensions=[
            {"name": "教育背景", "score": 82}, {"name": "项目经验", "score": 90},
            {"name": "工作经验", "score": 88}, {"name": "技能匹配", "score": 92},
        ],
        analyzed_at=days_ago(70),
    ))

    db.add(ResumeScore(
        candidate_id=candidate.id, position_id=pos.id,
        skill_match=23, project_match=19, position_exp=14, achievement=14,
        industry_exp=9, learning=4, stability=4, bonus_skill=1, total=88, grade="A",
        advice={
            "summary": "8维评分优异，全栈技能扎实，建议直接进入一面",
            "highlights": ["技能匹配度高", "项目成果可量化"],
            "concerns": [],
        },
        created_at=days_ago(69), updated_at=days_ago(69),
    ))

    # ── 一面（r1） ──
    q1 = InterviewQuestion(
        candidate_id=candidate.id, round="r1", index_num=1,
        content="请介绍一下你主导的交易中台重构项目，具体做了哪些技术选型？",
        category="项目经验", difficulty="中", source="pre_generated", question_type="standard",
        created_at=days_ago(63),
    )
    q2 = InterviewQuestion(
        candidate_id=candidate.id, round="r1", index_num=2,
        content="React 中如何做性能优化？请结合你的实际经验说明。",
        category="技术能力", difficulty="中", source="pre_generated", question_type="standard",
        created_at=days_ago(63),
    )
    db.add_all([q1, q2])
    await db.flush()

    db.add_all([
        InterviewEvaluation(
            candidate_id=candidate.id, round="r1", question_id=q1.id,
            answer="我们把单体拆成了交易、库存、履约三个微服务，用 PostgreSQL 分库分表，Node.js 做 BFF 层聚合数据。",
            ai_score=88, ai_dimensions={"技术深度": 88, "表达清晰度": 85},
            hr_score=90, hr_dimensions={"技术深度": 90, "表达清晰度": 88},
            status="confirmed", audio_uploaded=False, updated_at=days_ago(62),
        ),
        InterviewEvaluation(
            candidate_id=candidate.id, round="r1", question_id=q2.id,
            answer="主要用 React.memo、虚拟列表、代码分割，配合 CDN 缓存静态资源，首屏时间从3.2s降到1.8s。",
            ai_score=85, ai_dimensions={"技术深度": 84, "表达清晰度": 87},
            hr_score=86, hr_dimensions={"技术深度": 85, "表达清晰度": 88},
            status="confirmed", audio_uploaded=False, updated_at=days_ago(62),
        ),
    ])

    interview_r1 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r1",
        scheduled_at=days_ago(63), interviewer_ids=[str(ivr.id)],
        status="completed", composite_score=86, conclusion="advance",
        created_at=days_ago(63), updated_at=days_ago(62),
    )
    db.add(interview_r1)
    await db.flush()
    db.add(InterviewerScore(
        interview_id=interview_r1.id, interviewer_id=ivr.id,
        dimensions={"技术能力": 88, "沟通表达": 86, "项目经验": 90, "学习能力": 82},
        total=86, comment="技术功底扎实，项目经历真实可信，建议进入二面。",
        is_submitted=True, created_at=days_ago(62), updated_at=days_ago(62),
    ))
    db.add(AIInterviewReport(
        candidate_id=candidate.id, interview_id=interview_r1.id, round="r1",
        score=87, authenticity_score=90,
        advice={"summary": "回答具体、有细节支撑，真实性高", "suggestions": ["可在二面深入考察系统设计能力"]},
        answered_directly=True, role_clear=True, concrete_result=True,
        process_described=True, contradiction_found=False, avoided_key=False, logical=True,
        created_at=days_ago(62),
    ))

    # ── 二面（r2，含实操分）──
    q3 = InterviewQuestion(
        candidate_id=candidate.id, round="r2", index_num=1,
        content="现场设计一个短链接服务的系统架构，说明如何应对高并发读。",
        category="系统设计", difficulty="难", source="pre_generated", question_type="practical",
        created_at=days_ago(58),
    )
    db.add(q3)
    await db.flush()
    db.add(InterviewEvaluation(
        candidate_id=candidate.id, round="r2", question_id=q3.id,
        answer="用一致性哈希做分片，读多写少场景加多级缓存（本地+Redis），写入用发号器保证唯一性。",
        ai_score=90, ai_dimensions={"系统设计": 90, "权衡取舍": 88},
        hr_score=91, hr_dimensions={"系统设计": 92, "权衡取舍": 89},
        status="confirmed", updated_at=days_ago(57),
    ))
    interview_r2 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r2",
        scheduled_at=days_ago(58), interviewer_ids=[str(ivr.id), str(mgr.id)],
        status="completed", practical_score=88, composite_score=90, conclusion="recommend",
        created_at=days_ago(58), updated_at=days_ago(57),
    )
    db.add(interview_r2)
    await db.flush()
    db.add_all([
        InterviewerScore(
            interview_id=interview_r2.id, interviewer_id=ivr.id,
            dimensions={"系统设计": 90, "编码能力": 88, "沟通协作": 87},
            total=88, comment="系统设计思路清晰，权衡取舍合理。", is_submitted=True,
            created_at=days_ago(57), updated_at=days_ago(57),
        ),
        InterviewerScore(
            interview_id=interview_r2.id, interviewer_id=mgr.id,
            dimensions={"系统设计": 92, "编码能力": 90, "沟通协作": 90},
            total=91, comment="综合素质优秀，推荐录用。", is_submitted=True,
            created_at=days_ago(57), updated_at=days_ago(57),
        ),
    ])
    db.add(AIInterviewReport(
        candidate_id=candidate.id, interview_id=interview_r2.id, round="r2",
        score=90, authenticity_score=92,
        advice={"summary": "实操环节表现优秀，架构设计合理", "suggestions": []},
        answered_directly=True, role_clear=True, concrete_result=True,
        process_described=True, contradiction_found=False, avoided_key=False, logical=True,
        created_at=days_ago(57),
    ))

    # ── offer 审批（已通过）──
    offer = OfferApproval(
        candidate_id=candidate.id, position_id=pos.id,
        resume_score=88, r1_score=86, r2_score=90, practical_score=88, team_score=88,
        final_score=88.4,
        ai_advice={"summary": "五维综合分88.4，达到优先录用线", "result": "priority"},
        strengths="全栈技术功底扎实，系统设计能力突出，沟通表达清晰。",
        capability_gaps="大规模分布式系统的实战经验可进一步积累。",
        risks_note="无明显风险。",
        suggested_salary="28K-32K",
        probation_goal="4周试用期内独立完成一个中等复杂度的模块开发并通过验收。",
        training_plan="第1周熟悉技术栈与团队规范，第2-4周参与核心项目并逐步独立负责模块。",
        mentor_id=mtr.id,
        conversion_criteria="试用期任务验收合格率≥80%，无重大违规。",
        elimination_criteria="连续两周任务不达标且无改善迹象。",
        approver_id=mgr.id, approved_at=days_ago(55),
        ai_result="priority", result="priority", status="approved",
        created_at=days_ago(56), updated_at=days_ago(55),
    )
    db.add(offer)

    db.add(StateTransition(
        entity_type="candidate", entity_id=candidate.id, from_status="round2",
        to_status="pending_offer", reason="二面综合分90，达到优先录用线",
        actor_id=mgr.id, actor_name="部门负责人", created_at=days_ago(56),
    ))
    db.add(StateTransition(
        entity_type="candidate", entity_id=candidate.id, from_status="pending_offer",
        to_status="hired", reason="offer 审批通过", actor_id=mgr.id,
        actor_name="部门负责人", created_at=days_ago(55),
    ))

    # ── 员工 + 试用期 ──
    employee = Employee(
        candidate_id=candidate.id, position_id=pos.id, name="陈明", gender="男", age=27,
        department="技术部", phone="13800001001", email="chenming@example.com",
        status="formal", onboard_date=date_days_ago(54), probation_end_date=date_days_ago(26),
        employee_type="tech", match_level="high", current_week=4, total_weeks=4,
        mentor="带教人", mentor_id=mtr.id, manager="部门负责人", manager_id=mgr.id,
        overall_score=90.5, risk_level="low",
        created_at=days_ago(54), updated_at=days_ago(20),
    )
    db.add(employee)
    await db.flush()

    for from_s, to_s, n, reason, actor in [
        (None, "pending_onboard", 54, "offer 审批通过，自动建档", mgr),
        ("pending_onboard", "training", 54, "入职当天进入培训", hr),
        ("training", "probation", 49, "5天标准培训完成，进入试用期", hr),
        ("probation", "pending_confirmation", 26, "4周试用期任务全部完成，提交转正评审", mtr),
        ("pending_confirmation", "formal", 20, "转正评审通过", mgr),
    ]:
        db.add(StateTransition(
            entity_type="employee", entity_id=employee.id, from_status=from_s, to_status=to_s,
            reason=reason, actor_id=actor.id, actor_name=actor.display_name, created_at=days_ago(n),
        ))

    db.add(ProbationPlan(
        employee_id=employee.id, type="tech", total_weeks=4,
        start_date=date_days_ago(49), end_date=date_days_ago(21), status="completed",
        ai_generated=True,
        weeks=[
            {"week": 1, "title": "环境搭建与规范熟悉", "focus": "熟悉团队技术规范与协作流程",
             "goals": ["搭建本地开发环境", "完成一个练习项目"], "trainingItems": ["代码规范培训", "Git 工作流培训"]},
            {"week": 2, "title": "核心模块开发", "focus": "参与核心项目模块开发",
             "goals": ["独立完成一个子模块"], "trainingItems": ["架构设计分享"]},
            {"week": 3, "title": "联调与优化", "focus": "跨模块联调与性能优化",
             "goals": ["完成联调并输出优化报告"], "trainingItems": []},
            {"week": 4, "title": "独立交付", "focus": "独立负责一个完整功能",
             "goals": ["功能上线并通过验收"], "trainingItems": []},
        ],
        created_by_id=mtr.id, created_at=days_ago(54), updated_at=days_ago(21),
    ))

    for week, score, comment, n in [
        (1, 82, "环境熟悉快，练习项目完成质量高。", 47),
        (2, 86, "独立完成子模块开发，代码质量良好。", 40),
        (3, 89, "联调积极主动，性能优化效果明显。", 33),
        (4, 92, "独立交付功能完整上线，验收一次通过。", 26),
    ]:
        db.add(ProbationWeekReview(
            employee_id=employee.id, week_number=week,
            dimensions={"工作态度": score - 2, "技术能力": score + 2, "协作沟通": score},
            total_score=score, comment=comment, reviewer_id=mtr.id, reviewer_name="带教人",
            reviewed_at=days_ago(n), created_at=days_ago(n),
        ))
        db.add(MentorRecord(
            employee_id=employee.id, week=week,
            training_content=f"第{week}周：{comment}",
            mastered_skills=["React", "TypeScript"] if week <= 2 else ["React", "TypeScript", "系统设计"],
            pending_skills=["大规模并发优化"] if week < 4 else [],
            completed_tasks=[f"第{week}周任务"], issues=[],
            improvement_plan="继续加强大规模并发场景的实战经验。" if week < 4 else "已达到独立交付水平。",
            mentor_score=score, employee_confirmed=True, confirmed_at=days_ago(n - 1),
            created_at=days_ago(n), created_by=mtr.id,
        ))

    task_defs = [
        ("环境搭建练习", 1, "passed", 20, "完成本地开发环境搭建并提交练习项目", 82),
        ("用户中心子模块开发", 2, "passed", 25, "独立完成用户中心子模块并通过代码评审", 86),
        ("交易链路联调与性能优化", 3, "passed", 30, "完成跨模块联调，接口平均响应时间下降15%", 89),
        ("独立交付：消息通知功能", 4, "closed", 40, "独立设计并交付消息通知功能，功能上线", 92),
    ]
    for title, week, status, points, obj, score in task_defs:
        db.add(ProbationTask(
            employee_id=employee.id, title=title, week_number=week,
            description=f"第{week}周试用期任务：{obj}", status=status,
            deadline=date_days_ago(54 - week * 7), objective=obj,
            assignee="陈明", deliverables="功能代码 + 说明文档", quality_standard="代码评审通过，无P0/P1缺陷",
            test_standard="单元测试覆盖率≥80%", reviewer="带教人", expected_points=points,
            score=score, project_scores={"代码质量": score, "进度把控": score - 3, "协作沟通": score + 1},
            submitted_at=days_ago(54 - week * 7 + 2), reviewed_at=days_ago(54 - week * 7 + 1),
            created_at=days_ago(54 - (week - 1) * 7),
        ))

    db.add(ConfirmationReview(
        employee_id=employee.id, week1_score=82, week2_score=86, week3_score=89, week4_score=92,
        mentor_score=90, discipline_score=95,
        overall_score=round(82 * 0.15 + 86 * 0.20 + 89 * 0.20 + 92 * 0.25 + 90 * 0.10 + 95 * 0.10, 2),
        ai_report={"summary": "四周评分持续走高，具备独立交付能力，建议转正", "risk": "低"},
        employee_summary="试用期内快速融入团队，独立完成消息通知功能上线，收获很大。",
        project_results=[{"name": "消息通知功能", "result": "已上线", "score": 92}],
        ability_gaps=[{"tag": "大规模并发", "gap": "需要更多实战积累"}],
        next_90days_goals=[{"goal": "主导一个中型项目模块", "measure": "按期交付且验收通过"}],
        recommendation="excellent", status="approved",
        mentor_comment="学习能力强，技术功底扎实，推荐转正。",
        manager_comment="综合表现优秀，同意转正。",
        manager_approved=True, manager_approved_at=days_ago(20),
        created_at=days_ago(21), updated_at=days_ago(20),
    ))

    # ── 转正后：登录账号 + 人才画像 + 工作任务/积分/奖惩 ──
    chenming_user = User(
        username="chenming01", password_hash=hash_password("123456"),
        display_name="陈明", department="技术部", employee_id=employee.id,
        must_change_password=False, created_at=days_ago(20),
    )
    db.add(chenming_user)
    await db.flush()
    db.add(UserRole(user_id=chenming_user.id, role_code="employee"))

    db.add(TalentProfile(
        employee_id=employee.id, current_position="全栈开发工程师", position_level="P2",
        ability_level="L3", skills=["React", "TypeScript", "Node.js", "PostgreSQL", "系统设计"],
        department="技术部", task_success_rate=96.5, on_time_rate=100, first_pass_rate=90,
        total_confirmed_points=680, task_count=3, reward_count=1, penalty_count=1, rework_rate=5,
        trainable_skills=["大规模分布式系统"], assignable_tasks=[{"level": "A", "count": 2}],
        can_mentor=True, promotion_readiness=92,
        talent_risk="normal",
        ai_analysis={"summary": "综合表现优异，具备晋升条件，建议纳入核心人才池"},
        created_at=days_ago(20), updated_at=days_ago(2),
    ))
    db.add(AbilityTag(
        employee_id=employee.id, tag="系统设计", level="L3",
        evidence=[{"type": "task", "ref": "交易链路联调与性能优化"}],
        confirmed_by=mtr.id, confirmed_at=days_ago(15), created_at=days_ago(15),
    ))
    db.add(AbilityTag(
        employee_id=employee.id, tag="React 工程化", level="L3",
        evidence=[{"type": "task", "ref": "独立交付：消息通知功能"}],
        confirmed_by=mtr.id, confirmed_at=days_ago(10), created_at=days_ago(10),
    ))

    wt1 = WorkTask(
        name="人才管理系统二期 - 候选人画像页", goal="重构候选人画像展示页，提升信息密度",
        owner_id=chenming_user.id, collaborator_ids=[], start_at=days_ago(18), deadline=days_ago(4),
        priority="high", difficulty="medium", level="A", base_points=300,
        acceptance_criteria="页面按设计稿实现，核心信息一屏可见，无P0/P1缺陷",
        acceptor_id=pld.id, deliverables="上线的候选人画像页", department="技术部",
        status="passed", created_by=pld.id, created_at=days_ago(18), updated_at=days_ago(4),
    )
    wt2 = WorkTask(
        name="试用期任务看板性能优化", goal="优化试用期任务看板加载速度",
        owner_id=chenming_user.id, collaborator_ids=[], start_at=days_ago(10), deadline=days_ago(3),
        priority="normal", difficulty="easy", level="B", base_points=120,
        acceptance_criteria="首屏加载时间从2.5s降到1s以内",
        acceptor_id=pld.id, deliverables="优化后的看板页面 + 性能对比报告", department="技术部",
        status="passed", created_by=pld.id, created_at=days_ago(10), updated_at=days_ago(3),
    )
    db.add_all([wt1, wt2])
    await db.flush()

    db.add(TaskAcceptance(
        task_id=wt1.id, acceptor_id=pld.id, completion_coeff=1.1, quality_coeff=1.1,
        timeliness_coeff=1.0, collaboration_points=10, innovation_points=15, penalty_points=0,
        notes="设计还原度高，主动提出了信息分组优化建议。", accepted_at=days_ago(4),
    ))
    db.add(TaskAcceptance(
        task_id=wt2.id, acceptor_id=pld.id, completion_coeff=1.0, quality_coeff=1.05,
        timeliness_coeff=1.1, collaboration_points=5, innovation_points=5, penalty_points=0,
        notes="按时交付，性能提升超预期。", accepted_at=days_ago(3),
    ))

    db.add(PointRecord(
        employee_id=employee.id, task_id=wt1.id, base_points=300, actual_points=380,
        ai_advice={"summary": "完成度与创新分均高于基准，建议按1.27倍系数结算"},
        status="confirmed", source="task", description="人才管理系统二期 - 候选人画像页",
        confirmed_by=pld.id, confirmed_at=days_ago(4), created_at=days_ago(4),
    ))
    db.add(PointRecord(
        employee_id=employee.id, task_id=wt2.id, base_points=120, actual_points=145,
        ai_advice={"summary": "按时交付且质量超出预期"},
        status="confirmed", source="task", description="试用期任务看板性能优化",
        confirmed_by=pld.id, confirmed_at=days_ago(3), created_at=days_ago(3),
    ))
    reward = RewardPenaltyRecord(
        employee_id=employee.id, type="reward", points=50,
        reason="主动提出候选人画像页信息分组优化建议，被团队采纳并推广。",
        rule_ref="创新贡献奖励条款", evidence=[{"type": "task", "ref": wt1.name}],
        requires_dual_approval=False, dual_approved=False,
        employee_notified=True, employee_acknowledged=True, employee_ack_at=days_ago(3),
        status="confirmed", created_by=pld.id, created_at=days_ago(4), confirmed_at=days_ago(3),
    )
    penalty = RewardPenaltyRecord(
        employee_id=employee.id, type="penalty", points=20,
        reason="一次提测未附带联调说明文档，导致测试同学返工。",
        rule_ref="提测规范条款3.2", evidence=[{"type": "task", "ref": wt2.name}],
        requires_dual_approval=False, dual_approved=False,
        employee_notified=True, employee_acknowledged=True, employee_ack_at=days_ago(2),
        status="confirmed", created_by=pld.id, created_at=days_ago(3), confirmed_at=days_ago(2),
    )
    db.add_all([reward, penalty])
    await db.flush()

    db.add(Appeal(
        employee_id=employee.id, target_type="penalty", target_id=penalty.id,
        content="提测时确实遗漏了文档，但联调群里已同步说明，申请酌情减免扣分。",
        status="resolved", handler_id=pld.id,
        resolution="核实群内确有口头说明，扣分维持但不计入本季度考核。",
        resolved_at=days_ago(1), created_at=days_ago(2), updated_at=days_ago(1),
    ))

    if not (await db.execute(select(PerformanceQuarter).where(PerformanceQuarter.quarter == "2026-Q3"))).scalar_one_or_none():
        db.add(PerformanceQuarter(quarter="2026-Q3", status="active", bonus_pool=500000, distributed=125000, initiated_at=days_ago(30)))
    db.add(PerformanceRecord(
        employee_id=employee.id, quarter="2026-Q3", tasks_completed=4, quality=90,
        speed=88, compliance=95, total_score=91, grade="A", bonus=8000, rank=2,
    ))

    db.add(PromotionRecord(
        employee_id=employee.id, from_level="L2", to_level="L3",
        from_position="全栈开发工程师", to_position="高级全栈开发工程师",
        criteria_check={
            "points_met": True, "core_ability_met": True, "stable_delivery": True,
            "no_major_violation": True, "higher_task_capable": True, "review_passed": True,
        },
        all_criteria_met=True,
        ai_recommendation={"summary": "六项条件全部满足，AI 建议批准晋级"},
        review_result="approved", approver_ids=[str(mgr.id), str(ceo.id)],
        status="approved", created_by=hr.id, created_at=days_ago(6), approved_at=days_ago(5),
    ))

    db.add(EquityRecord(
        employee_id=employee.id, long_term_points_score=88, core_project_score=90,
        professional_score=90, collaboration_score=85, responsibility_score=87,
        equity_score=round(88 * 0.30 + 90 * 0.25 + 90 * 0.15 + 85 * 0.15 + 87 * 0.15, 2),
        ai_recommendation={"summary": "综合评分88.15，建议授予期权", "suggested_units": 5000},
        approver_ids=[str(ceo.id), str(mgr.id)], status="approved",
        created_by=hr.id, created_at=days_ago(3), approved_at=days_ago(2),
    ))

    if "人才管理系统二期" in projects:
        db.add(ProjectAssignment(
            project_id=projects["人才管理系统二期"].id, employee_id=employee.id,
            employee_name="陈明", assigned_by="项目负责人", assigned_at=days_ago(18), created_at=days_ago(18),
        ))

    await db.commit()
    return employee


# ══════════════════════════════════════════════════════════════
# 故事线 2：赵敏 —— 已在职正式员工，晋级/期权待审批
# ══════════════════════════════════════════════════════════════

async def seed_zhao_min(db, positions, users, projects):
    hr, mgr, ceo, pld = users["hr01"], users["mgr01"], users["ceo01"], users["pld01"]
    pos = positions.get("数据分析师")

    employee = Employee(
        candidate_id=None, position_id=pos.id if pos else None, name="赵敏", gender="女", age=29,
        department="数据平台部", phone="13800001002", email="zhaomin@example.com",
        status="formal", onboard_date=date_days_ago(400), probation_end_date=date_days_ago(370),
        employee_type="tech", match_level="experienced", current_week=6, total_weeks=6,
        mentor="带教人", mentor_id=users["mtr01"].id, manager="部门负责人", manager_id=mgr.id,
        overall_score=88, risk_level="low",
        created_at=days_ago(400), updated_at=days_ago(1),
    )
    db.add(employee)
    await db.flush()

    zhaomin_user = User(
        username="zhaomin01", password_hash=hash_password("123456"),
        display_name="赵敏", department="数据平台部", employee_id=employee.id,
        must_change_password=False, created_at=days_ago(400),
    )
    db.add(zhaomin_user)
    await db.flush()
    db.add(UserRole(user_id=zhaomin_user.id, role_code="employee"))

    db.add(TalentProfile(
        employee_id=employee.id, current_position="数据分析师", position_level="P2",
        ability_level="L3", skills=["SQL", "Python", "数据建模", "数据可视化"],
        department="数据平台部", task_success_rate=94, on_time_rate=96, first_pass_rate=88,
        total_confirmed_points=1250, task_count=9, reward_count=2, penalty_count=0, rework_rate=6,
        trainable_skills=["机器学习建模"], assignable_tasks=[{"level": "A", "count": 3}],
        can_mentor=True, promotion_readiness=90, talent_risk="normal",
        ai_analysis={"summary": "长期稳定高质量交付，晋级条件已满足，建议审批"},
        created_at=days_ago(400), updated_at=days_ago(1),
    ))

    wt = WorkTask(
        name="数据平台看板体系搭建", goal="搭建全公司统一数据看板体系",
        owner_id=zhaomin_user.id, collaborator_ids=[], start_at=days_ago(60), deadline=days_ago(15),
        priority="high", difficulty="hard", level="A", base_points=350,
        acceptance_criteria="覆盖招聘/试用期/绩效三大域看板，数据日更新",
        acceptor_id=pld.id, deliverables="上线的数据看板", department="数据平台部",
        status="passed", created_by=pld.id, created_at=days_ago(60), updated_at=days_ago(15),
    )
    db.add(wt)
    await db.flush()
    db.add(PointRecord(
        employee_id=employee.id, task_id=wt.id, base_points=350, actual_points=410,
        ai_advice={"summary": "交付质量高，超出预期"}, status="confirmed", source="task",
        description="数据平台看板体系搭建", confirmed_by=pld.id, confirmed_at=days_ago(15),
        created_at=days_ago(15),
    ))

    if not (await db.execute(select(PerformanceRecord).where(
        PerformanceRecord.employee_id == employee.id, PerformanceRecord.quarter == "2026-Q3"
    ))).scalar_one_or_none():
        db.add(PerformanceRecord(
            employee_id=employee.id, quarter="2026-Q3", tasks_completed=9, quality=92,
            speed=90, compliance=96, total_score=93, grade="A", bonus=9000, rank=1,
        ))

    # 晋级 + 期权：都停在"待审批"，供你在 UI 里亲手完成闭环最后一步
    db.add(PromotionRecord(
        employee_id=employee.id, from_level="L3", to_level="L4",
        from_position="数据分析师", to_position="高级数据分析师",
        criteria_check={
            "points_met": True, "core_ability_met": True, "stable_delivery": True,
            "no_major_violation": True, "higher_task_capable": True, "review_passed": False,
        },
        all_criteria_met=False,
        ai_recommendation={"summary": "五项条件已满足，等待最终评审确认"},
        review_result="pending", approver_ids=[str(mgr.id), str(ceo.id)],
        status="pending", created_by=hr.id, created_at=days_ago(3),
    ))
    db.add(EquityRecord(
        employee_id=employee.id, long_term_points_score=90, core_project_score=88,
        professional_score=92, collaboration_score=87, responsibility_score=89,
        equity_score=round(90 * 0.30 + 88 * 0.25 + 92 * 0.15 + 87 * 0.15 + 89 * 0.15, 2),
        ai_recommendation={"summary": "长期贡献分排名靠前，建议授予期权", "suggested_units": 6000},
        approver_ids=[str(ceo.id), str(mgr.id)], status="pending",
        created_by=hr.id, created_at=days_ago(2),
    ))

    await db.commit()


# ══════════════════════════════════════════════════════════════
# 故事线 3：林娜 —— 试用期进行中（第2周）
# ══════════════════════════════════════════════════════════════

async def seed_lin_na(db, positions, users):
    hr, ivr, mgr, mtr = users["hr01"], users["ivr01"], users["mgr01"], users["mtr01"]
    pos = positions["数据开发工程师"]

    candidate = Candidate(
        name="林娜", gender="女", age=25, education="硕士", experience="2年",
        ethnicity="汉族", native_place="江苏南京", phone="13800001003",
        email="linna@example.com", position_id=pos.id, score=79,
        status="hired", resume_file="resumes/demo-linna.pdf",
        interviewer="王强", interview_round="r2",
        upload_time=date_days_ago(30), created_at=days_ago(30), updated_at=days_ago(14),
    )
    db.add(candidate)
    await db.flush()
    db.add_all([CandidateSkill(candidate_id=candidate.id, skill=s) for s in ["Python", "Spark", "Kafka", "SQL"]])
    db.add(CandidateAIAnalysis(
        candidate_id=candidate.id, overall_score=79,
        summary="林娜有2年数据开发经验，硕士学历，基础扎实，实战项目经验略少于资深候选人。",
        position_match="技能栈与数据开发工程师岗位匹配，Spark/Kafka 均有实际使用经验。",
        experience_insight="2年经验集中在数据管道搭建，缺少大规模数据治理经验。",
        recommendation="建议录用，试用期重点培养大规模数据治理能力。",
        keywords=["Python", "Spark", "Kafka", "数据开发"],
        highlights=["硕士学历，基础扎实", "有完整数据管道搭建经验"],
        risks=["大规模数据治理经验较少"],
        dimensions=[{"name": "教育背景", "score": 88}, {"name": "项目经验", "score": 72}, {"name": "工作经验", "score": 75}],
        analyzed_at=days_ago(30),
    ))
    db.add(ResumeScore(
        candidate_id=candidate.id, position_id=pos.id,
        skill_match=20, project_match=15, position_exp=11, achievement=10,
        industry_exp=8, learning=5, stability=5, bonus_skill=5, total=79, grade="B",
        advice={"summary": "7维评分良好，学习能力突出", "highlights": ["学习能力强"], "concerns": ["项目经验偏少"]},
        created_at=days_ago(29), updated_at=days_ago(29),
    ))

    interview_r1 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r1",
        scheduled_at=days_ago(25), interviewer_ids=[str(ivr.id)],
        status="completed", composite_score=78, conclusion="advance",
        created_at=days_ago(25), updated_at=days_ago(24),
    )
    interview_r2 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r2",
        scheduled_at=days_ago(20), interviewer_ids=[str(ivr.id), str(mgr.id)],
        status="completed", practical_score=76, composite_score=80, conclusion="recommend",
        created_at=days_ago(20), updated_at=days_ago(19),
    )
    db.add_all([interview_r1, interview_r2])
    await db.flush()
    db.add(InterviewerScore(
        interview_id=interview_r1.id, interviewer_id=ivr.id,
        dimensions={"技术能力": 78, "沟通表达": 80, "学习能力": 85},
        total=78, comment="基础扎实，学习能力强，建议进入二面。", is_submitted=True,
        created_at=days_ago(24), updated_at=days_ago(24),
    ))

    db.add(OfferApproval(
        candidate_id=candidate.id, position_id=pos.id,
        resume_score=79, r1_score=78, r2_score=80, practical_score=76, team_score=80,
        final_score=78.6,
        ai_advice={"summary": "五维综合分78.6，达到推荐录用线", "result": "recommend"},
        strengths="学习能力强，基础扎实，团队配合度高。",
        capability_gaps="大规模数据治理经验需要在实践中积累。",
        risks_note="无明显风险。", suggested_salary="18K-22K",
        probation_goal="4周内掌握团队数据管道规范，独立完成一个数据同步任务。",
        training_plan="第1周熟悉数据平台架构，第2周开始参与实际任务。",
        mentor_id=mtr.id, conversion_criteria="试用期任务验收合格率≥75%。",
        elimination_criteria="连续两周任务不达标。",
        approver_id=mgr.id, approved_at=days_ago(15),
        ai_result="recommend", result="recommend", status="approved",
        created_at=days_ago(16), updated_at=days_ago(15),
    ))

    employee = Employee(
        candidate_id=candidate.id, position_id=pos.id, name="林娜", gender="女", age=25,
        department="数据平台部", phone="13800001003", email="linna@example.com",
        status="probation", onboard_date=date_days_ago(14), probation_end_date=date_days_ago(-14),
        employee_type="tech", match_level="fresh", current_week=2, total_weeks=4,
        mentor="带教人", mentor_id=mtr.id, manager="部门负责人", manager_id=mgr.id,
        risk_level="low", created_at=days_ago(14), updated_at=days_ago(1),
    )
    db.add(employee)
    await db.flush()

    db.add(ProbationPlan(
        employee_id=employee.id, type="tech", total_weeks=4,
        start_date=date_days_ago(14), end_date=date_days_ago(-14), status="active",
        ai_generated=True,
        weeks=[
            {"week": 1, "title": "平台架构熟悉", "focus": "熟悉数据平台整体架构",
             "goals": ["完成架构文档阅读与环境搭建"], "trainingItems": ["数据平台架构培训"]},
            {"week": 2, "title": "数据同步任务实践", "focus": "参与实际数据同步任务",
             "goals": ["独立完成一个数据同步任务"], "trainingItems": []},
            {"week": 3, "title": "数据质量治理", "focus": "参与数据质量监控建设", "goals": [], "trainingItems": []},
            {"week": 4, "title": "独立交付", "focus": "独立负责一个数据模块", "goals": [], "trainingItems": []},
        ],
        created_by_id=mtr.id, created_at=days_ago(14), updated_at=days_ago(1),
    ))
    db.add(ProbationWeekReview(
        employee_id=employee.id, week_number=1,
        dimensions={"工作态度": 85, "技术能力": 80, "协作沟通": 84},
        total_score=83, comment="第一周状态积极，架构文档阅读认真，环境搭建顺利。",
        reviewer_id=mtr.id, reviewer_name="带教人", reviewed_at=days_ago(7), created_at=days_ago(7),
    ))
    db.add(MentorRecord(
        employee_id=employee.id, week=1,
        training_content="数据平台整体架构培训 + 开发环境搭建",
        mastered_skills=["数据平台架构"], pending_skills=["Airflow 调度"],
        completed_tasks=["环境搭建"], issues=[],
        improvement_plan="第二周开始接触实际数据同步任务。",
        mentor_score=83, employee_confirmed=True, confirmed_at=days_ago(6),
        created_at=days_ago(7), created_by=mtr.id,
    ))
    db.add(ProbationTask(
        employee_id=employee.id, title="开发环境搭建与架构文档学习", week_number=1,
        description="搭建本地数据开发环境，通读平台架构文档", status="passed",
        deadline=date_days_ago(7), objective="完成环境搭建与架构学习",
        assignee="林娜", deliverables="环境搭建记录 + 学习笔记", quality_standard="环境可正常运行",
        test_standard="能跑通示例数据管道", reviewer="带教人", expected_points=15,
        score=83, submitted_at=days_ago(7), reviewed_at=days_ago(6), created_at=days_ago(14),
    ))
    db.add(ProbationTask(
        employee_id=employee.id, title="用户行为数据同步任务", week_number=2,
        description="独立完成用户行为日志到数据仓库的同步任务", status="in_progress",
        deadline=date_days_ago(-1), objective="独立完成数据同步任务",
        assignee="林娜", deliverables="同步任务代码 + 运行文档", quality_standard="数据准确率100%",
        test_standard="连续3天同步无异常", reviewer="带教人", expected_points=25,
        created_at=days_ago(7),
    ))

    await db.commit()


# ══════════════════════════════════════════════════════════════
# 故事线 4：黄强 —— 二面进行中
# ══════════════════════════════════════════════════════════════

async def seed_huang_qiang(db, positions, users):
    ivr, mgr = users["ivr01"], users["mgr01"]
    pos = positions["Agent工程师"]

    candidate = Candidate(
        name="黄强", gender="男", age=30, education="本科", experience="6年",
        ethnicity="汉族", native_place="四川成都", phone="13800001004",
        email="huangqiang@example.com", position_id=pos.id, score=84,
        status="round2", resume_file="resumes/demo-huangqiang.pdf",
        interviewer="王强", interview_round="r2",
        upload_time=date_days_ago(10), created_at=days_ago(10), updated_at=days_ago(1),
    )
    db.add(candidate)
    await db.flush()
    db.add_all([CandidateSkill(candidate_id=candidate.id, skill=s) for s in ["LangChain", "RAG", "Python", "向量数据库"]])
    db.add(CandidateAIAnalysis(
        candidate_id=candidate.id, overall_score=84,
        summary="黄强有6年后端经验，近2年专注 Agent/RAG 方向，技术方向与岗位高度契合。",
        position_match="LangChain/RAG/向量数据库经验直接对口 Agent 工程师岗位。",
        experience_insight="6年经验，近期项目聚焦 Agent 工程化落地，有生产环境经验。",
        recommendation="建议尽快安排二面，重点考察 Agent 工程化落地经验。",
        keywords=["LangChain", "RAG", "Agent", "向量数据库"],
        highlights=["近2年专注 Agent 方向，方向高度契合"],
        risks=["需进一步核实大规模 Agent 系统的稳定性保障经验"],
        dimensions=[{"name": "项目经验", "score": 86}, {"name": "工作经验", "score": 85}, {"name": "技能匹配", "score": 90}],
        analyzed_at=days_ago(10),
    ))
    db.add(ResumeScore(
        candidate_id=candidate.id, position_id=pos.id,
        skill_match=24, project_match=18, position_exp=13, achievement=12,
        industry_exp=9, learning=4, stability=3, bonus_skill=1, total=84, grade="A",
        advice={"summary": "技能匹配度高，AI方向经验对口", "highlights": ["方向高度对口"], "concerns": []},
        created_at=days_ago(9), updated_at=days_ago(9),
    ))

    interview_r1 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r1",
        scheduled_at=days_ago(6), interviewer_ids=[str(ivr.id)],
        status="completed", composite_score=83, conclusion="advance",
        created_at=days_ago(6), updated_at=days_ago(5),
    )
    interview_r2 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r2",
        scheduled_at=NOW + timedelta(hours=3), interviewer_ids=[str(ivr.id), str(mgr.id)],
        status="in_progress", created_at=days_ago(2), updated_at=NOW,
    )
    db.add_all([interview_r1, interview_r2])
    await db.flush()
    db.add(InterviewerScore(
        interview_id=interview_r1.id, interviewer_id=ivr.id,
        dimensions={"技术能力": 84, "沟通表达": 82, "项目经验": 88},
        total=83, comment="Agent 方向经验扎实，推荐进入二面。", is_submitted=True,
        created_at=days_ago(5), updated_at=days_ago(5),
    ))

    await db.commit()


# ══════════════════════════════════════════════════════════════
# 故事线 5：周婷 —— offer 待审批
# ══════════════════════════════════════════════════════════════

async def seed_zhou_ting(db, positions, users):
    ivr, mgr, mtr = users["ivr01"], users["mgr01"], users["mtr01"]
    pos = positions["前沿部署工程师"]

    candidate = Candidate(
        name="周婷", gender="女", age=26, education="硕士", experience="3年",
        ethnicity="汉族", native_place="上海", phone="13800001005",
        email="zhouting@example.com", position_id=pos.id, score=86,
        status="pending_offer", resume_file="resumes/demo-zhouting.pdf",
        interviewer="王强", interview_round="r2",
        upload_time=date_days_ago(15), created_at=days_ago(15), updated_at=days_ago(3),
    )
    db.add(candidate)
    await db.flush()
    db.add_all([CandidateSkill(candidate_id=candidate.id, skill=s) for s in ["Kubernetes", "MLOps", "Python", "Terraform"]])
    db.add(CandidateAIAnalysis(
        candidate_id=candidate.id, overall_score=86,
        summary="周婷硕士学历，3年 MLOps/云原生经验，两轮面试表现稳定优秀。",
        position_match="Kubernetes/MLOps 经验与前沿部署工程师岗位高度匹配。",
        experience_insight="3年经验专注于模型部署与云原生基础设施，项目成果可量化。",
        recommendation="两轮面试综合分85+，建议尽快发起 offer 审批。",
        keywords=["Kubernetes", "MLOps", "云原生"],
        highlights=["两轮面试分数稳定在85以上"], risks=[],
        dimensions=[{"name": "项目经验", "score": 88}, {"name": "工作经验", "score": 85}],
        analyzed_at=days_ago(15),
    ))
    db.add(ResumeScore(
        candidate_id=candidate.id, position_id=pos.id,
        skill_match=24, project_match=19, position_exp=14, achievement=13,
        industry_exp=9, learning=4, stability=2, bonus_skill=1, total=86, grade="A",
        advice={"summary": "8维评分优秀", "highlights": ["技能高度匹配"], "concerns": []},
        created_at=days_ago(14), updated_at=days_ago(14),
    ))

    interview_r1 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r1",
        scheduled_at=days_ago(10), interviewer_ids=[str(ivr.id)],
        status="completed", composite_score=85, conclusion="advance",
        created_at=days_ago(10), updated_at=days_ago(9),
    )
    interview_r2 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r2",
        scheduled_at=days_ago(4), interviewer_ids=[str(ivr.id), str(mgr.id)],
        status="completed", practical_score=87, composite_score=88, conclusion="recommend",
        created_at=days_ago(4), updated_at=days_ago(3),
    )
    db.add_all([interview_r1, interview_r2])
    await db.flush()
    db.add_all([
        InterviewerScore(
            interview_id=interview_r1.id, interviewer_id=ivr.id,
            dimensions={"技术能力": 85, "沟通表达": 84, "项目经验": 87},
            total=85, comment="云原生基础扎实，推荐进入二面。", is_submitted=True,
            created_at=days_ago(9), updated_at=days_ago(9),
        ),
        InterviewerScore(
            interview_id=interview_r2.id, interviewer_id=mgr.id,
            dimensions={"系统设计": 88, "编码能力": 86, "沟通协作": 90},
            total=88, comment="实操环节表现出色，推荐录用。", is_submitted=True,
            created_at=days_ago(3), updated_at=days_ago(3),
        ),
    ])

    # offer 待审批，等你在 UI 里点击「批准」完成闭环
    db.add(OfferApproval(
        candidate_id=candidate.id, position_id=pos.id,
        resume_score=86, r1_score=85, r2_score=88, practical_score=87, team_score=86,
        final_score=86.5,
        ai_advice={"summary": "五维综合分86.5，AI 建议优先录用", "result": "priority"},
        strengths="云原生与 MLOps 经验扎实，两轮面试表现稳定。",
        capability_gaps="大规模集群故障排查经验可进一步积累。",
        risks_note="无明显风险。", suggested_salary="26K-30K",
        probation_goal="4周内独立完成一次模型部署上线。",
        training_plan="第1周熟悉部署流水线，第2-4周独立负责部署任务。",
        mentor_id=mtr.id, conversion_criteria="试用期任务验收合格率≥80%。",
        elimination_criteria="连续两周任务不达标。",
        ai_result="priority", status="pending",
        created_at=days_ago(3), updated_at=days_ago(3),
    ))

    await db.commit()


# ══════════════════════════════════════════════════════════════
# 故事线 6：吴帆 —— 已淘汰（人才库）
# ══════════════════════════════════════════════════════════════

async def seed_wu_fan(db, positions, users):
    ivr = users["ivr01"]
    pos = positions["知识图谱工程师"]

    candidate = Candidate(
        name="吴帆", gender="男", age=24, education="本科", experience="1年",
        ethnicity="汉族", native_place="湖北武汉", phone="13800001006",
        email="wufan@example.com", position_id=pos.id, score=58,
        status="rejected", resume_file="resumes/demo-wufan.pdf",
        interviewer="王强", interview_round="r1",
        upload_time=date_days_ago(20), created_at=days_ago(20), updated_at=days_ago(14),
    )
    db.add(candidate)
    await db.flush()
    db.add_all([CandidateSkill(candidate_id=candidate.id, skill=s) for s in ["Python", "Neo4j"]])
    db.add(CandidateAIAnalysis(
        candidate_id=candidate.id, overall_score=58,
        summary="吴帆应届毕业1年，图数据库相关经验有限，与岗位要求的工程化经验有差距。",
        position_match="仅有基础 Neo4j 使用经验，缺少图谱构建与知识抽取的工程实践。",
        experience_insight="1年经验多为学习性质项目，缺少生产环境项目经验。",
        recommendation="当前阶段与岗位要求差距较大，建议放入人才库，后续有更匹配岗位时再联系。",
        keywords=["Python", "Neo4j", "应届"], highlights=["学习能力尚可"],
        risks=["工程化经验不足", "项目经历缺乏量化成果"],
        dimensions=[{"name": "项目经验", "score": 45}, {"name": "工作经验", "score": 50}],
        analyzed_at=days_ago(20),
    ))
    db.add(ResumeScore(
        candidate_id=candidate.id, position_id=pos.id,
        skill_match=14, project_match=8, position_exp=6, achievement=5,
        industry_exp=4, learning=4, stability=3, bonus_skill=0, total=58, grade="D",
        advice={"summary": "8维评分偏低，工程经验不足", "highlights": [], "concerns": ["图谱工程化经验不足", "缺少量化成果"]},
        created_at=days_ago(19), updated_at=days_ago(19),
    ))

    interview_r1 = Interview(
        candidate_id=candidate.id, position_id=pos.id, round="r1",
        scheduled_at=days_ago(15), interviewer_ids=[str(ivr.id)],
        status="completed", composite_score=55, conclusion="reject",
        created_at=days_ago(15), updated_at=days_ago(14),
    )
    db.add(interview_r1)
    await db.flush()
    db.add(InterviewerScore(
        interview_id=interview_r1.id, interviewer_id=ivr.id,
        dimensions={"技术能力": 55, "沟通表达": 60, "项目经验": 45},
        total=55, comment="基础知识掌握尚可，但缺乏工程化项目经验，暂不匹配当前岗位需求。",
        is_submitted=True, created_at=days_ago(14), updated_at=days_ago(14),
    ))
    db.add(StateTransition(
        entity_type="candidate", entity_id=candidate.id, from_status="round1",
        to_status="rejected", reason="一面综合分55，工程化经验不足",
        actor_id=ivr.id, actor_name="面试官", created_at=days_ago(14),
    ))

    await db.commit()


# ══════════════════════════════════════════════════════════════
# 招聘需求 + 异常队列
# ══════════════════════════════════════════════════════════════

async def seed_recruitment_request(db, departments, users):
    hr, mgr = users["hr01"], users["mgr01"]
    dept = departments.get("AI 平台部")

    existing = await db.execute(select(RecruitmentRequest).where(RecruitmentRequest.position_name == "云原生工程师"))
    if existing.scalar_one_or_none():
        return

    request = RecruitmentRequest(
        position_name="云原生工程师", headcount=2, department_id=dept.id if dept else None,
        work_experience="3年以上云原生/容器化相关经验",
        education_requirement="本科及以上，计算机相关专业",
        job_responsibilities="负责公司 AI 中台的容器化部署与云原生基础设施建设",
        job_description="参与 AI 中台 Kubernetes 集群的运维与优化，保障模型服务的稳定部署",
        job_requirements="熟悉 Kubernetes、Docker，有 CI/CD 流水线搭建经验",
        bonus_items="有 AI/ML 模型部署经验优先",
        core_tasks="Kubernetes 集群运维、CI/CD 流水线建设、模型服务部署优化",
        required_skills=["Kubernetes", "Docker", "CI/CD"],
        preferred_skills=["MLOps", "Terraform"],
        deliverable_req="季度内完成 AI 中台核心服务的容器化改造",
        salary_range="25K-35K", probation_goal="4周内独立完成一次生产环境部署",
        elimination_criteria="连续两周任务不达标",
        interviewer_ids=[str(users["ivr01"].id)],
        direct_manager_id=mgr.id, direct_manager_name="部门负责人",
        submitter_id=hr.id, status="published",
        ai_draft={"summary": "AI 生成的招聘需求草稿，已通过 HR 与部门双确认"},
        hr_confirmed_by=hr.id, hr_confirmed_at=days_ago(8),
        dept_confirmed_by=mgr.id, dept_confirmed_at=days_ago(7),
        created_at=days_ago(9), updated_at=days_ago(7),
    )
    db.add(request)
    await db.flush()

    for from_s, to_s, n in [
        (None, "draft", 9), ("draft", "ai_generated", 9),
        ("ai_generated", "hr_confirmed", 8), ("hr_confirmed", "dept_confirmed", 7),
        ("dept_confirmed", "published", 7),
    ]:
        db.add(StateTransition(
            entity_type="recruitment_request", entity_id=request.id, from_status=from_s,
            to_status=to_s, reason="招聘需求流转", actor_id=hr.id, actor_name="HR 专员",
            created_at=days_ago(n),
        ))

    await db.commit()


async def seed_exceptions(db, users):
    hr = users["hr01"]
    existing = await db.execute(select(ExceptionQueue).where(ExceptionQueue.exception_type == "score_gap"))
    if existing.scalar_one_or_none():
        return
    db.add(ExceptionQueue(
        entity_type="candidate", entity_id=uuid.uuid4(), exception_type="score_gap",
        detail="两位面试官对同一候选人评分差距超过20分，已人工复核确认取平均分。",
        severity="warn", status="handled", handler_id=hr.id, handler_name="HR 专员",
        resolution="复核后确认取平均分，流程继续。", handled_at=days_ago(12), created_at=days_ago(13),
    ))
    db.add(ExceptionQueue(
        entity_type="offer", entity_id=uuid.uuid4(), exception_type="model_uncertain",
        detail="AI 建议结果为 conditional，人工复核后确认为 recommend。",
        severity="warn", status="handled", handler_id=hr.id, handler_name="HR 专员",
        resolution="人工复核后调整为 recommend。", handled_at=days_ago(6), created_at=days_ago(7),
    ))
    await db.commit()


# ══════════════════════════════════════════════════════════════
# 清理旧的占位测试候选人（张三/李四/王五/赵六 —— 之前调试上传功能时产生的占位数据）
# ══════════════════════════════════════════════════════════════

async def cleanup_placeholder_candidates(db):
    names = ["张三", "李四", "王五", "赵六"]
    rows = (await db.execute(select(Candidate).where(Candidate.name.in_(names)))).scalars().all()
    if not rows:
        return
    for c in rows:
        await db.delete(c)
    await db.commit()
    print(f"✓ 已清理 {len(rows)} 条占位测试候选人（{'/'.join(names)}）")


async def main():
    async with async_session_factory() as db:
        if await already_seeded(db):
            print("闭环测试数据已存在（候选人「陈明」已存在），跳过。如需重新生成，请先手动清理相关表。")
            return
        await cleanup_placeholder_candidates(db)

    async with async_session_factory() as db:
        positions, departments, users, projects = await _refs(db)
        required_positions = ["全栈开发工程师", "数据开发工程师", "Agent工程师", "前沿部署工程师", "知识图谱工程师"]
        missing = [p for p in required_positions if p not in positions]
        if missing:
            print(f"[错误] 缺少岗位种子数据: {missing}，请先运行 python setup_db.py 补齐基础种子。")
            return
        required_users = ["hr01", "ivr01", "mgr01", "mtr01", "ceo01", "pld01"]
        missing_users = [u for u in required_users if u not in users]
        if missing_users:
            print(f"[错误] 缺少种子账号: {missing_users}，请先运行 python setup_db.py 补齐基础种子。")
            return

    async with async_session_factory() as db:
        positions, departments, users, projects = await _refs(db)
        print("[1/8] 陈明 —— 旗舰全流程（候选人→录用→试用期→转正→晋级→期权）...")
        await seed_chen_ming(db, positions, users, projects)

    async with async_session_factory() as db:
        positions, departments, users, projects = await _refs(db)
        print("[2/8] 赵敏 —— 在职正式员工，晋级/期权待审批...")
        await seed_zhao_min(db, positions, users, projects)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[3/8] 林娜 —— 候选人已录用，试用期第2周进行中...")
        await seed_lin_na(db, positions, users)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[4/8] 黄强 —— 候选人二面进行中...")
        await seed_huang_qiang(db, positions, users)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[5/8] 周婷 —— 候选人 offer 待审批...")
        await seed_zhou_ting(db, positions, users)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[6/8] 吴帆 —— 候选人已淘汰（人才库）...")
        await seed_wu_fan(db, positions, users)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[7/8] 招聘需求单（云原生工程师，已发布）...")
        await seed_recruitment_request(db, departments, users)

    async with async_session_factory() as db:
        positions, departments, users, _ = await _refs(db)
        print("[8/8] 异常队列历史记录...")
        await seed_exceptions(db, users)

    print("\n✓ 闭环测试数据全部写入完成。")
    print("  可登录账号: chenming01/123456（陈明，全流程正式员工）、zhaomin01/123456（赵敏，待审批晋级/期权）")

    from app.database import engine
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
