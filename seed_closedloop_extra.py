"""补充闭环测试数据里剩下的几张空表：培训进度 / 面试片段评分 / 面试转写 / 岗位胜任力。

interview_media（需要真实音视频文件）和 knowledge_items（不属于招聘生命周期）
不在本次范围内，予以保留为空。

用法：
    python seed_closedloop_extra.py
（依赖 seed_closedloop_demo.py 已先运行过，陈明/全栈开发工程师/Agent工程师等数据已存在）
"""
import asyncio
import sys
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("seed_closedloop_extra.py", 1)[0])

from sqlalchemy import select

from app.database import async_session_factory
from app.models.recruitment import Candidate, Position
from app.models.interview import InterviewTranscript, InterviewSegmentEvaluation
from app.models.phase1 import Interview, PositionCompetency
from app.models.phase2 import TrainingCourse, EmployeeTrainingProgress
from app.models.probation import Employee

NOW = datetime.utcnow()


def days_ago(n: int) -> datetime:
    return NOW - timedelta(days=n)


async def main():
    async with async_session_factory() as db:
        candidate = (await db.execute(select(Candidate).where(Candidate.name == "陈明"))).scalar_one_or_none()
        employee = (await db.execute(select(Employee).where(Employee.name == "陈明"))).scalar_one_or_none()
        if not candidate or not employee:
            print("[错误] 未找到「陈明」的候选人/员工记录，请先运行 python seed_closedloop_demo.py")
            return

        interview_r1 = (await db.execute(
            select(Interview).where(Interview.candidate_id == candidate.id, Interview.round == "r1")
        )).scalar_one_or_none()

        # ── interview_transcripts：一面完整转写 ──
        existing_transcript = (await db.execute(
            select(InterviewTranscript).where(InterviewTranscript.candidate_id == candidate.id, InterviewTranscript.round == "r1")
        )).scalar_one_or_none()
        if not existing_transcript and interview_r1:
            transcript = InterviewTranscript(
                candidate_id=candidate.id, round="r1",
                content=(
                    "面试官：先做个自我介绍吧。\n"
                    "陈明：大家好，我叫陈明，4年全栈开发经验，最近在某电商科技公司负责交易中台的重构，"
                    "主导把单体应用拆分成交易、库存、履约三个微服务，页面加载耗时下降了40%。\n"
                    "面试官：请介绍一下你主导的交易中台重构项目，具体做了哪些技术选型？\n"
                    "陈明：我们把单体拆成了交易、库存、履约三个微服务，用 PostgreSQL 分库分表，"
                    "Node.js 做 BFF 层聚合数据……\n"
                    "面试官：React 中如何做性能优化？\n"
                    "陈明：主要用 React.memo、虚拟列表、代码分割，配合 CDN 缓存静态资源，"
                    "首屏时间从3.2s降到1.8s。\n"
                    "面试官：还有什么想问我们的吗？\n"
                    "陈明：想了解一下团队目前的技术栈演进方向，以及试用期的具体考核标准。"
                ),
                source="upload", filename="chenming_r1_transcript.txt",
                assessment_report={"summary": "回答具体、逻辑清晰，项目经历可验证性强", "risk": "低"},
                process_status="completed", process_progress=100,
                created_at=days_ago(63),
            )
            db.add(transcript)
            await db.flush()

            db.add(InterviewSegmentEvaluation(
                candidate_id=candidate.id, round="r1", transcript_id=transcript.id,
                segment_type="self_intro",
                content="大家好，我叫陈明，4年全栈开发经验，最近在某电商科技公司负责交易中台的重构，"
                        "主导把单体应用拆分成交易、库存、履约三个微服务，页面加载耗时下降了40%。",
                ai_score=88, ai_dimensions={"逻辑清晰度": 88, "重点突出度": 86},
                hr_score=88, hr_dimensions={"逻辑清晰度": 88, "重点突出度": 88},
                status="confirmed", updated_at=days_ago(62),
            ))
            db.add(InterviewSegmentEvaluation(
                candidate_id=candidate.id, round="r1", transcript_id=transcript.id,
                segment_type="reverse_question",
                content="想了解一下团队目前的技术栈演进方向，以及试用期的具体考核标准。",
                ai_score=82, ai_dimensions={"问题质量": 82, "求职诚意": 85},
                hr_score=84, hr_dimensions={"问题质量": 84, "求职诚意": 86},
                status="confirmed", updated_at=days_ago(62),
            ))
            print("✓ 已补充陈明一面转写记录 + 自我介绍/反问环节评分")
        else:
            print("○ 陈明一面转写记录已存在，跳过")

        # ── employee_training_progress：入职5天培训进度 ──
        existing_progress = (await db.execute(
            select(EmployeeTrainingProgress).where(EmployeeTrainingProgress.employee_id == employee.id)
        )).scalar_one_or_none()
        if not existing_progress:
            courses = (await db.execute(select(TrainingCourse).order_by(TrainingCourse.day))).scalars().all()
            db.add(EmployeeTrainingProgress(
                employee_id=employee.id,
                courses=[{"courseId": str(c.id), "completed": True, "score": 88 + i} for i, c in enumerate(courses)],
                overall_rate=100, completed_at=days_ago(50), created_at=days_ago(54), updated_at=days_ago(50),
            ))
            print("✓ 已补充陈明入职5天培训进度（全部完成）")
        else:
            print("○ 陈明培训进度已存在，跳过")

        # ── position_competency：给两个岗位补胜任力维度权重 ──
        dims = [
            ("专业技能", 30), ("项目经验", 20), ("学习能力", 15),
            ("沟通协作", 15), ("责任心", 10), ("文化契合度", 10),
        ]
        for pos_name in ["全栈开发工程师", "Agent工程师"]:
            pos = (await db.execute(select(Position).where(Position.name == pos_name))).scalar_one_or_none()
            if not pos:
                continue
            existing_comp = (await db.execute(
                select(PositionCompetency).where(PositionCompetency.position_id == pos.id)
            )).scalars().all()
            if existing_comp:
                print(f"○ {pos_name} 胜任力维度已存在，跳过")
                continue
            for dim, weight in dims:
                db.add(PositionCompetency(position_id=pos.id, dimension=dim, weight=weight, locked=True))
            print(f"✓ 已补充「{pos_name}」的6项胜任力维度权重（合计100%）")

        await db.commit()

    from app.database import engine
    await engine.dispose()
    print("\n✓ 补充数据写入完成。")


if __name__ == "__main__":
    asyncio.run(main())
