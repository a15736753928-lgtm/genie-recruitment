from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models.candidate import Candidate, Position
from app.models.interview import InterviewEvaluation, InterviewQuestion
from app.models.probation import Employee
from app.models.performance import PerformanceRecord

router = APIRouter(tags=["看板"])


@router.get("/dashboard/overview")
async def get_overview(db: AsyncSession = Depends(get_db)):
    # Stats
    total_resumes = (await db.execute(select(func.count()).select_from(Candidate))).scalar() or 0
    total_positions = (await db.execute(select(func.count()).select_from(Position))).scalar() or 0
    total_interviews = (await db.execute(
        select(func.count()).select_from(Candidate).where(
            Candidate.status.in_(["passed", "first_interview", "second_interview"])
        )
    )).scalar() or 0
    total_employees = (await db.execute(select(func.count()).select_from(Employee))).scalar() or 0

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "kpis": [
                {"key": "resumes", "label": "简历总量", "value": total_resumes, "hint": "累计接收", "hintTone": "up"},
                {"key": "positions", "label": "在招岗位", "value": total_positions, "hint": "持续招聘中", "hintTone": "default"},
                {"key": "interviews", "label": "面试进行中", "value": total_interviews, "hint": "当前在流程", "hintTone": "default"},
                {"key": "employees", "label": "在职员工", "value": total_employees, "hint": "包含试用期", "hintTone": "up"},
            ],
            "funnel": [],
            "monthlyTrend": [],
            "positionDistribution": [],
            "statusDistribution": [],
            "departmentPerformance": [],
            "conversionRates": [],
            "topCandidates": [],
            "recentActivities": [],
        },
    }


@router.get("/dashboard/operations")
async def get_operations(db: AsyncSession = Depends(get_db)):
    # Build operations dashboard from real data
    total_resumes = (await db.execute(select(func.count()).select_from(Candidate))).scalar() or 0
    total_employees = (await db.execute(select(func.count()).select_from(Employee))).scalar() or 0

    # Status counts
    job_hunting = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "job_hunting")
    )).scalar() or 0
    passed_count = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "passed")
    )).scalar() or 0
    first_interview = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "first_interview")
    )).scalar() or 0
    second = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "second_interview")
    )).scalar() or 0
    # 兼容历史状态
    legacy_passed = (await db.execute(
        select(func.count()).select_from(Candidate).where(
            Candidate.status.in_(["pending_interview", "probation", "onboarded"])
        )
    )).scalar() or 0
    passed = passed_count + legacy_passed
    failed = (await db.execute(
        select(func.count()).select_from(Candidate).where(Candidate.status == "failed")
    )).scalar() or 0

    avg_score_result = (await db.execute(select(func.avg(Candidate.score)).select_from(Candidate)))
    avg_score = round(float(avg_score_result.scalar() or 0), 1)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "summary": {
                "title": "招聘运营概览",
                "text": f"当前系统共有 {total_resumes} 份简历，{total_employees} 名在职员工。候选人平均匹配度 {avg_score} 分。",
                "stats": [
                    {"label": "候选人总数", "value": str(total_resumes), "meta": "累计入库", "tone": "blue"},
                    {"label": "已通过", "value": str(passed), "meta": "初筛通过", "tone": "green"},
                    {"label": "一面中", "value": str(first_interview), "meta": "面试进行中", "tone": "blue"},
                    {"label": "二面中", "value": str(second), "meta": "深度评估", "tone": "purple"},
                ],
            },
            "metrics": [
                {"label": "简历总量", "value": str(total_resumes), "meta": "份", "tone": "blue"},
                {"label": "平均匹配度", "value": str(avg_score), "meta": "/100", "tone": "purple"},
                {"label": "已通过", "value": str(passed), "meta": "人", "tone": "green"},
                {"label": "未通过", "value": str(failed), "meta": "人", "tone": "red"},
            ],
            "risks": [
                {
                    "role": "招聘专员",
                    "stage": "简历筛选",
                    "reason": f"有 {job_hunting} 份简历待处理",
                    "advice": "建议尽快筛选并安排面试",
                    "actions": ["批量筛选", "AI 自动匹配"],
                },
            ] if job_hunting > 10 else [],
            "funnel": [
                {"name": "求职中", "value": job_hunting, "ratio": round(job_hunting / max(total_resumes, 1) * 100)},
                {"name": "已通过", "value": passed, "ratio": round(passed / max(total_resumes, 1) * 100)},
                {"name": "一面中", "value": first_interview, "ratio": round(first_interview / max(total_resumes, 1) * 100)},
                {"name": "二面中", "value": second, "ratio": round(second / max(total_resumes, 1) * 100)},
            ],
            "insights": [
                {
                    "title": "招聘漏斗分析",
                    "reason": f"从求职到通过的转化率为 {round(passed / max(total_resumes, 1) * 100)}%",
                    "advice": "关注各阶段流失率，优化面试流程",
                    "actions": ["查看详细漏斗", "优化筛选策略"],
                },
            ],
            "history": [
                {"title": "系统启动", "meta": "招聘系统初始化完成", "actions": ["查看详情"]},
            ],
            "capabilities": [
                {"code": "resume_parse", "name": "AI 简历解析", "status": "正常", "meta": "DeepSeek"},
                {"code": "question_gen", "name": "AI 出题", "status": "正常", "meta": "DeepSeek"},
                {"code": "interview_score", "name": "AI 面试评分", "status": "正常", "meta": "DeepSeek"},
                {"code": "rag", "name": "RAG 知识检索", "status": "正常", "meta": "Milvus Lite"},
            ],
        },
    }
