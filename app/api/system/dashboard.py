from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models.recruitment import Candidate, Position
from app.models.interview import InterviewEvaluation, InterviewQuestion
from app.models.probation import Employee
from app.models.performance import PerformanceRecord
from app.utils.responses import ok

router = APIRouter(tags=["看板"])


async def query_recruitment_summary(db: AsyncSession) -> dict:
    """汇总候选人/岗位等核心计数，供 HTTP 路由与 Agent 工具直接调用。"""
    total_resumes = (await db.execute(select(func.count()).select_from(Candidate))).scalar() or 0
    total_positions = (await db.execute(select(func.count()).select_from(Position))).scalar() or 0
    total_employees = (await db.execute(select(func.count()).select_from(Employee))).scalar() or 0

    # 按状态机词表逐状态统计（2026-08-02 起按新业务词表：求职中/一面/二面/待发offer/已录用/未通过/已失效）。
    rows = (await db.execute(
        select(Candidate.status, func.count()).group_by(Candidate.status)
    )).all()
    by_status = {s or "unknown": int(n) for s, n in rows}

    def n(*statuses: str) -> int:
        return sum(by_status.get(s, 0) for s in statuses)

    job_hunting = n("job_hunting")              # 求职中（含原 new/parsed/pending_screen/pending_materials）
    round1 = n("round1")                         # 一面中
    round2 = n("round2")                         # 二面中
    pending_offer = n("pending_offer")           # 待发offer
    hired = n("hired")                           # 已录用
    talent_pool = n("talent_pool")               # 已失效（人才池留档）
    rejected = n("rejected")                     # 未通过

    # 「在面试流程中」= 一面 + 二面
    total_interviews = round1 + round2

    avg_score_result = (await db.execute(select(func.avg(Candidate.score)).select_from(Candidate)))
    avg_score = round(float(avg_score_result.scalar() or 0), 1)

    return {
        "totalCandidates": int(total_resumes),
        "totalPositions": int(total_positions),
        "totalEmployees": int(total_employees),
        "totalInterviews": int(total_interviews),
        "jobHunting": job_hunting,
        "round1": round1,
        "round2": round2,
        "pendingOffer": pending_offer,
        "hired": hired,
        "talentPool": talent_pool,
        "rejected": rejected,
        "byStatus": by_status,
        "avgScore": avg_score,
    }


@router.get("/dashboard/overview")
async def get_overview(db: AsyncSession = Depends(get_db)):
    summary = await query_recruitment_summary(db)
    total_resumes = summary["totalCandidates"]
    total_positions = summary["totalPositions"]
    total_interviews = summary["totalInterviews"]
    total_employees = summary["totalEmployees"]

    return ok({
          "stats": {
              "totalResumes": total_resumes,
              "totalPositions": total_positions,
              "totalInterviews": total_interviews,
              "totalEmployees": total_employees,
          },
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
      })


@router.get("/dashboard/operations")
async def get_operations(db: AsyncSession = Depends(get_db)):
    summary = await query_recruitment_summary(db)
    total_resumes = summary["totalCandidates"]
    total_employees = summary["totalEmployees"]
    job_hunting = summary["jobHunting"]
    round1 = summary["round1"]
    round2 = summary["round2"]
    pending_offer = summary["pendingOffer"]
    hired = summary["hired"]
    rejected = summary["rejected"]
    avg_score = summary["avgScore"]

    return ok({
          "summary": {
              "title": "招聘运营概览",
              "text": f"当前系统共有 {total_resumes} 份简历，{total_employees} 名在职员工。候选人平均匹配度 {avg_score} 分。",
              "stats": [
                  {"label": "候选人总数", "value": str(total_resumes), "meta": "累计入库", "tone": "blue"},
                  {"label": "求职中", "value": str(job_hunting), "meta": "待筛选", "tone": "green"},
                  {"label": "一面中", "value": str(round1), "meta": "面试进行中", "tone": "blue"},
                  {"label": "二面中", "value": str(round2), "meta": "深度评估", "tone": "purple"},
              ],
          },
          "metrics": [
              {"label": "简历总量", "value": str(total_resumes), "meta": "份", "tone": "blue"},
              {"label": "平均匹配度", "value": str(avg_score), "meta": "/100", "tone": "purple"},
              {"label": "待发 Offer", "value": str(pending_offer), "meta": "人", "tone": "green"},
              {"label": "已录用", "value": str(hired), "meta": "人", "tone": "green"},
              {"label": "不予录用", "value": str(rejected), "meta": "人", "tone": "red"},
          ],
          "risks": [
              {
                  "role": "招聘专员",
                  "stage": "简历筛选",
                  "reason": f"有 {job_hunting} 份简历在求职中待筛选",
                  "advice": "建议尽快筛选并安排一面",
                  "actions": ["批量筛选", "AI 自动匹配"],
              },
          ] if job_hunting > 10 else [],
          "funnel": [
              {"name": "求职中", "value": job_hunting, "ratio": round(job_hunting / max(total_resumes, 1) * 100)},
              {"name": "一面中", "value": round1, "ratio": round(round1 / max(total_resumes, 1) * 100)},
              {"name": "二面中", "value": round2, "ratio": round(round2 / max(total_resumes, 1) * 100)},
              {"name": "待发 Offer", "value": pending_offer, "ratio": round(pending_offer / max(total_resumes, 1) * 100)},
              {"name": "已录用", "value": hired, "ratio": round(hired / max(total_resumes, 1) * 100)},
          ],
          "insights": [
              {
                  "title": "招聘漏斗分析",
                  "reason": f"简历到录用的整体转化率为 {round(hired / max(total_resumes, 1) * 100)}%",
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
      })
