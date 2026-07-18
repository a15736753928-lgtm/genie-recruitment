import json
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.database import get_db, async_session_factory
from app.models.agent_session import AgentSession, AgentMessage, AgentMaterial, AgentTask
from app.agent.tools import create_langchain_tools
from app.agent.graph import build_agent_graph, stream_agent_response, AgentResult
from app.config import get_settings
from app.infrastructure import minio_storage
from app.api.recruitment.resumes import extract_text_from_file, parse_resume_with_llm
import os

router = APIRouter(tags=["AI Agent"])
settings = get_settings()

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=60.0,
    max_retries=0,
)

# Agent configurations
# The system now exposes a single omnipotent "Genie" agent that has access to
# ALL tools (every business endpoint). The legacy 4 sub-agents are kept only as
# @-mention routing targets for backward compatibility.
AGENT_CONFIGS = {
    "genie": {
        "name": "Genie 全能助手",
        "description": "通过对话框完成招聘系统所有操作：简历筛选、面试出题、面试评定、试用期考核、绩效管理、知识库管理、系统设置",
        "icon": "search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "recruit": {
        "name": "招聘 Agent",
        "description": "负责简历解析、人才筛选、岗位匹配",
        "icon": "search",
        "iconBg": "#e8f4fd",
        "iconColor": "#2196f3",
    },
    "interview": {
        "name": "面试 Agent",
        "description": "AI 出题、面试分析、录用建议",
        "icon": "interview",
        "iconBg": "#fce4ec",
        "iconColor": "#e91e63",
    },
    "training": {
        "name": "培训 Agent",
        "description": "入职培养、试用期跟踪",
        "icon": "training",
        "iconBg": "#e8f5e9",
        "iconColor": "#4caf50",
    },
    "performance": {
        "name": "绩效 Agent",
        "description": "绩效分析、员工成长建议",
        "icon": "performance",
        "iconBg": "#fff3e0",
        "iconColor": "#ff9800",
    },
}


# ── SSE Helpers ─────────────────────────────────────────

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def execute_tool_call(tool_name: str, params: dict, db: AsyncSession) -> str:
    """Execute a tool and return a human-readable result summary."""
    try:
        if tool_name == "list_resumes":
            from app.api.recruitment.resumes import list_resumes as fn
            pos_id = params.get("positionId", "all")
            statuses = params.get("statuses", "")
            keyword = params.get("keyword", "")
            limit = params.get("limit", 10)
            # minScore 必须显式传 None：FastAPI 的 Query(None) 默认值在
            # 直接 Python 调用时不会解析，导致 Query 对象被注入 SQL 引发
            # "Query(None) cannot be interpreted as an integer" 错误。
            result = await fn(
                positionId=pos_id, statuses=statuses, keyword=keyword,
                sortBy="uploadTime", sortOrder="desc",
                page=1, pageSize=limit, minScore=None, db=db,
            )
            data = result["data"]
            if isinstance(data, dict) and "list" in data:
                candidates = data["list"]
                total = data["total"]
                if not candidates:
                    return f"未找到符合条件的候选人（共 {total} 位候选人）"
                lines = [f"共 {total} 位候选人，以下是前 {len(candidates)} 位："]
                for c in candidates:
                    skills = ", ".join(c.get("skills", [])[:5]) or "无"
                    lines.append(
                        f"  [{c['id']}] {c['name']} | {c['position']} | "
                        f"匹配度 {c['score']} 分 | 状态: {c['status']} | "
                        f"技能: {skills}"
                    )
                return "\n".join(lines)
            return f"查询结果：{json.dumps(data, ensure_ascii=False)[:500]}"

        elif tool_name == "get_resume":
            from app.api.recruitment.resumes import get_resume as fn
            result = await fn(resume_id=params["id"], db=db)
            data = result.get("data")
            if data:
                ai = data.get("aiAnalysis") or {}
                lines = [
                    f"候选人「{data['name']}」(ID: {data['id']})",
                    f"  岗位: {data['position']}({data.get('positionId','')}) | 匹配度: {data['score']} 分 | 状态: {data['status']}",
                    f"  性别: {data.get('gender','未知')} | 年龄: {data.get('age') or '未知'} | 民族: {data.get('ethnicity','未知')} | 籍贯: {data.get('nativePlace','未知')}",
                    f"  学历: {data.get('education','未知')} | 经验: {data.get('experience','未知')}",
                    f"  电话: {data.get('phone','未知')} | 邮箱: {data.get('email','未知')}",
                    f"  技能: {', '.join(data.get('skills', [])) or '无'}",
                ]
                if data.get("educationHistory"):
                    lines.append("  教育经历:")
                    for e in data["educationHistory"]:
                        lines.append(f"    - {e.get('school','')} | {e.get('degree','')} | {e.get('major','')} | {e.get('period','')}")
                if data.get("workHistory"):
                    lines.append("  工作经历:")
                    for w in data["workHistory"]:
                        lines.append(f"    - {w.get('company','')} | {w.get('role','')} | {w.get('period','')}")
                        if w.get('description'):
                            lines.append(f"      描述: {w['description'][:200]}")
                if data.get("projectHistory"):
                    lines.append("  项目经历:")
                    for p in data["projectHistory"]:
                        lines.append(f"    - {p.get('name','')} | {p.get('role','')} | {p.get('period','')}")
                        if p.get('description'):
                            lines.append(f"      描述: {p['description'][:200]}")
                if ai:
                    lines.append(f"  AI综合分析:")
                    if ai.get("summary"):
                        lines.append(f"    总结: {ai['summary']}")
                    if ai.get("overallScore"):
                        lines.append(f"    综合分: {ai['overallScore']}")
                    if ai.get("positionMatch"):
                        lines.append(f"    岗位匹配: {ai['positionMatch']}")
                    if ai.get("experienceInsight"):
                        lines.append(f"    经验洞察: {ai['experienceInsight']}")
                    if ai.get("keywords"):
                        lines.append(f"    关键词: {', '.join(ai['keywords'])}")
                    if ai.get("highlights"):
                        lines.append(f"    亮点: {'; '.join(ai['highlights'])}")
                    if ai.get("risks"):
                        lines.append(f"    风险: {'; '.join(ai['risks'])}")
                    if ai.get("recommendation"):
                        lines.append(f"    建议: {ai['recommendation']}")
                    if ai.get("dimensions"):
                        lines.append(f"    维度评分:")
                        for d in ai["dimensions"]:
                            lines.append(f"      - {d.get('name','')}: {d.get('score',0)}分 ({d.get('comment','')[:200] if d.get('comment') else ''})")
                return "\n".join(lines)
            return "候选人不存在"

        elif tool_name == "list_positions":
            from app.api.recruitment.positions import list_positions as fn
            result = await fn(db=db)
            data = result["data"]
            lines = [f"共 {len(data)} 个岗位："]
            for p in data:
                lines.append(f"  [{p['id']}] {p['name']} | 部门: {p.get('department', '未设置')}")
            return "\n".join(lines)

        elif tool_name == "update_resume":
            from app.api.recruitment.resumes import update_resume as fn
            fields = params.get("fields", {})
            if "status" in params:
                fields["status"] = params["status"]
            result = await fn(resume_id=params["id"], body=fields, db=db)
            if result["code"] == 0:
                return f"已成功更新候选人 {params['id']} 的信息"
            return f"更新失败：{result['message']}"

        elif tool_name == "upload_resume":
            # fileKey is a MinIO object key uploaded by the frontend "添加资料" flow.
            from app.api.recruitment.resumes import upload_resume as fn
            from fastapi import UploadFile
            import io
            object_key = params["fileKey"]
            file_name = params["fileName"]
            position_id = params["positionId"]
            tmp_path = await asyncio.to_thread(minio_storage.download_to_temp, object_key)
            try:
                with open(tmp_path, "rb") as f:
                    content = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            upload_file = UploadFile(file=io.BytesIO(content), filename=file_name)
            result = await fn(file=upload_file, positionId=position_id, db=db)
            if result["code"] == 0:
                d = result.get("data", {})
                return f"已上传简历「{d.get('name', file_name)}」，匹配度 {d.get('score', 0)} 分，状态 {d.get('status', '')}。{result.get('message', '')}"
            return f"上传失败：{result.get('message', '')}"

        elif tool_name == "batch_parse_resumes":
            from app.api.recruitment.resumes import batch_parse as fn
            result = await fn(body={"ids": params["ids"]}, db=db)
            return f"已批量解析 {len(params['ids'])} 位候选人" if result["code"] == 0 else f"批量解析失败：{result.get('message', '')}"

        elif tool_name == "reanalyze_resume":
            from app.api.recruitment.resumes import reanalyze_resume as fn
            result = await fn(resume_id=params["id"], db=db)
            if result["code"] == 0:
                d = result.get("data", {})
                return f"已重新解析「{d.get('name', '')}」，新匹配度 {d.get('score', 0)} 分"
            return f"重新解析失败：{result.get('message', '')}"

        elif tool_name == "delete_resume":
            from app.api.recruitment.resumes import delete_resume as fn
            result = await fn(resume_id=params["id"], db=db)
            return f"已删除候选人 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"

        elif tool_name == "get_position":
            from app.api.recruitment.positions import get_position as fn
            result = await fn(position_id=params["id"], db=db)
            d = result.get("data") or {}
            if not d:
                return "岗位不存在"
            lines = [
                f"岗位「{d.get('name', '')}」(ID: {d.get('id', '')})",
                f"  部门: {d.get('department', '未设置')} | 章节数: {d.get('chapterNumber', '未设置')}",
            ]
            if d.get("jdResponsibilities"):
                lines.append(f"  岗位职责: {d['jdResponsibilities']}")
            if d.get("jdRequirements"):
                lines.append(f"  任职要求: {d['jdRequirements']}")
            if d.get("jdPreferred"):
                lines.append(f"  加分项: {d['jdPreferred']}")
            if d.get("jdTechStack"):
                lines.append(f"  技术栈: {d['jdTechStack']}")
            if d.get("screeningCriteria"):
                lines.append(f"  筛选标准: {json.dumps(d['screeningCriteria'], ensure_ascii=False)}")
            if d.get("interviewCriteriaR1"):
                lines.append(f"  一面标准: {json.dumps(d['interviewCriteriaR1'], ensure_ascii=False)}")
            if d.get("interviewCriteriaR2"):
                lines.append(f"  二面标准: {json.dumps(d['interviewCriteriaR2'], ensure_ascii=False)}")
            return "\n".join(lines)

        elif tool_name == "create_position":
            from app.api.recruitment.positions import create_position as fn, CreatePositionRequest
            req = CreatePositionRequest(**params)
            result = await fn(req=req, db=db)
            if result["code"] == 0:
                return f"已创建岗位「{params['name']}」"
            return f"创建失败：{result.get('message', '')}"

        elif tool_name == "update_position":
            from app.api.recruitment.positions import update_position as fn, UpdatePositionRequest
            req = UpdatePositionRequest(**params.get("fields", {}))
            result = await fn(position_id=params["id"], req=req, db=db)
            return f"已更新岗位 {params['id']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        elif tool_name == "delete_position":
            from app.api.recruitment.positions import delete_position as fn
            result = await fn(position_id=params["id"], db=db)
            return f"已删除岗位 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"

        elif tool_name == "get_position_questions":
            from app.api.recruitment.positions import get_position_questions as fn
            result = await fn(position_id=params["positionId"], round=params["round"], db=db)
            qs = result.get("data", [])
            return f"岗位题库（{params['round']}）共 {len(qs)} 道题"

        elif tool_name == "save_position_questions":
            from app.api.recruitment.positions import save_position_questions as fn
            result = await fn(position_id=params["positionId"], body={"round": params["round"], "questions": params["questions"]}, db=db)
            return f"已保存岗位题库 {len(params['questions'])} 道题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"

        elif tool_name == "get_questions":
            from app.api.talent.interview import get_questions as fn
            result = await fn(candidateId=params["candidateId"], round=params["round"], db=db)
            questions = result.get("data", [])
            return f"共 {len(questions)} 道题"

        elif tool_name == "generate_questions":
            from app.api.talent.interview import regenerate_questions as fn
            result = await fn(body=params, db=db)
            questions = result.get("data", [])
            return f"已生成 {len(questions)} 道新题目"

        elif tool_name == "get_evaluation":
            from app.api.talent.interview import get_evaluation as fn
            result = await fn(candidate_id=params["candidateId"], round=params.get("round", "first"), db=db)
            scores = result.get("data", [])
            scored = [s for s in scores if s.get("hrScore") is not None]
            return f"面试评分：{len(scored)}/{len(scores)} 题已评分"

        elif tool_name == "ai_score_question":
            from app.api.talent.interview import ai_score_question as fn
            result = await fn(question_id=params["questionId"], body={"answer": params.get("answer", "")}, db=db)
            data = result.get("data", {})
            return f"AI评分：{data.get('score', 0)} 分"

        elif tool_name == "get_leaderboard":
            from app.api.talent.interview import get_leaderboard as fn
            result = await fn(category=params["category"], db=db)
            data = result.get("data", [])
            return f"排行榜共 {len(data)} 人"

        elif tool_name == "save_questions":
            from app.api.talent.interview import save_questions as fn
            result = await fn(body={"candidateId": params["candidateId"], "round": params["round"], "questions": params["questions"]}, db=db)
            return f"已保存 {len(params['questions'])} 道面试题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"

        elif tool_name == "replace_question":
            from app.api.talent.interview import replace_question as fn
            result = await fn(question_id=params["questionId"], body=params, db=db)
            qs = result.get("data", [])
            return f"已替换题目，当前题单共 {len(qs)} 道" if result["code"] == 0 else f"替换失败：{result.get('message', '')}"

        elif tool_name == "save_evaluation":
            from app.api.talent.interview import save_evaluation as fn
            result = await fn(candidate_id=params["candidateId"], body={"round": params["round"], "scores": params["scores"]}, db=db)
            return f"已保存 {len(params['scores'])} 题评分" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"

        elif tool_name == "submit_evaluation":
            from app.api.talent.interview import submit_evaluation as fn
            result = await fn(candidate_id=params["candidateId"], db=db)
            return f"已提交候选人 {params['candidateId']} 的面试评定" if result["code"] == 0 else f"提交失败：{result.get('message', '')}"

        elif tool_name == "get_rankings":
            from app.api.talent.interview import get_rankings as fn
            result = await fn(candidateId=params["candidateId"], db=db)
            data = result.get("data", [])
            if not data:
                return "暂无同岗位排名数据"
            lines = [f"同岗位排名共 {len(data)} 人："]
            for r in data[:20]:
                lines.append(
                    f"  #{r.get('rank','?')} {r.get('name','?')} | "
                    f"匹配度 {r.get('score',0)} 分 | 状态: {r.get('status','?')}"
                )
            return "\n".join(lines)

        elif tool_name == "list_probation":
            from app.api.talent.probation import list_probation as fn
            result = await fn(department=params.get("department", "all"), status=params.get("status", "all"), db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"试用期员工：共 {data.get('total', 0)} 人"
            return "查询完成"

        elif tool_name == "get_probation_stats":
            from app.api.talent.probation import get_probation_stats as fn
            result = await fn(db=db)
            d = result.get("data", {})
            return f"试用期统计：总 {d.get('total', 0)}，考核中 {d.get('assessing', 0)}，通过 {d.get('passed', 0)}，未通过 {d.get('failed', 0)}"

        elif tool_name == "get_probation_employee":
            from app.api.talent.probation import get_probation_employee as fn
            result = await fn(employee_id=params["id"], db=db)
            d = result.get("data") or {}
            return f"员工「{d.get('name', '')}」- {d.get('positionName', '')}，状态 {d.get('status', '')}，任务进度 {d.get('taskProgress', 0)}%"

        elif tool_name == "create_probation_employee":
            from app.api.talent.probation import create_employee as fn, CreateEmployeeRequest
            req = CreateEmployeeRequest(**params)
            result = await fn(req=req, db=db)
            if result["code"] == 0:
                return f"已新增试用期员工「{params['name']}」"
            return f"新增失败：{result.get('message', '')}"

        elif tool_name == "save_week1_assessment":
            from app.api.talent.probation import save_week1_assessment as fn, Week1AssessmentRequest
            req = Week1AssessmentRequest(**params)
            result = await fn(employee_id=params["employeeId"], req=req, db=db)
            if result["code"] == 0:
                d = result.get("data", {})
                verdict = "通过" if d.get("passed") else "未通过"
                return f"第一周评估已保存，总分 {d.get('totalScore', 0)}，{verdict}"
            return f"保存失败：{result.get('message', '')}"

        elif tool_name == "save_conversion":
            from app.api.talent.probation import save_conversion as fn, ConversionRequest
            req = ConversionRequest(**params)
            result = await fn(employee_id=params["employeeId"], req=req, db=db)
            if result["code"] == 0:
                d = result.get("data", {})
                decision_map = {"converted": "转正", "extended": "延长试用期", "rejected": "不通过"}
                return f"转正评估已保存，总分 {d.get('totalScore', 0)}，结论：{decision_map.get(d.get('decision', ''), d.get('decision', ''))}"
            return f"保存失败：{result.get('message', '')}"

        elif tool_name == "create_probation_task":
            from app.api.talent.probation import create_probation_task as fn
            result = await fn(body=params, db=db)
            return f"已为员工 {params['employeeId']} 新增任务「{params.get('title', '')}」" if result["code"] == 0 else f"新增失败：{result.get('message', '')}"

        elif tool_name == "update_probation_task":
            from app.api.talent.probation import update_probation_task as fn
            result = await fn(task_id=params["taskId"], body=params.get("fields", {}), db=db)
            return f"已更新任务 {params['taskId']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        elif tool_name == "ai_evaluate_probation":
            from app.api.talent.probation import ai_evaluate_probation as fn
            result = await fn(employee_id=params["employeeId"], db=db)
            if result["code"] == 0:
                d = result.get("data", {})
                return f"AI评估完成：综合分 {d.get('score', 0)}，结论 {d.get('result', '')}。{d.get('comment', '')}"
            return f"AI评估失败：{result.get('message', '')}"

        elif tool_name == "update_probation_status":
            from app.api.talent.probation import update_probation_status as fn
            result = await fn(employee_id=params["employeeId"], body={"status": params["status"]}, db=db)
            return f"已更新员工 {params['employeeId']} 状态为 {params['status']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        elif tool_name == "manual_review_probation":
            from app.api.talent.probation import manual_review as fn
            result = await fn(employee_id=params["employeeId"], body={"aiScore": params.get("aiScore"), "aiResult": params.get("aiResult")}, db=db)
            return f"已手动录入员工 {params['employeeId']} 的评估" if result["code"] == 0 else f"录入失败：{result.get('message', '')}"

        elif tool_name == "list_performance":
            from app.api.talent.performance import list_performance as fn
            result = await fn(quarter=params["quarter"], db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"绩效数据：共 {data.get('total', 0)} 条记录"
            return "查询完成"

        elif tool_name == "get_performance_stats":
            from app.api.talent.performance import get_performance_stats as fn
            result = await fn(quarter=params["quarter"], db=db)
            d = result.get("data", {})
            return f"绩效统计：参与 {d.get('participants', 0)} 人，平均分 {d.get('avgScore', 0)}，优秀 {d.get('excellentCount', 0)}，待改进 {d.get('needsImprovement', 0)}"

        elif tool_name == "get_department_performance":
            from app.api.talent.performance import get_department_performance as fn
            result = await fn(quarter=params["quarter"], db=db)
            data = result.get("data", [])
            return f"部门绩效共 {len(data)} 个部门"

        elif tool_name == "get_grade_distribution":
            from app.api.talent.performance import get_grade_distribution as fn
            result = await fn(quarter=params["quarter"], db=db)
            data = result.get("data", [])
            parts = [f"{g['grade']}:{g['count']}" for g in data]
            return f"等级分布：{', '.join(parts)}"

        elif tool_name == "get_bonus_info":
            from app.api.talent.performance import get_bonus_info as fn
            result = await fn(quarter=params["quarter"], db=db)
            d = result.get("data", {})
            return f"奖金池：总额 {d.get('totalPool', 0)}，已分配 {d.get('distributed', 0)}，待分配 {d.get('pending', 0)}"

        elif tool_name == "get_quarter_trends":
            from app.api.talent.performance import get_quarter_trends as fn
            result = await fn(db=db)
            data = result.get("data", [])
            return f"近 {len(data)} 个季度趋势"

        elif tool_name == "initiate_appraisal":
            from app.api.talent.performance import initiate_appraisal as fn
            result = await fn(body={"quarter": params["quarter"], "employeeIds": params["employeeIds"]}, db=db)
            return f"已为 {params['quarter']} 发起绩效考核，{len(params['employeeIds'])} 人参与" if result["code"] == 0 else f"发起失败：{result.get('message', '')}"

        elif tool_name == "update_bonus":
            from app.api.talent.performance import update_bonus as fn
            result = await fn(employee_id=params["employeeId"], body={"bonus": params["bonus"]}, db=db)
            return f"已更新员工 {params['employeeId']} 奖金为 {params['bonus']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        elif tool_name == "rag_search":
            try:
                from app.services.rag.search_service import search as rag_search_fn
                from app.services.system.system_settings import get_system_setting
                top_k = params.get("topK", 5)
                recall_threshold = float(await get_system_setting(db, "recallThreshold", 0.75) or 0.75)
                results = await rag_search_fn(
                    query=params["query"],
                    kb_ids=None,
                    top_k=top_k,
                    min_similarity=recall_threshold,
                )
                if results:
                    summaries = []
                    for r in results[:3]:
                        summaries.append(
                            f"[{r['kb_name']}] {r['file_name']}: {r['content'][:300]}"
                        )
                    return f"检索到 {len(results)} 条相关知识:\n" + "\n---\n".join(summaries)
                return "未检索到相关知识"
            except Exception:
                return "知识库检索暂不可用"

        elif tool_name == "list_knowledge":
            from app.api.knowledge.knowledge_base import list_knowledge as fn
            result = await fn(categoryKey=params.get("categoryKey", "all"), keyword=params.get("keyword", ""), db=db)
            data = result.get("data", {})
            if isinstance(data, dict):
                return f"知识库：共 {data.get('total', 0)} 条素材"
            return "查询完成"

        elif tool_name == "get_knowledge_stats":
            from app.api.knowledge.knowledge_base import get_knowledge_stats as fn
            result = await fn(db=db)
            d = result.get("data", {})
            return f"知识库统计：共 {d.get('total', 0)} 条，本月新增 {d.get('newThisMonth', 0)}"

        elif tool_name == "get_knowledge_categories":
            from app.api.knowledge.knowledge_base import get_categories as fn
            result = await fn(db=db)
            data = result.get("data", [])
            return f"知识库分类共 {len(data)} 个根分类"

        elif tool_name == "upload_knowledge_file":
            # fileKey is a MinIO object key uploaded by the frontend "添加资料" flow.
            object_key = params["fileKey"]
            file_name = params["fileName"]
            return f"知识库素材文件「{file_name}」已就绪，object key={object_key}。请接着调用 create_knowledge_item 完成入库。"

        elif tool_name == "create_knowledge_item":
            from app.api.knowledge.knowledge_base import create_knowledge_item as fn
            result = await fn(body=params, db=db)
            if result["code"] == 0:
                return f"已创建知识库素材「{params.get('name', '')}」"
            return f"创建失败：{result.get('message', '')}"

        elif tool_name == "update_knowledge_item":
            from app.api.knowledge.knowledge_base import update_knowledge_item as fn
            result = await fn(item_id=params["id"], body=params.get("fields", {}), db=db)
            return f"已更新素材 {params['id']}" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        elif tool_name == "delete_knowledge_item":
            from app.api.knowledge.knowledge_base import delete_knowledge_item as fn
            result = await fn(item_id=params["id"], db=db)
            return f"已删除素材 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"

        elif tool_name == "recall_test":
            from app.api.knowledge.knowledge_base import recall_test as fn
            result = await fn(body={"query": params["query"]}, db=db)
            data = result.get("data", [])
            return f"召回测试命中 {len(data)} 条"

        elif tool_name == "list_knowledge_bases":
            from app.api.knowledge.rag import list_knowledge_bases as fn
            result = await fn(keyword=params.get("keyword", ""), db=db)
            d = result.get("data", {})
            return f"RAG知识库共 {d.get('total', 0)} 个"

        elif tool_name == "create_knowledge_base":
            from app.api.knowledge.rag import create_knowledge_base as fn
            result = await fn(body={"name": params["name"], "description": params.get("description", "")}, db=db)
            return f"已创建RAG知识库「{params['name']}」" if result.get("code") == 200 or result.get("code") == 0 else f"创建失败：{result.get('message', '')}"

        elif tool_name == "update_knowledge_base":
            from app.api.knowledge.rag import update_knowledge_base as fn
            result = await fn(kb_id=params["id"], body=params.get("fields", {}), db=db)
            return f"已更新RAG知识库 {params['id']}" if result.get("code") in (0, 200) else f"更新失败：{result.get('message', '')}"

        elif tool_name == "delete_knowledge_base":
            from app.api.knowledge.rag import delete_knowledge_base as fn
            result = await fn(kb_id=params["id"], db=db)
            return f"已删除RAG知识库 {params['id']}" if result.get("code") in (0, 200) else f"删除失败：{result.get('message', '')}"

        elif tool_name == "upload_document":
            # fileKey is a MinIO object key uploaded by the frontend "添加资料" flow.
            # The RAG upload_document endpoint expects an UploadFile + kb_id form.
            from app.api.knowledge.rag import upload_document as fn
            from fastapi import UploadFile
            import io
            object_key = params["fileKey"]
            file_name = params["fileName"]
            kb_id = params["kbId"]
            tmp_path = await asyncio.to_thread(minio_storage.download_to_temp, object_key)
            try:
                with open(tmp_path, "rb") as f:
                    content = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            upload_file = UploadFile(file=io.BytesIO(content), filename=file_name)
            result = await fn(file=upload_file, kb_id=kb_id)
            if result.get("code") in (0, 200):
                d = result.get("data", {})
                return f"已上传文档「{file_name}」到知识库 {kb_id}，文档ID={d.get('docId', '')}，任务ID={d.get('taskId', '')}。入库异步进行中。"
            return f"上传失败：{result.get('message', '')}"

        elif tool_name == "list_documents":
            from app.api.knowledge.rag import list_documents as fn
            result = await fn(kb_id=params.get("kbId", ""), db=db)
            d = result.get("data", {})
            return f"文档列表共 {d.get('total', 0)} 个"

        elif tool_name == "delete_document":
            from app.api.knowledge.rag import delete_document as fn
            result = await fn(doc_id=params["id"], db=db)
            return f"已删除文档 {params['id']}" if result.get("code") in (0, 200) else f"删除失败：{result.get('message', '')}"

        elif tool_name == "get_operations_dashboard":
            from app.api.system.dashboard import get_operations as fn
            result = await fn(db=db)
            data = result.get("data", {})
            summary = data.get("summary", {})
            return f"运营概览：{summary.get('title', '')} - {summary.get('text', '')}"

        elif tool_name == "get_dashboard_overview":
            from app.api.system.dashboard import get_overview as fn
            result = await fn(db=db)
            d = result.get("data", {})
            stats = d.get("stats", {})
            return (
                f"数据看板：简历总数 {stats.get('totalResumes', 0)}，"
                f"岗位数 {stats.get('totalPositions', 0)}，"
                f"面试中 {stats.get('totalInterviews', 0)}，"
                f"在职员工 {stats.get('totalEmployees', 0)}"
            )

        elif tool_name == "get_settings":
            from app.api.system.settings import get_settings as fn
            result = await fn(db=db)
            data = result.get("data", {})
            return f"系统设置：公司「{data.get('companyName', '')}」，AI解析={'开启' if data.get('aiResumeAnalysis') else '关闭'}"

        elif tool_name == "update_settings":
            from app.api.system.settings import update_settings as fn
            result = await fn(body=params.get("fields", {}), db=db)
            return f"已更新系统设置" if result["code"] == 0 else f"更新失败：{result.get('message', '')}"

        else:
            return f"工具 {tool_name} 执行完成"

    except Exception as e:
        return f"工具执行错误: {str(e)}"


# ── Agent System Prompt ─────────────────────────────────

def build_system_prompt(agent_id: str = "genie") -> str:
    agent_info = AGENT_CONFIGS.get(agent_id, AGENT_CONFIGS["genie"])
    return f"""你是 Genie 智能招聘系统的「{agent_info['name']}」，{agent_info['description']}。

你的核心使命：用户可以通过对话框完成系统中**所有**原本需要用鼠标点击的操作。主打全能智能化。

你可以完成以下所有功能：
- **简历筛选**：上传/查询/评分/删除/重新解析简历，按岗位/状态/关键词筛选候选人
- **岗位管理**：创建/查询/更新/删除岗位，管理岗位题库
- **面试出题**：生成/保存/替换面试题目（一面/二面），结合候选人简历和知识库出题
- **面试评定**：录入候选人回答、AI 评分、HR 评分、提交评定、查看排行榜
- **试用期考核**：新增试用期员工、录入第一周评估、转正评估、任务跟踪、AI 自动评估
- **绩效管理**：发起季度考核、查询绩效统计/部门绩效/等级分布/奖金池、调整奖金
- **知识库管理**：上传/创建/更新/删除知识库素材、RAG 知识库 CRUD、文档入库、语义检索
- **系统设置**：查看/修改公司信息、AI 开关、合格分数线、试用期天数等配置
- **数据看板**：查看运营概览、招聘漏斗、风险提示

工作规则：
1. **必须使用工具**获取和修改系统中的实际数据，绝对不要编造数据或凭空回答
2. 当用户提出操作需求时，先理解意图，再调用最合适的工具。复杂任务拆解为多步骤，按顺序执行并汇报进度
3. 工具返回结果后，用简洁的中文总结结果，并主动给出下一步建议
4. **文件类操作**（上传简历/上传知识库文档）：用户已通过「添加资料」按钮上传的文件会以 materialId/objectKey 形式出现在对话上下文中。直接使用对应的 upload_xxx 工具，把 fileKey 传进去即可
5. 对于知识类问题（公司制度、技术规范、面试题库），优先使用 rag_search 检索知识库
6. 涉及多个模块的复杂任务，可以协调子 Agent 协作（使用 @interview、@training、@performance 标记）
7. 回复使用 Markdown 格式，包含清晰的标题、列表和重点
8. 如果工具执行失败，如实告知错误并给出修复建议

可用子 Agent（@-mention 协作）：
- @interview：面试出题与评定
- @training：试用期跟踪
- @performance：绩效分析
（默认情况下你已拥有所有工具，无需 @ 也可直接处理）"""


# ── API Endpoints ───────────────────────────────────────

@router.get("/ai-agent/overview")
async def get_ai_agent_overview(db: AsyncSession = Depends(get_db)):
    """Workspace overview."""
    # Active task
    task_result = await db.execute(
        select(AgentTask).where(AgentTask.status == "running").limit(1)
    )
    active_task = task_result.scalar_one_or_none()

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "agents": [
                {
                    "id": aid, "name": cfg["name"], "description": cfg["description"],
                    "status": "就绪" if aid != "genie" else "全能就绪",
                    "tone": "active",
                    "icon": cfg["icon"], "iconBg": cfg["iconBg"], "iconColor": cfg["iconColor"],
                }
                for aid, cfg in AGENT_CONFIGS.items()
            ],
            "workflowSteps": [
                {"key": "recruit", "label": "筛选", "count": 0, "active": True, "badgeTone": "green"},
                {"key": "interview", "label": "面试", "count": 0, "active": False, "badgeTone": "blue"},
                {"key": "training", "label": "试用", "count": 0, "active": False, "badgeTone": "gray"},
                {"key": "performance", "label": "绩效", "count": 0, "active": False, "badgeTone": "gray"},
            ],
            "stats": [
                {"key": "resumes", "label": "待处理简历", "value": 0, "hint": "今日新增", "hintTone": "up"},
                {"key": "interviews", "label": "待面试", "value": 0, "hint": "本周安排", "hintTone": "default"},
                {"key": "offers", "label": "待发offer", "value": 0, "hint": "审批中", "hintTone": "default"},
                {"key": "hours", "label": "AI节省工时", "value": "0h", "hint": "本月累计", "hintTone": "up"},
            ],
            "suggestions": [
                {"id": "s1", "priority": "P1", "title": "处理高匹配候选人", "description": "有3位候选人匹配度超过90分", "actionLabel": "查看详情"},
                {"id": "s2", "priority": "P2", "title": "安排面试", "description": "5位候选人等待一面安排", "actionLabel": "立即安排"},
            ],
            "teamDynamics": [],
            "teamOutput": {"completedTasks": 0, "savedHours": "0h"},
            "activeTask": {
                "title": active_task.title, "description": active_task.description or "",
                "progress": active_task.progress or 0, "elapsed": "刚刚开始",
            } if active_task else None,
            "collaborationSteps": [],
            "phaseResults": [],
        },
    }


@router.get("/ai-agent/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentSession)
        .order_by(desc(AgentSession.updated_at))
        .limit(50)
    )
    sessions = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(s.id),
                "title": s.title or "新对话",
                "agentId": s.agent_id,
                "createdAt": s.created_at.isoformat() if s.created_at else "",
                "updatedAt": s.updated_at.isoformat() if s.updated_at else "",
            }
            for s in sessions
        ],
    }


@router.post("/ai-agent/sessions")
async def create_session(
    db: AsyncSession = Depends(get_db),
):
    """Create a new empty chat session."""
    session = AgentSession(
        title="新对话",
        agent_id="genie",
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(session.id),
            "title": session.title,
            "agentId": session.agent_id,
            "createdAt": session.created_at.isoformat() if session.created_at else "",
            "updatedAt": session.updated_at.isoformat() if session.updated_at else "",
        },
    }


@router.delete("/ai-agent/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentSession).where(
            AgentSession.id == session_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"code": 404, "message": "对话不存在", "data": None}

    # Clean up MinIO objects for the session's materials before cascade delete
    mat_result = await db.execute(
        select(AgentMaterial).where(AgentMaterial.session_id == session_id)
    )
    for mat in mat_result.scalars().all():
        if mat.file_path and not (os.path.isabs(mat.file_path)):
            await asyncio.to_thread(minio_storage.delete_object, mat.file_path)

    await db.delete(session)
    return {"code": 0, "message": "ok", "data": None}


@router.get("/ai-agent/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentMessage)
        .where(AgentMessage.session_id == session_id)
        .order_by(AgentMessage.created_at)
    )
    messages = result.scalars().all()
    return {
        "code": 0,
        "message": "ok",
        "data": [
            {
                "id": str(m.id),
                "sessionId": str(m.session_id),
                "role": m.role,
                "content": m.content,
                "createdAt": m.created_at.isoformat() if m.created_at else "",
            }
            for m in messages
        ],
    }


@router.post("/ai-agent/materials")
async def upload_material(
    type: str = Form(...),
    file: UploadFile = File(None),
    knowledgeId: str = Form(None),
    knowledgeName: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Get or create session
    session_result = await db.execute(
        select(AgentSession).order_by(desc(AgentSession.updated_at)).limit(1)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        session = AgentSession(title="新对话", agent_id="recruit")
        db.add(session)
        await db.flush()

    material_name = ""
    material_path = ""
    if file:
        file_ext = os.path.splitext(file.filename or "material")[1] or ".pdf"
        object_key = f"agent/{uuid.uuid4()}{file_ext}"
        content = await file.read()
        try:
            await asyncio.to_thread(
                minio_storage.upload_bytes, object_key, content, "application/octet-stream"
            )
        except Exception as e:
            return {"code": 500, "message": f"资料存储失败: {e}", "data": None}
        material_path = object_key
        material_name = file.filename or "material"
    elif knowledgeName:
        material_name = knowledgeName

    material = AgentMaterial(
        session_id=session.id,
        name=material_name,
        type=type,
        knowledge_id=knowledgeId,
        file_path=material_path,
    )
    db.add(material)
    await db.flush()
    await db.refresh(material)

    # If uploading a resume, trigger AI analysis
    analysis = None
    if type in ("resume", "file") and material_path:
        try:
            text, extract_error = extract_text_from_file(material_path)
            if text.strip():
                parsed, _ = await parse_resume_with_llm(text)
                analysis_data = parsed.get("analysis", {}) if parsed else {}
                # 维度评分由 5 个独立子 Agent 计算（教育背景走规则，其余走 LLM）。
                from app.services.resume_scoring import score_all
                dimensions = await score_all(text, parsed or {}, "")
                analysis = {
                    "overallScore": analysis_data.get("overallScore"),
                    "summary": analysis_data.get("summary", ""),
                    "keywords": analysis_data.get("keywords", []),
                    "dimensions": dimensions,
                    "highlights": analysis_data.get("highlights", []),
                    "risks": analysis_data.get("risks", []),
                    "recommendation": analysis_data.get("recommendation", ""),
                    "positionMatch": analysis_data.get("positionMatch", ""),
                    "experienceInsight": analysis_data.get("experienceInsight", ""),
                }
                # Also extract basic candidate info
                analysis["candidateName"] = parsed.get("name", "")
                analysis["skills"] = parsed.get("skills", [])
        except Exception as e:
            print(f"AI Agent resume analysis error: {e}")

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "id": str(material.id),
            "name": material.name,
            "type": material.type,
            "knowledgeId": str(material.knowledge_id) if material.knowledge_id else None,
            "uploadedAt": material.uploaded_at.isoformat() if material.uploaded_at else "",
            "analysis": analysis,
        },
    }


@router.post("/ai-agent/suggestions/{suggestion_id}/trigger")
async def trigger_suggestion(
    suggestion_id: str,
    db: AsyncSession = Depends(get_db),
):
    return {"code": 0, "message": "ok", "data": None}


@router.post("/ai-agent/chat")
async def agent_chat(
    request: Request,
    body: dict,
):
    """Main SSE streaming chat endpoint.

    This endpoint deliberately does NOT use ``Depends(get_db)``. Holding a DB
    session for the whole streaming duration leaks connections: if the client
    disconnects while the generator is blocked inside a long LLM call or a
    sync tool call, the generator cannot be cancelled and the session is
    never returned to the pool. After a few such leaks the PostgreSQL pool is
    exhausted and every other API call hangs ("前端点几次就收不到请求").

    Instead, all DB work is done in short-lived sessions (``async with
    async_session_factory()``) that are committed and closed immediately, so
    no connection is held while the SSE stream is open.
    """
    message = body.get("message", "")
    session_id = body.get("sessionId")
    agent_id = body.get("agentId", "genie")
    mentioned_agent_ids = body.get("mentionedAgentIds", [])
    material_ids = body.get("materialIds", [])

    # ── Fetch attached materials and build context ──
    material_context = ""
    if material_ids:
        try:
            async with async_session_factory() as db:
                mat_result = await db.execute(
                    select(AgentMaterial).where(AgentMaterial.id.in_(material_ids))
                )
                attached_materials = mat_result.scalars().all()
            if attached_materials:
                lines = ["\n\n--- 附件资料 ---"]
                for mat in attached_materials:
                    lines.append(f"\n[{mat.type}] {mat.name}")
                    if mat.file_path:
                        try:
                            file_text, _ = extract_text_from_file(mat.file_path)
                            if file_text.strip():
                                truncated = file_text[:3000] + ("..." if len(file_text) > 3000 else "")
                                lines.append(f"内容:\n{truncated}")
                        except Exception:
                            pass
                material_context = "\n".join(lines)
        except Exception:
            pass

    # ── Get or create session, save user message, fetch history ──
    is_new_session = False
    session_title = "新对话"
    history_rows: list[tuple[str, str]] = []

    async with async_session_factory() as db:
        session = None
        if session_id:
            try:
                result = await db.execute(select(AgentSession).where(AgentSession.id == session_id))
                session = result.scalar_one_or_none()
            except Exception:
                session = None

        if not session:
            is_new_session = True
            session = AgentSession(
                title="新对话",
                agent_id="genie",
            )
            db.add(session)
            await db.flush()
            session_id = str(session.id)

        # Save user message
        user_msg = AgentMessage(
            session_id=session.id,
            role="user",
            content=message,
        )
        db.add(user_msg)
        await db.flush()

        # Update session title
        if session.title in ("新对话", None, ""):
            session.title = message[:50] if message else "新对话"
        session_title = session.title or "新对话"

        # Fetch conversation history (last 20)
        history_result = await db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session.id)
            .order_by(AgentMessage.created_at)
            .limit(20)
        )
        history = history_result.scalars().all()
        for h in history[:-1]:  # exclude the just-saved user message
            history_rows.append((h.role, h.content or ""))

        await db.commit()
        session_obj_id = session.id

    # ── Build LangChain message history from plain data ──
    lc_messages = []
    for role, content in history_rows:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    lc_messages.append(HumanMessage(content=message + material_context))

    async def event_stream() -> AsyncGenerator[str, None]:
        result = AgentResult()
        task_id = None

        try:
            # Send initial thinking
            thinking_text = f"收到任务，正在作为{AGENT_CONFIGS.get(agent_id, {}).get('name', 'AI Agent')}分析您的指令..."
            yield sse_event("thinking", {"text": thinking_text, "append": False})

            # Sub-agent handoffs
            for mid in mentioned_agent_ids:
                agent_info = AGENT_CONFIGS.get(mid, {})
                yield sse_event("agent_handoff", {
                    "from": AGENT_CONFIGS.get(agent_id, {}).get("name", "主Agent"),
                    "to": agent_info.get("name", mid),
                    "reason": f"需要{agent_info.get('description', '协作处理')}",
                })

            # Create task for tracking (short-lived session)
            async with async_session_factory() as db:
                task = AgentTask(
                    session_id=session_obj_id,
                    title=message[:50] if message else "处理中",
                    description=message,
                    progress=0,
                    status="running",
                    started_at=datetime.utcnow(),
                )
                db.add(task)
                await db.flush()
                task_id = task.id
                await db.commit()

            # Build LangGraph agent
            langchain_tools = create_langchain_tools(agent_id)
            system_prompt = build_system_prompt(agent_id)
            graph = build_agent_graph(langchain_tools, system_prompt)

            # Stream agent execution
            async for sse_str in stream_agent_response(graph, lc_messages, result):
                if await request.is_disconnected():
                    break
                yield sse_str

            # Save assistant message + update task (short-lived session)
            async with async_session_factory() as db:
                assistant_msg = AgentMessage(
                    session_id=session_obj_id,
                    role="assistant",
                    content=result.full_content,
                    thinking=None,
                    tool_blocks=result.tool_blocks if result.tool_blocks else None,
                )
                db.add(assistant_msg)
                await db.flush()

                # AI-generated session title
                if is_new_session or session_title == "新对话":
                    try:
                        title_prompt = (
                            f"根据以下对话内容，生成一个简短的标题（10个字以内，不要引号）：\n"
                            f"用户：{message[:200]}\nAI：{result.full_content[:200]}"
                        )
                        title_resp = await llm_client.chat.completions.create(
                            model=settings.deepseek_model,
                            messages=[{"role": "user", "content": title_prompt}],
                            temperature=0.7,
                            max_tokens=32,
                        )
                        new_title = title_resp.choices[0].message.content.strip().strip('"').strip("'")
                        if new_title and len(new_title) > 1:
                            sess_result = await db.execute(
                                select(AgentSession).where(AgentSession.id == session_obj_id)
                            )
                            sess = sess_result.scalar_one_or_none()
                            if sess:
                                sess.title = new_title[:50]
                                await db.flush()
                    except Exception:
                        pass

                # Update task status
                if task_id is not None:
                    task_result = await db.execute(select(AgentTask).where(AgentTask.id == task_id))
                    task_obj = task_result.scalar_one_or_none()
                    if task_obj:
                        task_obj.status = "done"
                        task_obj.progress = 100
                        task_obj.finished_at = datetime.utcnow()

                await db.commit()

            yield sse_event("phase_result", {
                "id": f"phase_{uuid.uuid4().hex[:6]}",
                "tone": "success",
                "title": "任务完成",
                "description": message[:80] + ("..." if len(message) > 80 else ""),
            })

            yield sse_event("done", {})

        except asyncio.CancelledError:
            # Client disconnected — mark the task cancelled (short-lived session),
            # then re-raise so Starlette closes the stream and the socket is released.
            try:
                async with async_session_factory() as db:
                    if task_id is not None:
                        task_result = await db.execute(select(AgentTask).where(AgentTask.id == task_id))
                        task_obj = task_result.scalar_one_or_none()
                        if task_obj:
                            task_obj.status = "cancelled"
                            task_obj.finished_at = datetime.utcnow()
                    await db.commit()
            except Exception:
                pass
            raise
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
