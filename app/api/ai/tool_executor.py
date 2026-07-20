"""Tool executor — translates LLM tool calls into API calls and formats results.

Each tool returns a human-readable text summary that the LLM uses to
formulate its response to the user.  Separated from agent_chat.py so the
600-line if/elif chain doesn't obscure the API plumbing.
"""

from __future__ import annotations

import json
import os
import re
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure import minio_storage


def _camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case.

    >>> _camel_to_snake("jdRequirements")
    'jd_requirements'
    >>> _camel_to_snake("interviewCriteriaR1")
    'interview_criteria_r1'
    >>> _camel_to_snake("weeks24Plan")
    'weeks_2_4_plan'
    """
    # Insert underscore before capital letters that follow lowercase or digits
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    # Insert underscore between digit sequences and letters
    s = re.sub(r"(\d+)([A-Za-z])", r"\1_\2", s)
    s = re.sub(r"([A-Za-z])(\d+)", r"\1_\2", s)
    return s.lower()


def _convert_keys(obj: dict, converter=_camel_to_snake) -> dict:
    """Recursively convert dict keys using *converter*."""
    result = {}
    for k, v in obj.items():
        new_key = converter(k)
        result[new_key] = _convert_keys(v, converter) if isinstance(v, dict) else v
    return result


def sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event string for streaming responses."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def execute_tool_call(tool_name: str, params: dict, db: AsyncSession) -> str:
    """Execute a tool and return a human-readable result summary."""
    try:
        # ── Recruitment / Resumes ──────────────────────────

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
                            comment = (d.get('comment') or '').strip()
                            suffix = f" — {comment[:200]}" if comment else ""
                            lines.append(f"      - {d.get('name','')}: {d.get('score',0)}分{suffix}")
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

        # ── Positions ───────────────────────────────────────

        elif tool_name == "get_position":
            from app.api.recruitment.positions import get_position as fn
            result = await fn(position_id=params["id"], db=db)
            d = result.get("data") or {}
            if not d:
                return "岗位不存在"
            # 默认只返回 JD 正文，避免把整份评分表 JSON 塞进上下文诱使模型做「匹配分析」
            lines = [
                f"岗位「{d.get('name', '')}」(ID: {d.get('id', '')})",
                f"  部门: {d.get('department', '未设置')}",
            ]
            if d.get("jdResponsibilities"):
                lines.append(f"  岗位职责:\n{d['jdResponsibilities']}")
            if d.get("jdRequirements"):
                lines.append(f"  任职要求:\n{d['jdRequirements']}")
            if d.get("jdPreferred"):
                lines.append(f"  加分项:\n{d['jdPreferred']}")
            if d.get("jdTechStack"):
                lines.append(f"  技术栈: {d['jdTechStack']}")
            extras = []
            if d.get("screeningCriteria"):
                extras.append("筛选评分标准")
            if d.get("interviewCriteriaR1"):
                extras.append("一面评分标准")
            if d.get("interviewCriteriaR2"):
                extras.append("二面评分标准")
            if extras:
                lines.append(
                    f"  （另有配置未展开：{' / '.join(extras)}。用户明确要求查看评分标准时再说明即可）"
                )
            return "\n".join(lines)

        elif tool_name == "create_position":
            from sqlalchemy import text as sa_text
            col_map = {
                "name": "name", "department": "department",
                "jdResponsibilities": "jd_responsibilities",
                "jdRequirements": "jd_requirements",
                "jdPreferred": "jd_preferred",
                "jdTechStack": "jd_tech_stack",
                "educationRequirement": "education_requirement",
                "experienceRequirement": "experience_requirement",
                "ageRequirement": "age_requirement",
                "salaryRange": "salary_range",
            }
            columns = []
            values = {}
            for camel_key, value in params.items():
                col = col_map.get(camel_key)
                if col and value is not None:
                    columns.append(col)
                    values[col] = value
            if not columns:
                return "创建失败：没有提供有效字段"
            placeholders = ", ".join(f":{c}" for c in columns)
            cols_str = ", ".join(columns)
            sql = f"INSERT INTO positions ({cols_str}) VALUES ({placeholders}) RETURNING id"
            result = await db.execute(sa_text(sql), values)
            row = result.fetchone()
            new_id = str(row[0]) if row else "?"
            await db.flush()
            return f"已创建岗位「{params.get('name', '未命名')}」(ID: {new_id})"

        elif tool_name == "update_position":
            from sqlalchemy import text as sa_text
            fields = params.get("fields", {})
            position_id = params["id"]
            col_map = {
                "name": "name", "department": "department",
                "jdResponsibilities": "jd_responsibilities",
                "jdRequirements": "jd_requirements",
                "jdPreferred": "jd_preferred",
                "jdTechStack": "jd_tech_stack",
                "educationRequirement": "education_requirement",
                "experienceRequirement": "experience_requirement",
                "ageRequirement": "age_requirement",
                "salaryRange": "salary_range",
                "screeningCriteria": "screening_criteria",
                "interviewCriteriaR1": "interview_criteria_r1",
                "interviewCriteriaR2": "interview_criteria_r2",
                "week1ProjectRequirement": "week1_project_requirement",
                "weeks24Plan": "weeks_2_4_plan",
                "laterWeekScoring": "later_week_scoring",
                "conversionCriteria": "conversion_criteria",
            }
            set_clauses = []
            values = {"id": position_id}
            for camel_key, value in fields.items():
                col = col_map.get(camel_key)
                if col and value is not None:
                    set_clauses.append(f"{col} = :{col}")
                    # JSON fields: dump to string if dict
                    if isinstance(value, dict):
                        values[col] = json.dumps(value, ensure_ascii=False)
                    else:
                        values[col] = value
            if not set_clauses:
                return "没有可更新的字段（提供的字段名可能不正确）"
            sql = f"UPDATE positions SET {', '.join(set_clauses)} WHERE id = :id"
            result = await db.execute(sa_text(sql), values)
            await db.flush()
            rowcount = result.rowcount
            # Verify: read back the updated columns
            verify_cols = ", ".join(col_map[c] for c in fields if c in col_map)
            previews = []
            if verify_cols:
                verify = await db.execute(
                    sa_text(f"SELECT {verify_cols} FROM positions WHERE id = :id"),
                    {"id": position_id},
                )
                row = verify.fetchone()
                if row:
                    for i, col_name in enumerate(verify_cols.split(", ")):
                        val = row[i]
                        preview = (str(val) or "")[:80]
                        previews.append(f"{col_name}={preview}")
            preview_text = "; ".join(previews) if previews else "ok"
            return (
                f"已更新岗位，影响 {rowcount} 行。回读验证: {preview_text}"
            )

        elif tool_name == "delete_position":
            from app.api.recruitment.positions import delete_position as fn
            result = await fn(position_id=params["id"], db=db)
            return f"已删除岗位 {params['id']}" if result["code"] == 0 else f"删除失败：{result.get('message', '')}"

        elif tool_name == "get_position_questions":
            from app.api.recruitment.positions import get_position_questions as fn
            result = await fn(position_id=params["positionId"], round=params["round"], db=db)
            qs = result.get("data", [])
            if not qs:
                return f"岗位题库（{params['round']}）暂无题目"
            lines = [f"岗位题库（{params['round']}）共 {len(qs)} 道题："]
            for q in qs:
                lines.append(f"  [{q.get('index','?')}] {q.get('content','')} | 分类:{q.get('category','')} | 难度:{q.get('difficulty','')}")
            return "\n".join(lines)

        elif tool_name == "save_position_questions":
            from app.api.recruitment.positions import save_position_questions as fn
            result = await fn(position_id=params["positionId"], body={"round": params["round"], "questions": params["questions"]}, db=db)
            return f"已保存岗位题库 {len(params['questions'])} 道题" if result["code"] == 0 else f"保存失败：{result.get('message', '')}"

        # ── Interview ────────────────────────────────────────

        elif tool_name == "get_questions":
            from app.api.talent.interview import get_questions as fn
            result = await fn(candidateId=params["candidateId"], round=params["round"], db=db)
            questions = result.get("data", [])
            if not questions:
                return f"暂无{params['round']}面试题，请先生成"
            lines = [f"候选人 {params['candidateId']} 的{params['round']}面试题（共 {len(questions)} 道）："]
            for q in questions:
                lines.append(
                    f"  [{q.get('index','?')}] {q.get('content','')} | "
                    f"分类:{q.get('category','')} | 难度:{q.get('difficulty','')} | ID:{q.get('id','')}"
                )
            return "\n".join(lines)

        elif tool_name == "generate_questions":
            from app.api.talent.interview import regenerate_questions as fn
            result = await fn(body=params, db=db)
            questions = result.get("data", [])
            if not questions:
                return f"题目生成失败：{result.get('message', '未知错误')}"
            lines = [f"已生成 {len(questions)} 道{params.get('round','')}面试题："]
            for q in questions:
                lines.append(
                    f"  [{q.get('index','?')}] {q.get('content','')} | "
                    f"分类:{q.get('category','')} | 难度:{q.get('difficulty','')} | ID:{q.get('id','')}"
                )
            return "\n".join(lines)

        elif tool_name == "get_evaluation":
            from app.api.talent.interview import get_evaluation as fn
            result = await fn(candidate_id=params["candidateId"], round=params.get("round", "first"), db=db)
            scores = result.get("data", [])
            if not scores:
                return "暂无面试评分数据"
            scored = [s for s in scores if s.get("primaryScore") is not None]
            lines = [f"面试评分（{len(scored)}/{len(scores)} 题已评分）："]
            for s in scores:
                score_str = f"{s.get('primaryScore')}分" if s.get('primaryScore') is not None else "未评分"
                answer_preview = (s.get('answer') or '')[:100]
                lines.append(
                    f"  [{s.get('index','?')}] {s.get('content','')[:80]}... | "
                    f"得分:{score_str} | 分类:{s.get('category','')}"
                )
                if answer_preview:
                    lines.append(f"      回答: {answer_preview}...")
            return "\n".join(lines)

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

        # ── Probation ────────────────────────────────────────

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

        # ── Performance ─────────────────────────────────────

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

        # ── Knowledge / RAG ─────────────────────────────────

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

        # ── Dashboard / Settings ────────────────────────────

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

        # ── Database direct access (natural-language CRUD) ──

        elif tool_name == "db_list_tables":
            from sqlalchemy import text
            result = await db.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            ))
            tables = [row[0] for row in result.fetchall()]
            if not tables:
                return "数据库中没有找到任何表"
            lines = [f"数据库共 {len(tables)} 张表："]
            for t in tables:
                lines.append(f"  - {t}")
            return "\n".join(lines)

        elif tool_name == "db_describe_table":
            from sqlalchemy import text
            table = params["table"]
            result = await db.execute(text(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :tbl "
                "ORDER BY ordinal_position"
            ), {"tbl": table})
            rows = result.fetchall()
            if not rows:
                return f"表「{table}」不存在或没有列"
            lines = [f"表「{table}」结构（共 {len(rows)} 列）："]
            for col in rows:
                nullable = "可空" if col[2] == "YES" else "非空"
                default = f" 默认={col[3]}" if col[3] else ""
                lines.append(f"  {col[0]:30s} {col[1]:20s} {nullable}{default}")
            return "\n".join(lines)

        elif tool_name == "db_query":
            from sqlalchemy import text as sa_text
            sql = params["sql"].strip()
            sql_upper = sql.upper()
            if not sql_upper.startswith("SELECT"):
                return "❌ db_query 只允许执行 SELECT 查询。如需修改数据请使用 db_update。"
            if any(kw in sql_upper for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE")):
                return "❌ db_query 只允许只读的 SELECT 查询。如需修改数据请使用 db_update。"
            limit = params.get("limit", 20)
            if "LIMIT" not in sql_upper:
                sql = f"{sql.rstrip(';')} LIMIT {limit}"
            result = await db.execute(sa_text(sql))
            rows = result.fetchall()
            cols = list(result.keys())
            if not rows:
                return "查询结果为空"
            lines = [f"查询返回 {len(rows)} 行（列: {', '.join(cols)}）："]
            for i, row in enumerate(rows):
                cells = ", ".join(f"{cols[j]}={row[j]!r}" for j in range(len(cols)))
                lines.append(f"  [{i+1}] {cells}")
            return "\n".join(lines)

        elif tool_name == "db_update":
            from sqlalchemy import text as sa_text
            sql = params["sql"].strip()
            sql_upper = sql.upper()
            # Safety: UPDATE/DELETE must have WHERE
            if sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE"):
                if "WHERE" not in sql_upper:
                    return (
                        "❌ 安全限制：UPDATE 和 DELETE 必须包含 WHERE 条件，"
                        "禁止全表修改。请加上 WHERE 后重试。"
                    )
            if sql_upper.startswith("SELECT"):
                return "❌ db_update 用于写操作。查询请使用 db_query。"
            # Execute
            result = await db.execute(sa_text(sql))
            # Flush first so rowcount is available, then commit via the caller
            await db.flush()
            rowcount = result.rowcount if hasattr(result, 'rowcount') else "?"
            return f"✅ SQL 执行成功，影响 {rowcount} 行。已提交到数据库。你可以用 db_query 验证结果。"

        else:
            return f"工具 {tool_name} 执行完成"

    except Exception as e:
        return f"工具执行错误: {str(e)}"
