"""模块二: 简历评分 + 筛选决策 API。"""
from __future__ import annotations
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.recruitment import Candidate
from app.models.phase1 import ResumeScore
from app.models.settings import SystemSetting
from app.core.security import get_current_user, require_permission, CurrentUser
from app.core.state_machine import transition, StateError
from app.core.exceptions import push_exception
from app.schemas.ai_advice import AIAdvice
from app.utils.responses import ok, fail, not_found
from app.utils.audit import write_audit
from app.utils.clock import iso_utc
from app.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

router = APIRouter(tags=["简历筛选"])


from app.services.system.system_settings import get_system_setting as _get_threshold


def compute_grade(total: int, thresholds: dict) -> str:
    if total >= thresholds.get("A", 85): return "A"
    if total >= thresholds.get("B", 70): return "B"
    if total >= thresholds.get("C", 60): return "C"
    return "D"


def serialize_score(s: ResumeScore, viewer: CurrentUser | None = None) -> dict:
    return {
        "id": str(s.id),
        "candidateId": str(s.candidate_id),
        "positionId": str(s.position_id) if s.position_id else None,
        "total": s.total,
        "grade": s.grade,
        "dimensions": {
            "skillMatch": s.skill_match,
            "projectMatch": s.project_match,
            "positionExp": s.position_exp,
            "achievement": s.achievement,
            "industryExp": s.industry_exp,
            "learning": s.learning,
            "stability": s.stability,
            "bonusSkill": s.bonus_skill,
        },
        "advice": s.advice,   # already snake_case AIAdvice dict
        "createdAt": iso_utc(s.created_at),
    }


class ScoreRequest(BaseModel):
    model_config = {"populate_by_name": True}
    candidate_id: str = Field(alias="candidateId")
    position_id: Optional[str] = Field(None, alias="positionId")


@router.post("/candidates/score")
async def score_candidate(
    body: ScoreRequest,
    current: CurrentUser = Depends(require_permission("resume:view")),
    db: AsyncSession = Depends(get_db),
):
    try:
        cid = uuid.UUID(body.candidate_id)
    except ValueError:
        return not_found("候选人不存在")
    cand_row = await db.execute(select(Candidate).where(Candidate.id == cid))
    candidate = cand_row.scalar_one_or_none()
    if candidate is None:
        return not_found("候选人不存在")

    pid = None
    if body.position_id:
        try:
            pid = uuid.UUID(body.position_id)
        except ValueError:
            return not_found("岗位不存在")

    thresholds = await _get_threshold(db, "resume_grade_thresholds", {"A": 85, "B": 70, "C": 60})
    confidence_threshold = await _get_threshold(db, "ai_confidence_threshold", 0.6)

    # 调用 AI 评分
    try:
        from app.services.ai import llm_chat
        from app.models.recruitment import CandidateAIAnalysis
        ai_row = await db.execute(
            select(CandidateAIAnalysis).where(CandidateAIAnalysis.candidate_id == cid)
        )
        ai_ana = ai_row.scalar_one_or_none()
        resume_summary = ""
        if ai_ana:
            resume_summary = f"技能: {ai_ana.keywords}\n工作经历: {ai_ana.experience_insight}\n推荐: {ai_ana.recommendation}"
        else:
            resume_summary = f"姓名:{candidate.name} 学历:{candidate.education} 工作年限:{candidate.experience}"

        import json
        prompt = (
            "你是专业 HR，请对以下候选人简历做 8 维评分，满分 100 分。\n"
            f"候选人信息: {resume_summary}\n\n"
            "返回 JSON（所有数值为整数/浮点数，字段不可缺失）:\n"
            '{"skill_match":0-25,"project_match":0-20,"position_exp":0-15,'
            '"achievement":0-15,"industry_exp":0-10,"learning":0-5,"stability":0-5,"bonus_skill":0-5,'
            '"total":0-100,"confidence":0.0-1.0,'
            '"evidence":["..."],"strengths":["..."],"risks":["..."],'
            '"missing_information":["..."],"recommended_action":"...","requires_human_confirmation":true}'
        )
        resp_text = await llm_chat([{"role": "user", "content": prompt}])
        raw = extract_json_object(resp_text)
        if raw is None:
            raise ValueError("模型未返回可解析的 JSON")
    except Exception as e:
        return fail(500, f"AI 评分失败: {e}")

    # 验证 evidence
    evidence = raw.get("evidence") or []
    if not evidence:
        await push_exception(db, entity_type="candidate", entity_id=cid,
                             exception_type="low_confidence",
                             detail="AI 评分未返回证据", severity="block")
        return fail(422, "AI 建议缺少证据")

    confidence = float(raw.get("confidence", 0.0))
    if confidence < confidence_threshold:
        await push_exception(db, entity_type="candidate", entity_id=cid,
                             exception_type="low_confidence",
                             detail=f"AI 置信度 {confidence:.2f} < {confidence_threshold}", severity="block")

    total = int(raw.get("total", 0))
    grade = compute_grade(total, thresholds)

    advice_dict = {
        "result": raw.get("recommended_action", ""),
        "score": total,
        "confidence": confidence,
        "evidence": evidence,
        "strengths": raw.get("strengths") or [],
        "risks": raw.get("risks") or [],
        "missing_information": raw.get("missing_information") or [],
        "recommended_action": raw.get("recommended_action", ""),
        "requires_human_confirmation": bool(raw.get("requires_human_confirmation", True)),
    }

    # upsert
    existing_row = await db.execute(
        select(ResumeScore).where(ResumeScore.candidate_id == cid,
                                   ResumeScore.position_id == pid)
    )
    score_obj = existing_row.scalar_one_or_none()
    if score_obj is None:
        score_obj = ResumeScore(candidate_id=cid, position_id=pid)
        db.add(score_obj)

    score_obj.skill_match = int(raw.get("skill_match", 0))
    score_obj.project_match = int(raw.get("project_match", 0))
    score_obj.position_exp = int(raw.get("position_exp", 0))
    score_obj.achievement = int(raw.get("achievement", 0))
    score_obj.industry_exp = int(raw.get("industry_exp", 0))
    score_obj.learning = int(raw.get("learning", 0))
    score_obj.stability = int(raw.get("stability", 0))
    score_obj.bonus_skill = int(raw.get("bonus_skill", 0))
    score_obj.total = total
    score_obj.grade = grade
    score_obj.advice = advice_dict

    # score 与 screening_ai_score 都要写：列表页与 grade 换算读的是 score
    # （见 resume_serializer.serialize_candidate），此前只写 screening_ai_score
    # 而那个字段全仓无人序列化，导致「跑完 AI 评分，界面分数和等级毫无变化」。
    # 上传解析时写入的 score 只是解析阶段的粗估，这里的 8 维加权分是权威值，应覆盖它。
    candidate.screening_ai_score = total
    candidate.score = total
    await db.flush()

    # 评分完成即进入待筛选池。迁移失败不阻断打分结果返回（分数本身已算出且有效），
    # 但必须记日志——静默 pass 会让「候选人卡在 new 永远进不了筛选列表」无声无息。
    if candidate.status in ("new", "parsed"):
        try:
            await transition(db, "candidate", candidate, "pending_screen",
                             actor_id=current.id, actor_name=current.username,
                             skip_block_check=True)
        except StateError as e:
            logger.warning("候选人 %s 打分后无法进入待筛选: %s", cid, e.message)

    return ok(serialize_score(score_obj, current))


@router.get("/candidates/ranking")
async def candidate_ranking(
    position_id: str = Query(alias="positionId"),
    grade: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    current: CurrentUser = Depends(require_permission("resume:view")),
    db: AsyncSession = Depends(get_db),
):
    try:
        pid = uuid.UUID(position_id)
    except ValueError:
        return fail(400, "positionId 格式无效")
    q = (select(ResumeScore, Candidate)
         .join(Candidate, Candidate.id == ResumeScore.candidate_id)
         .where(ResumeScore.position_id == pid))
    if grade:
        q = q.where(ResumeScore.grade == grade.upper())
    total = (await db.execute(
        select(func.count()).select_from(q.subquery())
    )).scalar() or 0
    rows = (await db.execute(
        q.order_by(ResumeScore.total.desc())
         .offset((page - 1) * page_size).limit(page_size)
    )).all()
    result = []
    for rank_idx, (score, cand) in enumerate(rows, start=(page - 1) * page_size + 1):
        risks = (score.advice or {}).get("risks", []) if score.advice else []
        result.append({
            "candidateId": str(cand.id),
            "name": cand.name,
            "total": score.total,
            "grade": score.grade,
            "rank": rank_idx,
            "status": cand.status,
            "risks": risks,
        })
    return ok({"list": result, "total": total, "page": page, "pageSize": page_size})


class DecisionRequest(BaseModel):
    model_config = {"populate_by_name": True}
    action: str    # invite/supplement/transfer/reserve/reject
    reason: Optional[str] = None
    target_position_id: Optional[str] = Field(None, alias="targetPositionId")
    material_note: Optional[str] = Field(None, alias="materialNote")
    resume_viewed: bool = Field(alias="resumeViewed")


@router.post("/candidates/{cid}/decision")
async def candidate_decision(
    cid: str,
    body: DecisionRequest,
    current: CurrentUser = Depends(require_permission("resume:decide")),
    db: AsyncSession = Depends(get_db),
):
    try:
        cand_id = uuid.UUID(cid)
    except ValueError:
        return not_found("候选人不存在")
    row = await db.execute(select(Candidate).where(Candidate.id == cand_id))
    candidate = row.scalar_one_or_none()
    if candidate is None:
        return not_found("候选人不存在")

    if not body.resume_viewed:
        return fail(422, "请先查看原始简历")

    # AI 冲突检测
    score_row = await db.execute(
        select(ResumeScore).where(ResumeScore.candidate_id == cand_id)
    )
    score = score_row.scalar_one_or_none()
    if score and score.advice:
        ai_action = (score.advice.get("recommended_action") or "").lower()
        human = body.action.lower()
        conflict = (
            ("面试" in ai_action and human in ("reject", "reserve")) or
            ("暂不" in ai_action and human == "invite")
        )
        if conflict and not body.reason:
            return fail(400, "决策与 AI 建议不一致，请填写原因")

    # 转岗的目标岗位先解析校验再赋值——放在 transition 之前赋值会在迁移失败时
    # 留下「岗位已改但状态没动」的脏数据（get_db 不会为正常 return 回滚）。
    target_position: Optional[uuid.UUID] = None
    if body.action == "transfer":
        if not body.target_position_id:
            return fail(400, "转岗需指定目标岗位")
        try:
            target_position = uuid.UUID(body.target_position_id)
        except ValueError:
            return fail(400, "目标岗位 ID 格式不正确")
        exists = await db.execute(select(Position.id).where(Position.id == target_position))
        if exists.scalar_one_or_none() is None:
            return not_found("目标岗位不存在")

    # 状态迁移
    try:
        if body.action == "invite":
            await transition(db, "candidate", candidate, "invited",
                             actor_id=current.id, actor_name=current.username,
                             reason=body.reason)
        elif body.action == "reserve":
            await transition(db, "candidate", candidate, "talent_pool",
                             actor_id=current.id, actor_name=current.username,
                             reason=body.reason)
        elif body.action == "reject":
            await transition(db, "candidate", candidate, "rejected",
                             actor_id=current.id, actor_name=current.username,
                             reason=body.reason)
        elif body.action == "transfer":
            candidate.position_id = target_position
            await transition(db, "candidate", candidate, "pending_screen",
                             actor_id=current.id, actor_name=current.username,
                             reason=body.reason, skip_block_check=True)
        elif body.action == "supplement":
            await transition(db, "candidate", candidate, "pending_materials",
                             actor_id=current.id, actor_name=current.username,
                             reason=body.material_note or body.reason,
                             skip_block_check=True)
        else:
            return fail(400, f"不支持的操作: {body.action}")
    except StateError as e:
        await db.rollback()
        return fail(409, e.message)

    note = f"（{body.material_note}）" if body.material_note else ""
    await write_audit(db, actor=current.username,
                      action=f"简历决策: {body.action}{note}", section="resume")
    return ok({"status": candidate.status})


@router.get("/candidates/{cid}/score")
async def get_score(
    cid: str,
    position_id: Optional[str] = Query(None, alias="positionId"),
    current: CurrentUser = Depends(require_permission("resume:view")),
    db: AsyncSession = Depends(get_db),
):
    try:
        cand_id = uuid.UUID(cid)
    except ValueError:
        return not_found("简历评分不存在")
    q = select(ResumeScore).where(ResumeScore.candidate_id == cand_id)
    if position_id:
        try:
            q = q.where(ResumeScore.position_id == uuid.UUID(position_id))
        except ValueError:
            pass
    row = await db.execute(q.order_by(ResumeScore.created_at.desc()).limit(1))
    score = row.scalar_one_or_none()
    if score is None:
        return not_found("该候选人暂无评分记录")
    return ok(serialize_score(score, current))
