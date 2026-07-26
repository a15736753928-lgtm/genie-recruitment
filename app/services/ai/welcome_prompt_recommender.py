"""欢迎页推荐 Agent —— 根据当前系统状态，AI 推荐用户可以做什么。

设计原则：
- **对用户无感**：HTTP 接口只读缓存，绝不在请求路径上调 LLM。
- **后台默默跑**：由 APScheduler 定时任务（默认每 30 分钟）调用
  ``refresh_welcome_prompts``，把结果写入 ``system_settings`` 表。
- **启动时补一次**：服务启动后异步跑一遍，避免冷启动缓存为空。

缓存 key：``welcome_prompts_cache``
存储结构：``{"prompts": [...], "updatedAt": "ISO8601"}``
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from openai import AsyncOpenAI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.interview import InterviewEvaluation
from app.models.probation import Employee
from app.models.recruitment import Candidate, Position
from app.models.settings import SystemSetting
from app.utils.json_utils import extract_json_array_from_text

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_KEY = "welcome_prompts_cache"

# 默认推荐：缓存为空 / LLM 失败时的兜底。
DEFAULT_PROMPTS: List[dict] = [
    {
        "id": "operations",
        "title": "分析招聘运营情况",
        "meta": "查看岗位推进、风险与本周效率",
        "promptText": "分析招聘运营情况",
    },
    {
        "id": "candidate-recommendation",
        "title": "推荐高潜力候选人",
        "meta": "基于岗位匹配、经历和面试记录排序",
        "promptText": "推荐当前最匹配岗位的高潜力候选人",
    },
    {
        "id": "probation-report",
        "title": "入职试用期新人工作情况汇报",
        "meta": "汇总新人任务进展、风险与培养建议",
        "promptText": "入职试用期新人工作情况汇报",
    },
]


def _normalize_prompts(data) -> Optional[List[dict]]:
    """校验并规整 LLM / 缓存里的 prompts，非法则返回 None。"""
    if not isinstance(data, list):
        return None

    cleaned: List[dict] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(data[:3]):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or f"prompt-{idx}").strip() or f"prompt-{idx}"
        if item_id in seen_ids:
            item_id = f"{item_id}-{idx}"
        seen_ids.add(item_id)

        title = str(item.get("title") or "").strip()
        meta = str(item.get("meta") or "").strip()
        prompt_text = str(item.get("promptText") or "").strip()
        if not title or not prompt_text:
            continue
        cleaned.append({
            "id": item_id,
            "title": title[:32],
            "meta": meta[:48],
            "promptText": prompt_text[:200],
        })

    return cleaned if len(cleaned) == 3 else None


async def get_cached_welcome_prompts(db: AsyncSession) -> List[dict]:
    """读缓存。无缓存时返回 DEFAULT_PROMPTS（不触发 LLM）。"""
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == CACHE_KEY)
    )
    row = result.scalar_one_or_none()
    if not row or not isinstance(row.value, dict):
        return DEFAULT_PROMPTS

    prompts = _normalize_prompts(row.value.get("prompts"))
    if prompts is None:
        return DEFAULT_PROMPTS
    return prompts


async def _save_cache(db: AsyncSession, prompts: List[dict]) -> None:
    payload = {
        "prompts": prompts,
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == CACHE_KEY)
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = payload
        row.updated_at = datetime.utcnow()
    else:
        db.add(SystemSetting(key=CACHE_KEY, value=payload))
    await db.flush()


async def _gather_system_snapshot(db: AsyncSession) -> dict:
    """汇总当前系统状态，作为 LLM 推荐的上下文。"""
    cand_rows = await db.execute(
        select(Candidate.status, func.count(Candidate.id)).group_by(Candidate.status)
    )
    candidate_by_status = {row[0] or "unknown": int(row[1]) for row in cand_rows.all()}
    total_candidates = sum(candidate_by_status.values())

    pos_count_row = await db.execute(select(func.count(Position.id)))
    position_count = int(pos_count_row.scalar() or 0)

    emp_rows = await db.execute(
        select(Employee.status, func.count(Employee.id)).group_by(Employee.status)
    )
    employee_by_status = {row[0] or "unknown": int(row[1]) for row in emp_rows.all()}
    total_employees = sum(employee_by_status.values())

    pending_eval_row = await db.execute(
        select(func.count(InterviewEvaluation.id)).where(
            InterviewEvaluation.status == "pending"
        )
    )
    pending_evaluations = int(pending_eval_row.scalar() or 0)

    return {
        "candidateTotal": total_candidates,
        "candidateByStatus": candidate_by_status,
        "positionCount": position_count,
        "employeeTotal": total_employees,
        "employeeByStatus": employee_by_status,
        "pendingInterviewEvaluations": pending_evaluations,
    }


def _build_prompt(snapshot: dict) -> str:
    return (
        "你是招聘系统的「欢迎页推荐 Agent」。请根据当前系统状态，推荐用户**现在最值得做的 3 件事**，"
        "作为欢迎页的快捷入口。\n\n"
        "## 当前系统状态\n"
        f"{json.dumps(snapshot, ensure_ascii=False)}\n\n"
        "## 状态字段说明\n"
        "- candidateByStatus: 候选人按状态分桶，常见状态：job_hunting(求职中) / first_interview(一面中) "
        "/ second_interview(二面中) / passed(已通过) / failed(未通过) / expired(已失效)\n"
        "- employeeByStatus: 试用期员工按状态分桶，常见状态：assessing(考核中) / passed(已转正) "
        "/ failed(未通过) / extended(延期)\n"
        "- pendingInterviewEvaluations: 待处理的面试评估数量\n"
        "- positionCount: 在招岗位数量\n\n"
        "## 推荐规则\n"
        "1. 优先推荐能解决**当前瓶颈**的动作：例如一面中候选人很多 → 推荐安排二面；"
        "待评估面试多 → 推荐做面试评定；试用期员工多 → 推荐做试用期汇报；\n"
        "2. 也可以推荐通用动作：分析招聘运营情况、推荐高潜力候选人、知识库问答等；\n"
        "3. 每条推荐要具体、可执行，title 控制在 16 字以内，meta 是一句话说明（20 字以内），"
        "promptText 是发给 AI Agent 的实际指令（自然语言句子，可以带具体岗位名或数量）；\n"
        "4. 推荐要参考 snapshot 里的真实数字，不要凭空编造；如果某项数据为 0，"
        "不要推荐与该数据强相关的动作。\n\n"
        "## 输出格式\n"
        "只返回纯 JSON 数组，不要任何 markdown 代码块或额外文字：\n"
        "[\n"
        '  {"id": "short-kebab-id", "title": "标题", "meta": "一句话说明", "promptText": "发给 Agent 的指令"},\n'
        "  ... 共 3 条\n"
        "]"
    )


async def _generate_with_llm(db: AsyncSession) -> List[dict]:
    """调 LLM 生成推荐。失败时返回 DEFAULT_PROMPTS。"""
    try:
        snapshot = await _gather_system_snapshot(db)
    except Exception as e:
        logger.warning("welcome_prompt_recommender 收集系统状态失败: %s", e)
        return DEFAULT_PROMPTS

    try:
        from app.services.ai import get_llm_client
        client = get_llm_client()
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": _build_prompt(snapshot)}],
            temperature=0.4,
            max_tokens=512,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(extract_json_array_from_text(content))
    except json.JSONDecodeError as e:
        logger.warning("welcome_prompt_recommender 返回非法 JSON: %s", e)
        return DEFAULT_PROMPTS
    except Exception as e:
        logger.warning("welcome_prompt_recommender LLM 调用失败: %s", e)
        return DEFAULT_PROMPTS

    prompts = _normalize_prompts(data)
    return prompts if prompts is not None else DEFAULT_PROMPTS


async def refresh_welcome_prompts(db: AsyncSession) -> List[dict]:
    """后台刷新：调 LLM → 写缓存。供定时任务 / 启动任务调用。"""
    prompts = await _generate_with_llm(db)
    try:
        await _save_cache(db, prompts)
        logger.info("欢迎页推荐已刷新（%d 条）", len(prompts))
    except Exception as e:
        logger.warning("欢迎页推荐写缓存失败: %s", e)
    return prompts


# 兼容旧调用名：对外只读缓存，不触发 LLM。
async def recommend_welcome_prompts(db: AsyncSession) -> List[dict]:
    return await get_cached_welcome_prompts(db)
