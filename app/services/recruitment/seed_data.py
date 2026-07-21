"""Seed database with default categories, positions, and settings."""
import uuid
from datetime import date
from sqlalchemy import select, func
from app.database import async_session_factory, Base, engine
from app.models.recruitment import Position
from app.models.knowledge import KnowledgeCategory
from app.models.settings import SystemSetting

# 标记岗位已完成首次初始化；之后启动不再按名称重建用户已删除的岗位
POSITIONS_SEEDED_KEY = "positions_seeded"


async def _should_create_seed_positions(db) -> bool:
    """仅在空库首次安装时创建种子岗位；已有数据或已标记则不再补建。"""
    flag = await db.execute(
        select(SystemSetting).where(SystemSetting.key == POSITIONS_SEEDED_KEY)
    )
    if flag.scalar_one_or_none():
        return False
    count_result = await db.execute(select(func.count()).select_from(Position))
    return (count_result.scalar() or 0) == 0


async def _ensure_positions_seeded_flag(db) -> None:
    existing = await db.execute(
        select(SystemSetting).where(SystemSetting.key == POSITIONS_SEEDED_KEY)
    )
    if existing.scalar_one_or_none() is None:
        db.add(SystemSetting(key=POSITIONS_SEEDED_KEY, value={"done": True}))

DEFAULT_SETTINGS = {
    "systemName": "Genie 智能招聘系统",
    "companyName": "Genie Tech",
    "contactEmail": "hr@genietech.com",
    "defaultQuarter": "2026-Q3",
    "autoParseResume": True,
    "minMatchScore": 70,
    "defaultPositionId": "",
    "offerApprovalRequired": True,
    "defaultQuestionCount": 8,
    "defaultScoringMode": "ai",
    "passScoreThreshold": 75,
    "allowAudioUpload": True,
    "probationDays": 90,
    "defaultProbationTasks": 5,
    "aiResumeAnalysis": True,
    "aiQuestionGeneration": True,
    "aiInterviewScoring": True,
    "recallThreshold": 0.75,
    "kbDefaultTopK": 10,
    "kbRerankEnabled": True,
    "kbOcrEnabled": True,
    "notifyNewResume": True,
    "notifyInterviewReminder": True,
    "notifyOfferPending": True,
    "notifyProbationRisk": True,
    "notifyPerformanceDue": True,
    "dataRetentionDays": 365,
    "exportFormat": "xlsx",
    "webhookEnabled": False,
    "webhookUrl": "",
}

DEFAULT_CATEGORIES = [
    {"key": "interview", "title": "面试题库", "parent_key": None, "sort_order": 0},
    {"key": "interview-frontend", "title": "前端面试题", "parent_key": "interview", "sort_order": 1},
    {"key": "interview-backend", "title": "后端面试题", "parent_key": "interview", "sort_order": 2},
    {"key": "interview-algorithm", "title": "算法面试题", "parent_key": "interview", "sort_order": 3},
    {"key": "interview-system", "title": "系统设计", "parent_key": "interview", "sort_order": 4},
    {"key": "standard", "title": "标准文档", "parent_key": None, "sort_order": 1},
    {"key": "standard-hr", "title": "HR制度", "parent_key": "standard", "sort_order": 1},
    {"key": "standard-tech", "title": "技术规范", "parent_key": "standard", "sort_order": 2},
    {"key": "rule", "title": "规章制度", "parent_key": None, "sort_order": 2},
    {"key": "data", "title": "数据资料", "parent_key": None, "sort_order": 3},
]

DEFAULT_POSITIONS = [
    "前端开发工程师",
    "后端开发工程师",
    "全栈开发工程师",
    "测试工程师",
    "产品经理",
    "UI/UX 设计师",
    "数据分析师",
    "运维工程师",
]


async def seed_all():
    # Also seed V2 positions from the handbook
    from app.services.recruitment.seed_data_v2 import seed_positions_v2

    async with async_session_factory() as db:
        create_missing_positions = await _should_create_seed_positions(db)

    await seed_positions_v2(create_missing=create_missing_positions)

    async with async_session_factory() as db:
        # ── Default categories ───────────────────────
        for cat in DEFAULT_CATEGORIES:
            result = await db.execute(
                select(KnowledgeCategory).where(KnowledgeCategory.key == cat["key"])
            )
            if not result.scalar_one_or_none():
                db.add(KnowledgeCategory(**cat))

        # ── Default positions（仅首次空库创建，避免删掉后又被启动脚本写回）──
        if create_missing_positions:
            for name in DEFAULT_POSITIONS:
                result = await db.execute(select(Position).where(Position.name == name))
                if not result.scalar_one_or_none():
                    db.add(Position(name=name))

        # ── Default settings ─────────────────────────
        result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
        if not result.scalar_one_or_none():
            db.add(SystemSetting(key="global", value=DEFAULT_SETTINGS))

        await _ensure_positions_seeded_flag(db)
        await db.commit()
