"""Seed database with root user, default categories, and default settings."""
import uuid
from datetime import date
from sqlalchemy import select
from app.database import async_session_factory, Base, engine
from app.models.user import User
from app.models.candidate import Position
from app.models.knowledge import KnowledgeCategory
from app.models.settings import SystemSetting
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
    "autoTrainKnowledge": False,
    "autoTrainSchedule": "0 2 * * 0",
    "recallThreshold": 0.75,
    "notifyNewResume": True,
    "notifyInterviewReminder": True,
    "notifyOfferPending": True,
    "notifyProbationRisk": True,
    "notifyPerformanceDue": True,
    "dataRetentionDays": 365,
    "exportFormat": "xlsx",
    "webhookEnabled": False,
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
    async with async_session_factory() as db:
        # ── Root user ────────────────────────────────
        result = await db.execute(select(User).where(User.username == "admin"))
        if not result.scalar_one_or_none():
            user = User(
                id=uuid.uuid4(),
                username="admin",
                password_hash=pwd_context.hash("12345678"),
                display_name="HR 管理员",
                email="admin@genietech.com",
                role="hr_admin",
                department="人力资源部",
                is_active=True,
            )
            db.add(user)

        # ── Default categories ───────────────────────
        for cat in DEFAULT_CATEGORIES:
            result = await db.execute(
                select(KnowledgeCategory).where(KnowledgeCategory.key == cat["key"])
            )
            if not result.scalar_one_or_none():
                db.add(KnowledgeCategory(**cat))

        # ── Default positions ────────────────────────
        for name in DEFAULT_POSITIONS:
            result = await db.execute(select(Position).where(Position.name == name))
            if not result.scalar_one_or_none():
                db.add(Position(name=name))

        # ── Default settings ─────────────────────────
        result = await db.execute(select(SystemSetting).where(SystemSetting.key == "global"))
        if not result.scalar_one_or_none():
            db.add(SystemSetting(key="global", value=DEFAULT_SETTINGS))

        await db.commit()
