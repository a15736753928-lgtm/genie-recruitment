"""Seed database with default categories, positions, settings, and RBAC."""
import uuid
from datetime import date
from sqlalchemy import select, func
from app.database import async_session_factory, Base, engine
from app.models.recruitment import Position, Department
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


async def seed_departments(db) -> None:
    """幂等创建 10 个默认部门。"""
    names = [
        ("技术部", "负责软件开发、架构设计和技术创新"),
        ("产品部", "负责产品规划、需求分析和用户体验"),
        ("设计部", "负责UI/UX设计、品牌视觉和交互设计"),
        ("市场部", "负责市场推广、品牌营销和渠道拓展"),
        ("运营部", "负责业务运营、客户服务和流程优化"),
        ("财务部", "负责财务管理、预算控制和成本分析"),
        ("人力资源部", "负责招聘、培训、绩效和员工关系"),
        ("数据平台部", "负责数据平台建设、数据治理和分析"),
        ("AI 平台部", "负责AI能力建设、模型训练和算法优化"),
        ("综合管理部", "负责行政、法务和综合事务管理"),
    ]
    for name, desc in names:
        exists = await db.execute(
            select(Department).where(Department.name == name)
        )
        if not exists.scalar_one_or_none():
            db.add(Department(name=name, description=desc))
    await db.flush()


async def seed_all():
    # Also seed V2 positions from the handbook
    from app.services.recruitment.seed_data_v2 import seed_positions_v2

    async with async_session_factory() as db:
        create_missing_positions = await _should_create_seed_positions(db)
        await seed_departments(db)
        await db.commit()

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

        # ── PDF 业务规则默认值(各模块从此读取,不硬编码)──
        BUSINESS_RULE_DEFAULTS = {
            "resume_grade_thresholds": {"A": 85, "B": 70, "C": 60},   # D<60
            "ai_confidence_threshold": 0.6,
            "interviewer_score_gap": 20,
            "r1_conclusion_thresholds": {"priority": 85, "advance": 75, "review": 65},
            "conversion_thresholds": {"excellent": 85, "normal": 75, "conditional": 65},
            "offer_thresholds": {"priority": 85, "recommend": 75, "conditional": 65, "reserve": 55},
            "standard_question_min_ratio": 0.6,
            "major_penalty_threshold": 200,
            "task_level_points": {
                "S": [500, 1000], "A": [200, 500], "B": [80, 200], "C": [20, 80], "D": [5, 20]
            },
        }
        for key, value in BUSINESS_RULE_DEFAULTS.items():
            existing = await db.execute(select(SystemSetting).where(SystemSetting.key == key))
            if not existing.scalar_one_or_none():
                db.add(SystemSetting(key=key, value=value))

        await _ensure_positions_seeded_flag(db)
        await db.commit()

    # ── 项目需求(人员推荐组队)──────────────────
    await seed_projects()

    # ── RBAC: roles + initial users ──────────────────
    await seed_roles()


async def seed_projects() -> None:
    """播种示例项目需求(供项目人员推荐页展示)。"""
    from app.models.phase4 import Project
    default_projects = [
        {"name": "人才管理系统二期", "required_skills": ["React", "TypeScript"],
         "required_level": "L3", "headcount": 2, "department": "技术部"},
        {"name": "618 活动复盘优化", "required_skills": ["活动策划", "数据分析"],
         "required_level": "L3", "headcount": 1, "department": "运营部"},
        {"name": "客户 B AI 化方案", "required_skills": ["企业问诊", "流程分析"],
         "required_level": "L4", "headcount": 1, "department": "咨询部"},
    ]
    async with async_session_factory() as db:
        for proj in default_projects:
            exists = await db.execute(select(Project).where(Project.name == proj["name"]))
            if not exists.scalar_one_or_none():
                db.add(Project(**proj))
        await db.commit()


async def seed_roles() -> None:
    """幂等写入角色、权限点、初始账号(admin + hr01)。

    全部操作对已有数据跳过,适合每次启动调用。
    角色编码以原型口径为准:ceo/hr/manager/interviewer/mentor/project_lead/employee/admin(+equity_committee)。
    """
    from app.models.auth import User, Role, UserRole, RolePermission
    from app.core.permissions import ROLES, ROLE_PERMISSIONS
    from app.core.security import hash_password

    async with async_session_factory() as db:
        # ── 1. 写入角色 ──
        for code, (name, description) in ROLES.items():
            row = await db.execute(select(Role).where(Role.code == code))
            if row.scalar_one_or_none() is None:
                db.add(Role(code=code, name=name, description=description))
        await db.flush()

        # ── 2. 写入权限点 ──
        for role_code, perm_keys in ROLE_PERMISSIONS.items():
            for key in perm_keys:
                row = await db.execute(
                    select(RolePermission).where(
                        RolePermission.role_code == role_code,
                        RolePermission.permission_key == key,
                    )
                )
                if row.scalar_one_or_none() is None:
                    db.add(RolePermission(role_code=role_code, permission_key=key))
        await db.flush()

        # ── 3. 初始账号 ──
        INIT_USERS = [
            {"username": "admin", "password": "admin@123", "display_name": "系统管理员", "role": "admin"},
            {"username": "hr01", "password": "hr@123", "display_name": "HR 专员", "role": "hr"},
        ]
        for u_info in INIT_USERS:
            row = await db.execute(select(User).where(User.username == u_info["username"]))
            if row.scalar_one_or_none() is None:
                new_user = User(
                    username=u_info["username"],
                    password_hash=hash_password(u_info["password"]),
                    display_name=u_info["display_name"],
                    must_change_password=True,
                )
                db.add(new_user)
                await db.flush()
                db.add(UserRole(user_id=new_user.id, role_code=u_info["role"]))
        await db.commit()
