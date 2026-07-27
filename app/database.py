from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=30,
    max_overflow=20,
    pool_pre_ping=True,
    pool_timeout=30,      # 容忍短暂的池子耗尽（如批量上传）
    pool_recycle=1800,    # 30 分钟回收，避免长连接被 PG 端踢掉
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Sync engine for background threads ──────────────────

from sqlalchemy import create_engine as _create_sync_engine
from sqlalchemy.orm import Session as _SyncSession, sessionmaker as _sync_sessionmaker

_sync_url = settings.database_url.replace("+asyncpg", "+psycopg2") if settings.database_url else ""
_sync_engine = _create_sync_engine(
    _sync_url or settings.database_url_sync or "postgresql+psycopg2://postgres:1234@localhost:5432/genie_recruitment",
    echo=False,
    pool_size=10,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=30,
    pool_recycle=1800,
)

SyncSessionFactory = _sync_sessionmaker(
    _sync_engine,
    class_=_SyncSession,
    expire_on_commit=False,
)


def get_sync_db() -> _SyncSession:
    """Get a synchronous DB session (for use in background threads)."""
    return SyncSessionFactory()


async def init_db():
    """Create all tables and apply pending migrations."""
    import app.models  # noqa: F401 — register all ORM tables before create_all

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Auto-migration: add owner_id column to existing knowledge_bases tables
        await conn.run_sync(_run_migrations)


def _add_column_if_missing(connection, table: str, column: str, ddl_type: str) -> None:
    """Add a column if absent; skip entirely when it already exists.

    Querying ``information_schema`` takes no lock on the target table, so a
    normal restart (column already present) doesn't request an ACCESS EXCLUSIVE
    lock and won't block behind an idle-in-transaction session. When the column
    genuinely needs adding, ``SET LOCAL lock_timeout = '3s'`` makes the DDL
    fail fast instead of hanging startup — the caller surfaces a clear error
    rather than waiting out the async timeout.
    """
    from sqlalchemy import text
    exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.columns"
        "  WHERE table_name = :table AND column_name = :column"
        ")"
    ), {"table": table, "column": column}).scalar()
    if exists:
        return
    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    connection.execute(text(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl_type}"
    ))


def _ensure_index_if_missing(connection, index_name: str, table: str, column: str) -> None:
    """Create a btree index on a column when absent (idempotent)."""
    from sqlalchemy import text

    exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM pg_indexes"
        "  WHERE schemaname = 'public' AND indexname = :index_name"
        ")"
    ), {"index_name": index_name}).scalar()
    if exists:
        return
    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    connection.execute(text(
        f"CREATE INDEX {index_name} ON {table} ({column})"
    ))


def _drop_column_if_exists(connection, table: str, column: str) -> None:
    """幂等删除列:列不存在则跳过,避免重启时反复尝试 DROP 报错。"""
    from sqlalchemy import text
    exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.columns"
        "  WHERE table_name = :table AND column_name = :column"
        ")"
    ), {"table": table, "column": column}).scalar()
    if not exists:
        return
    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    connection.execute(text(
        f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}"
    ))


def _migrate_audit_logs(connection) -> None:
    """重建 audit_logs 为新结构（time/actor），旧表若为 actor_name 版本则 drop 重建。"""
    from sqlalchemy import text

    table_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'audit_logs'"
        ")"
    )).scalar()
    if not table_exists:
        return

    has_time = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.columns"
        "  WHERE table_name = 'audit_logs' AND column_name = 'time'"
        ")"
    )).scalar()
    if has_time:
        return

    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    connection.execute(text("DROP TABLE IF EXISTS audit_logs CASCADE"))
    connection.execute(text(
        "CREATE TABLE audit_logs ("
        "  id VARCHAR(36) PRIMARY KEY,"
        "  time VARCHAR(32) NOT NULL,"
        "  actor VARCHAR(64) NOT NULL DEFAULT '系统',"
        "  action VARCHAR(255) NOT NULL,"
        "  section VARCHAR(32),"
        "  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()"
        ")"
    ))


def _ensure_agent_projects_table(connection) -> None:
    """Create agent_projects if missing (covers DBs initialized before the model existed)."""
    from sqlalchemy import text

    exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'agent_projects'"
        ")"
    )).scalar()
    if exists:
        return

    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    connection.execute(text(
        "CREATE TABLE agent_projects ("
        "  id UUID PRIMARY KEY,"
        "  name VARCHAR(64) NOT NULL,"
        "  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),"
        "  updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()"
        ")"
    ))


def _run_migrations(connection):
    """Idempotent schema migrations for existing databases."""
    # v1: Add owner_id column for KB ownership (security)
    _add_column_if_missing(connection, "knowledge_bases", "owner_id", "VARCHAR(36)")
    # v2: 系统设置审计日志表结构（time/actor）
    _migrate_audit_logs(connection)
    # v3: 知识文档来源追踪（简历级联删除）
    _add_column_if_missing(connection, "knowledge_documents", "source_type", "VARCHAR(32) DEFAULT ''")
    _add_column_if_missing(connection, "knowledge_documents", "source_id", "VARCHAR(64) DEFAULT ''")
    # v4: 修正历史脏状态 low_match —— 该状态不在状态机合法词表内，恢复为「待筛选」
    _cleanup_low_match_status(connection)
    # v5: 面试题来源标记（pre_generated/transcript），区分面试出题与面试评定抽取的题目
    _add_column_if_missing(connection, "interview_questions", "source", "VARCHAR(16) NOT NULL DEFAULT 'pre_generated'")
    # v6: 唯一约束加入 source，让两类题目各自独立编号互不冲突
    _migrate_interview_questions_unique_constraint(connection)
    # v7: 支持面试评定的多次上传历史记录
    _migrate_transcript_history(connection)
    # v8: 岗位结构化任职要求(学历/经验/年龄/薪资范围)
    _add_column_if_missing(connection, "positions", "education_requirement", "VARCHAR(32)")
    _add_column_if_missing(connection, "positions", "experience_requirement", "VARCHAR(32)")
    _add_column_if_missing(connection, "positions", "age_requirement", "VARCHAR(32)")
    _add_column_if_missing(connection, "positions", "salary_range", "VARCHAR(64)")
    # v9: 移除无业务含义的 chapter_number 列(仅种子数据排序用途),列表改用 created_at 排序
    _drop_column_if_exists(connection, "positions", "chapter_number")
    # v10: 简历文件 SHA256，用于上传查重
    _add_column_if_missing(connection, "candidates", "resume_file_hash", "VARCHAR(64)")
    _ensure_index_if_missing(
        connection,
        "ix_candidates_resume_file_hash",
        "candidates",
        "resume_file_hash",
    )
    # v11: Agent 对话项目分组
    _ensure_agent_projects_table(connection)
    _add_column_if_missing(connection, "agent_sessions", "project_id", "UUID NULL")
    _ensure_index_if_missing(
        connection,
        "ix_agent_sessions_project_id",
        "agent_sessions",
        "project_id",
    )
    # v12: 面试转写记录存 AI 全方位评定报告
    _add_column_if_missing(connection, "interview_transcripts", "assessment_report", "JSON")
    # v13: 面试转写异步处理进度（后台任务回写 + 前端轮询）。历史记录默认 completed/100，不影响展示。
    _add_column_if_missing(connection, "interview_transcripts", "process_status", "VARCHAR(16) NOT NULL DEFAULT 'completed'")
    _add_column_if_missing(connection, "interview_transcripts", "process_progress", "INTEGER NOT NULL DEFAULT 100")
    _add_column_if_missing(connection, "interview_transcripts", "process_stage", "VARCHAR(64)")
    _add_column_if_missing(connection, "interview_transcripts", "process_message", "TEXT")
    # v14: 招聘需求表单改造 — departments 表 + 字段增减
    _migrate_v14_recruitment_form(connection)
    # v15: 直属负责人支持姓名填写（UUID 可选）
    _migrate_v15_direct_manager_name(connection)
    # v16: 候选人状态词表统一——历史上曾有两套并行词表(状态机词表 vs AI/简历模块词表)
    # 同表混存，见 app/core/state_machine.py TRANSITIONS["candidate"] 的注释。
    _migrate_v16_candidate_status_vocabulary(connection)

    _migrate_v17_confirmation_override(connection)


def _migrate_v17_confirmation_override(connection) -> None:
    """v17: 转正结论人工覆盖留痕（谁、何时、为什么改）。

    前端一直在收集「覆盖 AI 建议的原因」，但后端既无字段也不在白名单里，
    合规链路上等于走了个形式。这里补三列，纯 ADD COLUMN，不动存量数据。
    """
    _add_column_if_missing(connection, "confirmation_reviews", "override_reason", "TEXT")
    _add_column_if_missing(connection, "confirmation_reviews", "override_by", "UUID")
    _add_column_if_missing(connection, "confirmation_reviews", "override_at", "TIMESTAMP")


def _migrate_v15_direct_manager_name(connection) -> None:
    """v15: 招聘需求直属负责人支持手填姓名；direct_manager_id 改为可空。"""
    from sqlalchemy import text
    _add_column_if_missing(connection, "recruitment_requests", "direct_manager_name", "VARCHAR(64)")
    try:
        connection.execute(text(
            "ALTER TABLE recruitment_requests ALTER COLUMN direct_manager_id DROP NOT NULL"
        ))
    except Exception:
        pass


def _migrate_v14_recruitment_form(connection) -> None:
    """v14: 招聘需求表单改造 — 创建 departments 表, 增删列."""
    from sqlalchemy import text
    # 1. Create departments table if absent
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS departments ("
        "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
        "  name VARCHAR(64) UNIQUE NOT NULL,"
        "  description TEXT,"
        "  created_at TIMESTAMP DEFAULT now()"
        ")"
    ))
    # 2. Add new columns to recruitment_requests
    _add_column_if_missing(connection, "recruitment_requests", "department_id", "UUID")
    _add_column_if_missing(connection, "recruitment_requests", "work_experience", "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(connection, "recruitment_requests", "education_requirement", "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(connection, "recruitment_requests", "job_responsibilities", "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(connection, "recruitment_requests", "job_description", "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(connection, "recruitment_requests", "job_requirements", "TEXT NOT NULL DEFAULT ''")
    _add_column_if_missing(connection, "recruitment_requests", "bonus_items", "TEXT")
    # 3. Drop old columns
    _drop_column_if_exists(connection, "recruitment_requests", "position_goal")
    _drop_column_if_exists(connection, "recruitment_requests", "project_experience_req")
    # 4. FK constraint (skip if already exists)
    connection.execute(text(
        "DO $$ BEGIN"
        "  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_recruitment_requests_department') THEN"
        "    ALTER TABLE recruitment_requests ADD CONSTRAINT fk_recruitment_requests_department"
        "      FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE SET NULL;"
        "  END IF;"
        " END $$"
    ))


def _migrate_transcript_history(connection) -> None:
    """支持面试评定上传历史记录：
    1. 删除 interview_transcripts 上的 (candidate_id, round) 唯一约束，允许同轮多次上传。
    2. 给 interview_transcripts 加 filename 列，记录原始文件名用于历史展示。
    3. 给 interview_questions 加 transcript_id 列，关联到具体转写记录，删除记录时级联删除题目与评分。
    幂等：列/约束已存在则跳过。
    """
    from sqlalchemy import text

    transcripts_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'interview_transcripts'"
        ")"
    )).scalar()
    if not transcripts_exists:
        return

    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    # 1. 删除旧唯一约束（PostgreSQL 默认自动生成的名称）
    connection.execute(text(
        "ALTER TABLE interview_transcripts "
        "DROP CONSTRAINT IF EXISTS interview_transcripts_candidate_id_round_key"
    ))

    # 2. 加 filename 列
    _add_column_if_missing(connection, "interview_transcripts", "filename", "VARCHAR(255)")

    # 3. 给 interview_questions 加 transcript_id 列（带外键级联删除）
    questions_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.columns"
        "  WHERE table_name = 'interview_questions' AND column_name = 'transcript_id'"
        ")"
    )).scalar()
    if not questions_exists:
        connection.execute(text(
            "ALTER TABLE interview_questions "
            "ADD COLUMN transcript_id UUID NULL REFERENCES interview_transcripts(id) ON DELETE CASCADE"
        ))


def _cleanup_low_match_status(connection) -> None:
    """把 candidates.status = 'low_match' 的历史脏数据恢复为 'pending_screen'。

    low_match 曾被用作 AI 评分低于阈值时的标记，但它不在状态机合法词表
    （app/core/state_machine.py TRANSITIONS["candidate"]）中，会在界面上
    原样显示成英文串。低匹配信息已通过 screening_ai_score/score 保留，
    状态本身回归合法值即可。
    """
    from sqlalchemy import text

    table_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'candidates'"
        ")"
    )).scalar()
    if not table_exists:
        return

    connection.execute(text(
        "UPDATE candidates SET status = 'pending_screen' WHERE status = 'low_match'"
    ))


def _migrate_v16_candidate_status_vocabulary(connection) -> None:
    """把 candidates.status 里残留的旧词表值统一改写为状态机合法词表。

    历史上 AI 对话模块 / 简历上传接口曾使用另一套词表
    （job_hunting/passed/first_interview/second_interview/failed/expired），
    与 app/core/state_machine.py 的状态机词表
    （new/parsed/pending_screen/invited/round1/round2/pending_offer/hired/
    talent_pool/rejected）同表混存。此外还有更早的历史遗留值
    （pending_interview/onboarded/probation/offer_pending，其中 offer_pending
    是 pending_offer 的拼写颠倒）。这些非法值不在状态机迁移表内，会导致：
    - 前端状态徽章渲染为 undefined（标签表只认状态机词表）
    - AI 上传的简历进不了筛选流程（scoring.py 只放行 new/parsed）
    - 二面通过的候选人进不了录用审批（offer.py 要求 pending_offer）
    此迁移是幂等的，每次启动都会执行，不会影响已经是合法值的行。

    passed 的映射存在语义上的一次性判断：本项目 AI 提示词(prompts/system.txt)
    明确 passed 表示"初筛已通过"而非"全部面试通过/已入职"，故统一映射为
    invited（初筛通过待安排面试），而不是 hired。
    """
    from sqlalchemy import text

    table_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'candidates'"
        ")"
    )).scalar()
    if not table_exists:
        return

    # 旧值 -> 状态机合法值。执行顺序无关紧要，每条各自独立生效。
    mapping = {
        "job_hunting": "pending_screen",
        "passed": "invited",
        "first_interview": "round1",
        "second_interview": "round2",
        "failed": "rejected",
        "expired": "rejected",
        "pending_interview": "invited",
        "onboarded": "hired",
        "probation": "hired",
        "offer_pending": "pending_offer",
    }
    for old_status, new_status in mapping.items():
        connection.execute(
            text("UPDATE candidates SET status = :new_status WHERE status = :old_status"),
            {"new_status": new_status, "old_status": old_status},
        )


def _migrate_interview_questions_unique_constraint(connection) -> None:
    """把 interview_questions 的唯一约束从 (candidate_id, round, index_num)
    改为 (candidate_id, round, source, index_num)，让面试出题与面试评定转写抽取
    两类题目各自独立编号，互不冲突。幂等：新约束已存在则跳过。
    """
    from sqlalchemy import text

    table_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.tables"
        "  WHERE table_schema = 'public' AND table_name = 'interview_questions'"
        ")"
    )).scalar()
    if not table_exists:
        return

    new_constraint_exists = connection.execute(text(
        "SELECT EXISTS ("
        "  SELECT 1 FROM information_schema.table_constraints"
        "  WHERE table_schema = 'public'"
        "    AND table_name = 'interview_questions'"
        "    AND constraint_name = 'uq_interview_questions_cand_round_source_idx'"
        ")"
    )).scalar()
    if new_constraint_exists:
        return

    connection.execute(text("SET LOCAL lock_timeout = '3s'"))
    # 删除旧约束（PostgreSQL 默认自动生成的名称）
    connection.execute(text(
        "ALTER TABLE interview_questions "
        "DROP CONSTRAINT IF EXISTS interview_questions_candidate_id_round_index_num_key"
    ))
    connection.execute(text(
        "ALTER TABLE interview_questions "
        "ADD CONSTRAINT uq_interview_questions_cand_round_source_idx "
        "UNIQUE (candidate_id, round, source, index_num)"
    ))
