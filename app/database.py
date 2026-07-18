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


def _run_migrations(connection):
    """Idempotent schema migrations for existing databases."""
    # v1: Add owner_id column for KB ownership (security)
    _add_column_if_missing(connection, "knowledge_bases", "owner_id", "VARCHAR(36)")
    # v2: 系统设置审计日志表结构（time/actor）
    _migrate_audit_logs(connection)
    # v3: 知识文档来源追踪（简历级联删除）
    _add_column_if_missing(connection, "knowledge_documents", "source_type", "VARCHAR(32) DEFAULT ''")
    _add_column_if_missing(connection, "knowledge_documents", "source_id", "VARCHAR(64) DEFAULT ''")
