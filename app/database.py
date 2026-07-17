from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=10,       # fail fast when pool is exhausted (vs. hang forever)
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
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    pool_timeout=10,
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


def _run_migrations(connection):
    """Idempotent schema migrations for existing databases."""
    from sqlalchemy import text
    # v1: Add owner_id column for KB ownership (security)
    connection.execute(text(
        "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS owner_id VARCHAR(36)"
    ))
