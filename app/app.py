from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import sys
import io
from app.config import get_settings

# ── Early logging setup (before uvicorn takes over) ──
# On Windows the default stdout/stderr encoding is often GBK and chokes on
# emoji/box-drawing characters used in startup logs. Force UTF-8 so logs
# render correctly regardless of the system codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, io.UnsupportedOperation):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("genie.startup")

settings = get_settings()
_START_TIME = time.time()

def _elapsed() -> str:
    return f"+{time.time() - _START_TIME:.1f}s"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("Genie 招聘系统 v1.0 — 启动中")
    logger.info("=" * 55)

    # ── Step 1: Run all startup health checks ──
    from app.core.startup_checks import run_startup_checks, print_check_summary, CheckStatus

    results = await run_startup_checks(settings)
    print_check_summary(results)

    # If ANY check failed, refuse to start
    failures = [r for r in results if r.status == CheckStatus.FAIL]
    if failures:
        msg = "以下依赖未就绪，拒绝启动:\n" + "\n".join(
            f"  ✗ {f.name}: {f.detail}" for f in failures
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # ── Step 2: Seed data ──
    logger.info("检查/填充初始数据 (示例岗位)...")
    try:
        from app.services.seed import seed_all
        await seed_all()
        logger.info("✓ 初始数据就绪 (%s)", _elapsed())
    except Exception as e:
        logger.warning("⚠ 初始数据跳过 (可能已存在): %s (%s)", e, _elapsed())

    # ── Step 3: 数据保留定时清理（每天 02:00）──
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.database import async_session_factory
        from app.services.data_retention import cleanup_expired

        async def _scheduled_cleanup():
            async with async_session_factory() as session:
                try:
                    await cleanup_expired(session)
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    logger.warning("定时数据清理失败: %s", e)

        async def _scheduled_interview_reminders():
            from app.services.notification import check_interview_reminders
            async with async_session_factory() as session:
                try:
                    n = await check_interview_reminders(session)
                    await session.commit()
                    if n:
                        logger.info("面试提醒已记录 %d 条", n)
                except Exception as e:
                    await session.rollback()
                    logger.warning("面试提醒任务失败: %s", e)

        scheduler = AsyncIOScheduler()
        scheduler.add_job(_scheduled_cleanup, "cron", hour=2, minute=0, id="data_retention")
        scheduler.add_job(_scheduled_interview_reminders, "cron", hour=9, minute=0, id="interview_reminders")
        scheduler.start()
        logger.info("✓ 定时任务已注册 (清理 02:00 / 面试提醒 09:00) (%s)", _elapsed())
    except Exception as e:
        logger.warning("⚠ 定时任务未启动（可手动调用 POST /api/settings/cleanup）: %s", e)

    # ── Ready ──
    logger.info("=" * 55)
    logger.info("  全部配置加载完毕，开始接收请求 (%s)", _elapsed())
    logger.info("    地址: http://%s:%d" % (settings.app_host, settings.app_port))
    logger.info("    文档: http://127.0.0.1:%d/docs" % settings.app_port)
    logger.info("    健康: http://127.0.0.1:%d/api/health" % settings.app_port)
    logger.info("=" * 55)

    yield

    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
    logger.info("Genie 招聘系统 — 正在关闭...")


app = FastAPI(
    title="Genie 招聘系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request logger (before CORS so we see everything) ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %d (%.0fms)",
        request.method,
        request.url.path + ("?" + request.url.query if request.url.query else ""),
        response.status_code,
        duration_ms,
    )
    return response

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Invalid UUID path/query params surface as asyncpg DataError or
    # sqlalchemy DBAPIError. Convert to a clean 400/404 instead of a 500 so
    # the frontend gets a meaningful message and the server logs stay clean.
    msg = str(exc)
    exc_type = type(exc).__name__
    is_invalid_uuid = (
        "invalid UUID" in msg
        or "invalid input for query argument" in msg
        or "DataError" in exc_type
        or "DataError" in msg
    )
    if is_invalid_uuid:
        logger.warning("Bad UUID input on %s %s: %s", request.method, request.url.path, msg)
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "记录不存在或 ID 格式无效", "data": None},
        )
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal server error", "data": None},
    )


# ── Register routers ──
from app.routers import positions, resumes, interview, probation, performance
from app.routers import knowledge, dashboard, ai_agent, settings as settings_router
from app.routers import rag, export as export_router

app.include_router(positions.router, prefix="/api")
app.include_router(resumes.router, prefix="/api")
app.include_router(interview.router, prefix="/api")
app.include_router(probation.router, prefix="/api")
app.include_router(performance.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(ai_agent.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(export_router.router, prefix="/api")
app.include_router(rag.router)  # RAG endpoints (prefix defined in router)


@app.get("/api/health")
async def health():
    return {"code": 0, "message": "ok", "data": {"status": "running"}}


