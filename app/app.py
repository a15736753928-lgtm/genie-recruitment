from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import io
import os
import time
from app.config import get_settings
from app.log_config import configure_logging

# ── Early UTF-8 setup (before uvicorn takes over) ──
# On Windows the default stdout/stderr encoding is often GBK and chokes on
# emoji/box-drawing characters used in startup logs. Force UTF-8 so logs
# render correctly regardless of the system codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, io.UnsupportedOperation):
    pass


# 强制 Rich / uvicorn 等库启用颜色（它们依赖 FORCE_COLOR 环境变量）
os.environ.setdefault("FORCE_COLOR", "1")

# 启动早期即初始化彩色日志（log_config.py 的 formatter 是唯一实现）
configure_logging()

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
    from app.infrastructure.startup_checks import run_startup_checks, print_check_summary, CheckStatus

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
        from app.services.recruitment.seed_data import seed_all
        await seed_all()
        logger.info("✓ 初始数据就绪 (%s)", _elapsed())
    except Exception as e:
        logger.warning("⚠ 初始数据跳过 (可能已存在): %s (%s)", e, _elapsed())

    # ── Step 3: 定时任务（清理 / 面试提醒 / 欢迎页推荐）──
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.database import async_session_factory
        from app.services.system.data_retention import cleanup_expired

        async def _scheduled_cleanup():
            async with async_session_factory() as session:
                try:
                    await cleanup_expired(session)
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    logger.warning("定时数据清理失败: %s", e)

        async def _scheduled_interview_reminders():
            from app.services.system.notification import check_interview_reminders
            async with async_session_factory() as session:
                try:
                    n = await check_interview_reminders(session)
                    await session.commit()
                    if n:
                        logger.info("面试提醒已记录 %d 条", n)
                except Exception as e:
                    await session.rollback()
                    logger.warning("面试提醒任务失败: %s", e)

        async def _scheduled_welcome_prompts():
            """后台默默刷新欢迎页推荐缓存，用户无感。"""
            from app.services.ai.welcome_prompt_recommender import refresh_welcome_prompts
            async with async_session_factory() as session:
                try:
                    await refresh_welcome_prompts(session)
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    logger.warning("欢迎页推荐定时刷新失败: %s", e)

        from datetime import datetime, timedelta

        scheduler = AsyncIOScheduler()
        scheduler.add_job(_scheduled_cleanup, "cron", hour=2, minute=0, id="data_retention")
        scheduler.add_job(_scheduled_interview_reminders, "cron", hour=9, minute=0, id="interview_reminders")
        # 每 30 分钟刷新一次；启动 15 秒后先跑一遍（不阻塞启动，用户无感）
        scheduler.add_job(
            _scheduled_welcome_prompts,
            "interval",
            minutes=30,
            id="welcome_prompts",
            next_run_time=datetime.now() + timedelta(seconds=15),
        )
        scheduler.start()
        logger.info(
            "✓ 定时任务已注册 (清理 02:00 / 面试提醒 09:00 / 欢迎页推荐 每30分钟) (%s)",
            _elapsed(),
        )
    except Exception as e:
        logger.warning("⚠ 定时任务未启动（可手动调用 POST /api/settings/cleanup）: %s", e)

    # ── Warm up LLM Router ──
    try:
        from app.services.ai import reload_llm_config
        from app.database import async_session_factory
        async with async_session_factory() as session:
            await reload_llm_config(session)
            logger.info("✓ LLM Router 已预热 (%s)", _elapsed())
    except Exception as e:
        logger.warning("⚠ LLM Router 预热失败: %s (%s)", e, _elapsed())

    # ── Ready ──
    logger.info("=" * 55)
    logger.info("  全部配置加载完毕，开始接收请求 (%s)", _elapsed())
    logger.info("    地址: http://%s:%d" % (settings.app_host, settings.app_port))
    logger.info("    文档: http://127.0.0.1:%d/docs" % settings.app_port)
    logger.info("    健康: http://127.0.0.1:%d/api/health" % settings.app_port)
    logger.info("=" * 55)

    yield

    # ── Shutdown (order matters: scheduler → pools → engine) ──
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=True)  # wait=True: 等所有 job 彻底结束
        except Exception:
            pass

    # Dispose SQLAlchemy engine so connection pools are released.
    # Without this, asyncpg connections may linger and keep the process alive
    # after uvicorn signals "shutdown complete", causing the "need to restart
    # twice" issue in PyCharm.
    try:
        from app.database import engine
        await engine.dispose()
    except Exception:
        pass

    logger.info("Genie 招聘系统 — 正在关闭...")


app = FastAPI(
    title="Genie 招聘系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Trace ID (before everything — injects per-request trace_id into contextvars) ──
from app.middleware.trace import TraceMiddleware
app.add_middleware(TraceMiddleware)

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


from app.core.security import AuthError, PermissionError_
from app.utils.responses import ok, fail, not_found

@app.exception_handler(AuthError)
@app.exception_handler(PermissionError_)
async def api_error_handler(request: Request, exc):
    """AuthError(401) / PermissionError_(403) → 统一 JSON 信封。

    作为 HTTPException 子类的专用 handler，FastAPI 在依赖注入阶段即可原生拦截，
    不会被 ExceptionGroup 包装后泄漏到 uvicorn ERROR 日志。"""
    return JSONResponse(
        status_code=exc.status_code,
        content=fail(exc.status_code, exc.detail),
    )


from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """请求体/参数校验失败(422) → 统一 JSON 信封。

    FastAPI 默认返回 {"detail": [...]}，没有 code 字段，会打穿前端 client.ts
    的信封解析分支，最终弹出一个空白 toast，用户完全看不到错误原因。
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        # loc 形如 ("body", "role")，去掉开头的 body/query 前缀更贴近用户视角
        loc = [str(p) for p in first.get("loc", []) if p not in ("body", "query", "path")]
        field = ".".join(loc) if loc else "请求参数"
        message = f"参数校验失败: {field} {first.get('msg', '')}".strip()
    else:
        message = "参数校验失败"
    logger.warning("422 on %s %s: %s", request.method, request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content=fail(422, message),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # AuthError / PermissionError_ 已由专用 handler 拦截，此处不再需要 isinstance 检查
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
            content=not_found("记录不存在或 ID 格式无效"),
        )
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    # 中文提示: client.ts 对真实 5xx 会自行替换成中文文案，但 api/services/agent.ts
    # 是直接把 envelope.message 原样弹 toast——英文字符串会导致同一个后端 500
    # 在两个前端客户端里一个中文一个英文，此处统一在源头改成中文即可两边一致。
    return JSONResponse(
        status_code=500,
        content=fail(500, "服务器内部错误，请稍后重试或联系管理员"),
    )


# ── Register routers ──
# P0: 鉴权 & 用户管理 & 状态机/异常队列
from app.api.auth import auth as auth_router, users as users_router
from app.api.system import state_exceptions
# 招聘管理
from app.api.recruitment import positions, resumes
from app.api.recruitment import requests as recruitment_requests_router, scoring as resume_scoring_router
# 人才评估
from app.api.talent import interview, probation, performance, phase1_interview, phase2 as phase2_probation, phase4 as phase4_talent
from app.api.talent import offer as offer_router_module
# 知识库
from app.api.knowledge import knowledge_base as knowledge, rag
# AI 功能 — 唯一 Agent 入口（v1/v2/v3 已合并，agent_chat_v2 已删除）
from app.api.ai import agent_chat as ai_agent
# 第三期: 任务积分
from app.api.points import phase3 as points_router
# 系统管理
from app.api.system import dashboard, settings as settings_router, export as export_router
from app.api.system import llm_config as llm_config_router

# P0 先注册(鉴权白名单依赖路径匹配)
app.include_router(auth_router.router, prefix="/api")
app.include_router(users_router.router, prefix="/api")
# roles 路由附在 users_router 模块里,单独注册
from app.api.auth.users import roles_router
app.include_router(roles_router, prefix="/api")
app.include_router(state_exceptions.router, prefix="/api")
app.include_router(recruitment_requests_router.router, prefix="/api")
app.include_router(resume_scoring_router.router, prefix="/api")
app.include_router(positions.router, prefix="/api")
app.include_router(resumes.router, prefix="/api")
app.include_router(interview.router, prefix="/api")
app.include_router(phase1_interview.router, prefix="/api")
app.include_router(probation.router, prefix="/api")
app.include_router(offer_router_module.router, prefix="/api")
app.include_router(phase2_probation.router, prefix="/api")  # 第二期: 试用期+培训+带教
app.include_router(points_router.router, prefix="/api")    # 第三期: 任务积分奖惩申诉
app.include_router(phase4_talent.router, prefix="/api")    # 第四期: 人才池/晋级/期权/看板
app.include_router(performance.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(ai_agent.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(export_router.router, prefix="/api")
app.include_router(llm_config_router.router)
app.include_router(rag.router)  # RAG endpoints (prefix defined in router)


@app.get("/api/health")
async def health():
    return ok({"status": "running"})


