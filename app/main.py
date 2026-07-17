from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os
from app.config import get_settings
from app.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("startup")

settings = get_settings()

_START_TIME = time.time()

def _elapsed() -> str:
    return f"{time.time() - _START_TIME:.1f}s"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[1/4] 开始初始化数据库... (%s)", _elapsed())
    try:
        await init_db()
        logger.info("[1/4] 数据库初始化完成 (%s)", _elapsed())
    except Exception as e:
        logger.error("[1/4] 数据库初始化失败: %s", e, exc_info=True)
        raise

    logger.info("[2/4] 开始填充初始数据... (%s)", _elapsed())
    try:
        from app.services.seed import seed_all
        await seed_all()
        logger.info("[2/4] 初始数据填充完成 (%s)", _elapsed())
    except Exception as e:
        logger.warning("[2/4] 初始数据填充跳过（可能已存在）: %s (%s)", e, _elapsed())

    logger.info("[3/4] 注册路由... (%s)", _elapsed())
    from app.routers import auth, positions, resumes, interview, probation, performance
    from app.routers import knowledge, dashboard, ai_agent, settings as settings_router
    from app.routers import rag

    app.include_router(auth.router, prefix="/api")
    app.include_router(positions.router, prefix="/api")
    app.include_router(resumes.router, prefix="/api")
    app.include_router(interview.router, prefix="/api")
    app.include_router(probation.router, prefix="/api")
    app.include_router(performance.router, prefix="/api")
    app.include_router(knowledge.router, prefix="/api")
    app.include_router(dashboard.router, prefix="/api")
    app.include_router(ai_agent.router, prefix="/api")
    app.include_router(settings_router.router, prefix="/api")
    app.include_router(rag.router)
    logger.info("[3/4] 路由注册完成 (%s)", _elapsed())

    logger.info("[4/4] 应用启动完成 — 等待请求 (%s)", _elapsed())
    yield
    logger.info("应用关闭 (%s)", _elapsed())


app = FastAPI(
    title="Genie 招聘系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler to ensure ApiResponse format
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": str(exc), "data": None},
    )


@app.get("/api/health")
async def health():
    return {"code": 0, "message": "ok", "data": {"status": "running"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True, reload_dirs=["app"])
