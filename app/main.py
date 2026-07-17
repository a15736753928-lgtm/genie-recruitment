from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.config import get_settings
from app.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Seed root user and default data (non-fatal on error)
    try:
        from app.services.seed import seed_all
        await seed_all()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Seed skipped (data may already exist): {e}")
    yield


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


# Register routers
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
app.include_router(rag.router)  # RAG endpoints (prefix defined in router)


@app.get("/api/health")
async def health():
    return {"code": 0, "message": "ok", "data": {"status": "running"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
