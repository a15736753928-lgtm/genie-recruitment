"""Genie Recruitment System - Entry Point"""
import uvicorn
from app.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        # Scope reload-watching to the app/ source directory only. Without
        # this, uvicorn watches the whole project root and runtime files
        # (uploads/, cache/, milvus_lite.db) can trigger spurious reloads
        # that drop all in-flight connections.
        reload=True,
        reload_dirs=["app"],
        log_level="info",
    )
