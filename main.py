"""Genie 招聘系统 - 启动入口"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import uvicorn
from app.config import get_settings

settings = get_settings()

# reload=True spawns a worker subprocess. PyCharm's Stop button on Windows
# only kills the parent reloader, leaving the worker orphaned on port 8000 —
# so the next Run collides with the orphan and "does nothing" until you click
# again. Disable reload under PyCharm (it sets PYCHARM_HOSTED=1); keep it on
# in the terminal where Ctrl+C tears down both processes cleanly.
_in_pycharm = os.getenv("PYCHARM_HOSTED") == "1"
_reload = not _in_pycharm

if __name__ == "__main__":
    print(f"启动 Genie 招聘系统 API...")
    print(f"地址: http://127.0.0.1:{settings.app_port}")
    print(f"文档: http://127.0.0.1:{settings.app_port}/docs")
    print(f"自动重载: {'开（终端模式）' if _reload else '关（PyCharm 模式，改代码后 Ctrl+F5 重跑）'}")
    uvicorn.run(
        "app.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=_reload,
        reload_dirs=["app"] if _reload else None,
    )
