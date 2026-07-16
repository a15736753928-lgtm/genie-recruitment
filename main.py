"""Genie 招聘系统 - 启动入口"""
import uvicorn
from app.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    print(f"启动 Genie 招聘系统 API...")
    print(f"地址: http://127.0.0.1:{settings.app_port}")
    print(f"文档: http://127.0.0.1:{settings.app_port}/docs")
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
