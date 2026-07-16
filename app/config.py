from pydantic_settings import BaseSettings
from functools import lru_cache
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    # Database — set via DATABASE_URL / DATABASE_URL_SYNC in .env
    database_url: str = ""
    database_url_sync: str = ""

    # DeepSeek — set via DEEPSEEK_API_KEY in .env
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Vision (resume portrait gender inference)
    vision_enabled: bool = True
    vision_model: str = "deepseek-v4-flash"

    # JWT — set via JWT_SECRET_KEY in .env
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    # Upload
    upload_dir: str = os.path.join(_PROJECT_ROOT, "uploads")
    max_upload_size: int = 52428800  # 50MB

    # Milvus
    milvus_db_path: str = "milvus_lite.db"
    embedding_dim: int = 1536

    # Root User — set via ROOT_USERNAME / ROOT_PASSWORD in .env
    root_username: str = ""
    root_password: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
