"""
一键数据库初始化脚本 — 适用于新电脑/新环境首次部署。

使用方式:
    python setup_db.py

前置条件:
    1. PostgreSQL 已安装并运行（或使用 docker-compose up -d）
    2. .env 文件中已配置 DATABASE_URL（若不存在则自动从 .env.example 复制）

功能流程:
    1. 检查 .env 是否存在，首次运行时从 .env.example 复制
    2. 连接 PostgreSQL 默认库，创建 genie_recruitment 数据库（不存在时）
    3. 创建所有表（SQLAlchemy ORM）
    4. 应用所有迁移补丁（v1-v13，幂等）
    5. 写入种子数据（岗位、知识分类、系统设置）
    6. 验证各表创建结果并输出摘要
"""

import asyncio
import os
import sys

# ── 项目根路径 ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
ENV_EXAMPLE = os.path.join(PROJECT_ROOT, ".env.example")


# ══════════════════════════════════════════════════════════════
# Step 0 — 确保 .env 存在
# ══════════════════════════════════════════════════════════════

def ensure_env_file():
    """首次运行时从 .env.example 复制出 .env，提示用户填入实际值。"""
    if os.path.exists(ENV_FILE):
        return

    if not os.path.exists(ENV_EXAMPLE):
        print("[错误] .env.example 模板文件不存在，无法自动生成 .env")
        sys.exit(1)

    # 读取模板并替换占位符为开发默认值
    with open(ENV_EXAMPLE, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("YOUR_PASSWORD", "1234")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("✓ 已从 .env.example 生成 .env（数据库密码默认为 1234）")
    print("  如需修改数据库密码或 API Key，请编辑 .env 后重新运行本脚本")


# ══════════════════════════════════════════════════════════════
# Step 1 — 创建 PostgreSQL 数据库
# ══════════════════════════════════════════════════════════════

def parse_db_params(url: str) -> dict:
    """从 DATABASE_URL 里拆出 host / port / user / password / dbname。
    格式: postgresql+asyncpg://user:pass@host:port/dbname
    """
    # 去掉驱动前缀
    clean = url
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://", "postgresql://"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    else:
        raise ValueError(f"无法解析 DATABASE_URL: {url}")

    # user:pass@host:port/dbname  或  user@host:port/dbname
    user_pass, rest = clean.split("@", 1)
    if ":" in user_pass:
        user, password = user_pass.split(":", 1)
    else:
        user, password = user_pass, ""

    host_port, dbname = rest.rsplit("/", 1)
    if ":" in host_port:
        host, port = host_port.split(":", 1)
    else:
        host, port = host_port, "5432"

    return {
        "user": user,
        "password": password,
        "host": host,
        "port": int(port),
        "dbname": dbname,
    }


def create_database_if_missing():
    """连到 postgres 默认库，创建目标数据库（不存在时）。"""
    from app.config import get_settings

    settings = get_settings()
    db_url = settings.database_url or settings.database_url_sync

    if not db_url:
        print("[错误] .env 中未配置 DATABASE_URL")
        sys.exit(1)

    params = parse_db_params(db_url)
    target_db = params["dbname"]

    # 连接默认 postgres 库
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            dbname="postgres",
        )
        conn.autocommit = True
        cur = conn.cursor()

        # 检查目标库是否存在
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (target_db,)
        )
        if cur.fetchone():
            print(f"✓ 数据库 '{target_db}' 已存在，跳过创建")
        else:
            cur.execute(f'CREATE DATABASE "{target_db}"')
            print(f"✓ 数据库 '{target_db}' 创建成功")

        cur.close()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"[错误] 无法连接 PostgreSQL: {e}")
        print("  请确保 PostgreSQL 已启动，且 .env 中连接信息正确")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Step 2 — 确保 MinIO bucket 存在
# ══════════════════════════════════════════════════════════════

def ensure_minio_bucket():
    """创建 MinIO bucket（不存在时），非致命——MinIO 不可用时 skip。"""
    from app.config import get_settings

    settings = get_settings()
    if not settings.minio_enabled:
        print("○ MinIO 已禁用，跳过 bucket 检查")
        return

    try:
        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        bucket = settings.minio_bucket
        if client.bucket_exists(bucket):
            print(f"✓ MinIO bucket '{bucket}' 已存在")
        else:
            client.make_bucket(bucket)
            print(f"✓ MinIO bucket '{bucket}' 创建成功")
    except Exception as e:
        print(f"○ MinIO 不可用 ({e})，跳过 bucket 创建（上传时会回退到本地目录）")


# ══════════════════════════════════════════════════════════════
# Step 3 — 建表 & 迁移
# ══════════════════════════════════════════════════════════════

async def create_tables():
    """创建所有 ORM 表并应用迁移补丁。"""
    from app.database import engine, Base

    # 必须先 import models，让所有表注册到 Base.metadata
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        print("  创建/更新数据库表...")
        await conn.run_sync(Base.metadata.create_all)
        print("  应用迁移补丁 (v1-v13)...")
        from app.database import _run_migrations
        await conn.run_sync(_run_migrations)

    print("✓ 表结构与迁移完成")


# ══════════════════════════════════════════════════════════════
# Step 4 — 写入种子数据
# ══════════════════════════════════════════════════════════════

async def seed_data():
    """写入默认岗位、知识分类、系统设置（幂等：已存在则跳过）。"""
    from app.services.recruitment.seed_data import seed_all

    await seed_all()
    print("✓ 种子数据写入完成")


# ══════════════════════════════════════════════════════════════
# Step 5 — 验证
# ══════════════════════════════════════════════════════════════

async def verify():
    """统计每张表的行数，输出初始化摘要。"""
    from sqlalchemy import text
    from app.database import async_session_factory

    async with async_session_factory() as db:
        # 列出所有用户表
        result = await db.execute(text(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
        ))
        tables = sorted(row[0] for row in result.fetchall())

        print("\n" + "=" * 56)
        print(f"{'表名':<36} {'行数':>8}")
        print("-" * 56)

        total_rows = 0
        for t in tables:
            row_result = await db.execute(text(f'SELECT COUNT(*) FROM "{t}"'))
            count = row_result.scalar() or 0
            total_rows += count
            marker = " ✓" if count > 0 else ""
            print(f"  {t:<34} {count:>6}{marker}")

        print("-" * 56)
        print(f"  {'共 ' + str(len(tables)) + ' 张表，总计':>42} {total_rows:>6} 条记录")
        print("=" * 56)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def check_dependencies():
    """验证 Python 依赖是否已安装，缺少时给出明确的安装命令。"""
    missing = []
    for mod, pkg in [
        ("sqlalchemy", "sqlalchemy"),
        ("psycopg2", "psycopg2-binary"),
        ("asyncpg", "asyncpg"),
        ("pydantic_settings", "pydantic-settings"),
        ("minio", "minio"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("[错误] 缺少 Python 依赖，请先安装:")
        print(f"  pip install {' '.join(missing)}")
        print("  或一键安装全部依赖:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


async def main():
    print("=" * 56)
    print("  Genie 招聘系统 — 数据库初始化")
    print("=" * 56)

    # Step 0
    check_dependencies()
    ensure_env_file()

    # Step 1
    print("\n[1/4] 检查 PostgreSQL 数据库...")
    create_database_if_missing()

    # Step 2
    print("\n[2/4] 检查 MinIO bucket...")
    ensure_minio_bucket()

    # Step 3
    print("\n[3/4] 创建表 & 迁移...")
    await create_tables()

    # Step 4
    print("\n[4/4] 写入种子数据...")
    await seed_data()

    # Verify
    await verify()

    print("\n✓ 数据库初始化完成！")
    print("  启动后端: python main.py")
    print("  启动前端: cd ../genie-recruitment-front && npm run dev")
    print("  后端地址: http://127.0.0.1:8000")
    print("  前端地址: http://127.0.0.1:5173")
    print("  API 文档: http://127.0.0.1:8000/docs")

    # 关闭连接池
    from app.database import engine
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
