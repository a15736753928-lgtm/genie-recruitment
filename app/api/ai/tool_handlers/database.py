"""Database direct-access tool handlers."""

from __future__ import annotations

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession


async def _db_list_tables(params: dict, db: AsyncSession) -> str:
    from sqlalchemy import text
    result = await db.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    ))
    tables = [row[0] for row in result.fetchall()]
    if not tables:
        return "数据库中没有找到任何表"
    lines = [f"数据库共 {len(tables)} 张表："]
    for t in tables:
        lines.append(f"  - {t}")
    return "\n".join(lines)


async def _db_describe_table(params: dict, db: AsyncSession) -> str:
    from sqlalchemy import text
    table = params["table"]
    result = await db.execute(text(
        "SELECT column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :tbl "
        "ORDER BY ordinal_position"
    ), {"tbl": table})
    rows = result.fetchall()
    if not rows:
        return f"表「{table}」不存在或没有列"
    lines = [f"表「{table}」结构（共 {len(rows)} 列）："]
    for col in rows:
        nullable = "可空" if col[2] == "YES" else "非空"
        default = f" 默认={col[3]}" if col[3] else ""
        lines.append(f"  {col[0]:30s} {col[1]:20s} {nullable}{default}")
    return "\n".join(lines)


async def _db_query(params: dict, db: AsyncSession) -> str:
    sql = params["sql"].strip()
    sql_upper = sql.upper()
    if not sql_upper.startswith("SELECT"):
        return "❌ db_query 只允许执行 SELECT 查询。如需修改数据请使用 db_update。"
    if any(kw in sql_upper for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE")):
        return "❌ db_query 只允许只读的 SELECT 查询。如需修改数据请使用 db_update。"
    limit = params.get("limit", 20)
    if "LIMIT" not in sql_upper:
        sql = f"{sql.rstrip(';')} LIMIT {limit}"
    result = await db.execute(sa_text(sql))
    rows = result.fetchall()
    cols = list(result.keys())
    if not rows:
        return "查询结果为空"
    lines = [f"查询返回 {len(rows)} 行（列: {', '.join(cols)}）："]
    for i, row in enumerate(rows):
        cells = ", ".join(f"{cols[j]}={row[j]!r}" for j in range(len(cols)))
        lines.append(f"  [{i+1}] {cells}")
    return "\n".join(lines)


async def _db_update(params: dict, db: AsyncSession) -> str:
    sql = params["sql"].strip()
    sql_upper = sql.upper()
    if sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE"):
        if "WHERE" not in sql_upper:
            return (
                "❌ 安全限制：UPDATE 和 DELETE 必须包含 WHERE 条件，"
                "禁止全表修改。请加上 WHERE 后重试。"
            )
    if sql_upper.startswith("SELECT"):
        return "❌ db_update 用于写操作。查询请使用 db_query。"
    result = await db.execute(sa_text(sql))
    await db.flush()
    rowcount = result.rowcount if hasattr(result, 'rowcount') else "?"
    return f"✅ SQL 执行成功，影响 {rowcount} 行。已提交到数据库。你可以用 db_query 验证结果。"


def register_handlers(registry) -> None:
    registry.register("db_list_tables", _db_list_tables)
    registry.register("db_describe_table", _db_describe_table)
    registry.register("db_query", _db_query)
    registry.register("db_update", _db_update)
