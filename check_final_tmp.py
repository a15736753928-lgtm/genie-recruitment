import asyncio, sys
sys.stdout.reconfigure(encoding='utf-8')
from app.database import async_session_factory
from sqlalchemy import text

async def main():
    async with async_session_factory() as db:
        r = await db.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public' AND table_type='BASE TABLE'
            ORDER BY table_name
        """))
        tables = [row[0] for row in r.fetchall()]
        for t in tables:
            cnt = await db.execute(text(f'SELECT COUNT(*) FROM "{t}"'))
            n = cnt.scalar()
            marker = "  <-- 空表" if n == 0 else ""
            print(f"{t:<32}{n:>4}{marker}")

asyncio.run(main())
