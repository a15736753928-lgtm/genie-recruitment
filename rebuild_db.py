"""Rebuild DB: DROP SCHEMA → create_all → seed_all. One-shot script."""
import asyncio, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

async def main():
    from app.database import engine, init_db
    import app.models
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text('DROP SCHEMA public CASCADE'))
        await conn.execute(text('CREATE SCHEMA public'))
        print('schema reset')

    await init_db()
    print('create_all done')

    from app.services.recruitment.seed_data import seed_all
    await seed_all()
    print('seed done')

    from app.database import async_session_factory
    from sqlalchemy import select, func
    from app.models.phase2 import TrainingCourse
    async with async_session_factory() as db:
        n = (await db.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'"))).scalar()
        c = (await db.execute(select(func.count()).select_from(TrainingCourse))).scalar()
        print(f'tables={n} courses={c}')

if __name__ == '__main__':
    asyncio.run(main())
