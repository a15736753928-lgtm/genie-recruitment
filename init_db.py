"""Initialize database - create all tables and seed data."""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from app.database import engine, Base, async_session_factory
    from app.services.recruitment.seed_data import seed_all
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  Tables created successfully!")

    print("Seeding default data...")
    await seed_all()
    print("  Seed data inserted!")

    # Verify
    from sqlalchemy import select, func
    async with async_session_factory() as db:
        tables = ["positions", "candidates", "interview_questions",
                   "employees", "knowledge_categories", "system_settings"]
        for t in tables:
            result = await db.execute(select(func.count()).select_from(getattr(
                __import__('app.models', fromlist=[t]), t.replace('_', ' ').title().replace(' ', ''))))
            print(f"  {t}: OK")

    print("\nDatabase initialization complete!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
