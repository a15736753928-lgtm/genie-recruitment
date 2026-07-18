"""Initialize database - create all tables and seed data."""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from app.database import engine, Base, async_session_factory
    from app.services.recruitment.seed_data import seed_all
    # Import all models to register them with Base
    from app.models.recruitment import Position, Candidate, CandidateSkill, CandidateEducation
    from app.models.recruitment import CandidateWorkExperience, CandidateProjectExperience, CandidateAIAnalysis
    from app.models.interview import InterviewQuestion, InterviewEvaluation, InterviewTranscript
    from app.models.probation import Employee, ProbationTask
    from app.models.performance import PerformanceRecord, PerformanceQuarter
    from app.models.knowledge import KnowledgeCategory, KnowledgeItem
    from app.models.knowledge import KnowledgeBase, KnowledgeDocument, KnowledgeChunk, IngestionTask
    from app.models.agent_session import AgentSession, AgentMessage, AgentMaterial, AgentTask
    from app.models.settings import SystemSetting

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
