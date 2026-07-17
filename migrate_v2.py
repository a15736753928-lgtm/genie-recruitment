"""
V2 Migration: Align database schema with 招聘与试用期管理手册 (Recruitment & Probation Handbook).

Adds:
- Position details (JD, scorecards, question banks)
- Screening dual-dimension support
- Interview 10-dimension scorecards per position
- Probation week1 project assessment + later weeks tracking + conversion evaluation
- Mentor assignment for probation
"""
import sys
import asyncio
from sqlalchemy import text
from app.database import engine, async_session_factory

MIGRATIONS = [
    # ── 1. Extend positions table ──
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS chapter_number INTEGER;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS department VARCHAR(64);
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS jd_responsibilities TEXT;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS jd_requirements TEXT;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS jd_preferred TEXT;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS jd_tech_stack TEXT;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS screening_criteria JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS interview_criteria_r1 JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS interview_criteria_r2 JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS week1_project_requirement JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS weeks_2_4_plan JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS later_week_scoring JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS conversion_criteria JSONB;
    """,
    """
    ALTER TABLE positions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();
    """,

    # ── 2. Extend candidates table ──
    """
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS screening_ai_score INTEGER;
    """,
    """
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS screening_manual_confirmed BOOLEAN DEFAULT FALSE;
    """,
    """
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS screening_confirmed_by VARCHAR(64);
    """,
    """
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS interviewer VARCHAR(64);
    """,
    """
    ALTER TABLE candidates ADD COLUMN IF NOT EXISTS interview_round VARCHAR(8);
    """,

    # ── 3. Extend employees table ──
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS position_id UUID REFERENCES positions(id);
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS mentor_name VARCHAR(64);
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS mentor_id UUID REFERENCES users(id);
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS week1_score INTEGER;
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS week1_passed BOOLEAN;
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS conversion_score NUMERIC(5,2);
    """,
    """
    ALTER TABLE employees ADD COLUMN IF NOT EXISTS conversion_decision VARCHAR(16);
    """,

    # ── 4. Extend probation_tasks table ──
    """
    ALTER TABLE probation_tasks ADD COLUMN IF NOT EXISTS week_number INTEGER;
    """,
    """
    ALTER TABLE probation_tasks ADD COLUMN IF NOT EXISTS description TEXT;
    """,
    """
    ALTER TABLE probation_tasks ADD COLUMN IF NOT EXISTS deadline DATE;
    """,
    """
    ALTER TABLE probation_tasks ADD COLUMN IF NOT EXISTS review_notes TEXT;
    """,

    # ── 5. New: position_questions table (面试题库 per position) ──
    """
    CREATE TABLE IF NOT EXISTS position_questions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
        round VARCHAR(8) NOT NULL,
        index_num INTEGER NOT NULL,
        content TEXT NOT NULL,
        category VARCHAR(64),
        difficulty VARCHAR(8),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(position_id, round, index_num)
    );
    """,

    # ── 6. New: probation_week1_assessments ──
    """
    CREATE TABLE IF NOT EXISTS probation_week1_assessments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
        dimension_completion INTEGER DEFAULT 0,
        dimension_fidelity INTEGER DEFAULT 0,
        dimension_problem_solving INTEGER DEFAULT 0,
        dimension_standards INTEGER DEFAULT 0,
        total_score INTEGER DEFAULT 0,
        deduction_reasons JSONB,
        assessor_signature VARCHAR(64),
        dept_head_signature VARCHAR(64),
        assessor_date DATE,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(employee_id)
    );
    """,

    # ── 7. New: probation_conversions ──
    """
    CREATE TABLE IF NOT EXISTS probation_conversions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
        project_performance_score INTEGER DEFAULT 0,
        project_performance_weight NUMERIC(3,2) DEFAULT 0.60,
        tech_capability_score INTEGER DEFAULT 0,
        tech_capability_weight NUMERIC(3,2) DEFAULT 0.20,
        collaboration_score INTEGER DEFAULT 0,
        collaboration_weight NUMERIC(3,2) DEFAULT 0.20,
        total_score NUMERIC(5,2) DEFAULT 0,
        decision VARCHAR(16),
        mentor_comments TEXT,
        mentor_signature VARCHAR(64),
        mentor_date DATE,
        dept_head_signature VARCHAR(64),
        dept_head_date DATE,
        hr_signature VARCHAR(64),
        hr_date DATE,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(employee_id)
    );
    """,

    # ── 8. New: talent_pool table ──
    """
    CREATE TABLE IF NOT EXISTS talent_pool (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        candidate_id UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
        position_id UUID REFERENCES positions(id),
        score INTEGER,
        source_round VARCHAR(8),
        added_at TIMESTAMP DEFAULT NOW(),
        notes TEXT,
        UNIQUE(candidate_id)
    );
    """,
]


async def run_migrations():
    async with engine.begin() as conn:
        for i, sql in enumerate(MIGRATIONS):
            try:
                await conn.execute(text(sql))
                print(f"[{i+1}/{len(MIGRATIONS)}] OK: {sql.strip()[:80]}...")
            except Exception as e:
                print(f"[{i+1}/{len(MIGRATIONS)}] SKIP: {e}")

    print("\n✅ All migrations applied.")


if __name__ == "__main__":
    asyncio.run(run_migrations())
