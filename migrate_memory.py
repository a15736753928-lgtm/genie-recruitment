"""
One-shot migration: create the agent_memories table and sync file-based
memories into PostgreSQL.

Run once::

    python migrate_memory.py
"""

import asyncio
import os
import sys

# Ensure the server root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app.database import engine, async_session_factory
from app.models.agent_memory import AgentMemory
from app.agent_os.memory.manager import MemoryManager


async def migrate():
    # 1. Create table
    async with engine.begin() as conn:
        await conn.run_sync(AgentMemory.metadata.create_all)
    print("✓ agent_memories table created (or already exists)")

    # 2. Sync files → DB
    mm = MemoryManager(memory_dir="memory/")
    memories = mm.load_all()
    if not memories:
        print("  No file-based memories to sync.")
        return

    def _serialize_metadata(meta: dict) -> dict:
        """Convert any non-JSON-serializable values (datetime) to strings."""
        from datetime import datetime
        clean = {}
        for k, v in meta.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            elif isinstance(v, dict):
                clean[k] = _serialize_metadata(v)
            else:
                clean[k] = v
        return clean

    async with async_session_factory() as db:
        for mem in memories:
            # Upsert by name
            from sqlalchemy import select
            result = await db.execute(
                select(AgentMemory).where(AgentMemory.name == mem.name)
            )
            existing = result.scalar_one_or_none()
            clean_meta = _serialize_metadata(mem.metadata)
            if existing:
                existing.description = mem.description
                existing.content = mem.content
                existing.metadata_ = clean_meta
            else:
                db.add(AgentMemory(
                    name=mem.name,
                    description=mem.description,
                    content=mem.content,
                    metadata_=clean_meta,
                ))
        await db.commit()
    print(f"✓ Synced {len(memories)} memories to PostgreSQL")


if __name__ == "__main__":
    asyncio.run(migrate())
