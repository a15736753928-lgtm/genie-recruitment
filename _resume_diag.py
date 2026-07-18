# -*- coding: utf-8 -*-
"""Diagnose why resume parsing fails for candidate 430b1d6f-..."""
import asyncio
from sqlalchemy import select
from app.database import async_session_factory
from app.models.recruitment import Candidate
from app.infrastructure import minio_storage


async def main():
    rid = "430b1d6f-3606-43ab-bf25-416bc202b1de"
    async with async_session_factory() as db:
        result = await db.execute(select(Candidate).where(Candidate.id == rid))
        c = result.scalar_one_or_none()
        if not c:
            print("candidate not found")
            return
        print(f"name      : {c.name}")
        print(f"resume_file: {c.resume_file!r}")
        print(f"status     : {c.status}")
        print(f"phone/email/edu/exp/score: "
              f"{c.phone!r} / {c.email!r} / {c.education!r} / {c.experience!r} / {c.score!r}")

        rf = c.resume_file or ""
        # Is it a MinIO key or a legacy local path?
        import os
        is_abs = os.path.isabs(rf)
        local_exists = os.path.exists(rf) if rf else False
        print(f"is_absolute_path: {is_abs}, local_exists: {local_exists}")

        # If it looks like a MinIO key, check existence there
        if rf and not (is_abs and local_exists):
            exists_in_minio = minio_storage.object_exists(rf)
            print(f"minio object_exists({rf!r}): {exists_in_minio}")

        # Try the actual extraction path used by run_resume_parse
        from app.api.recruitment.resumes import extract_text_from_file
        try:
            text, err = extract_text_from_file(rf)
            print(f"extract_text_from_file -> text_len={len(text)}, err={err!r}")
            if text:
                print(f"  first 80 chars: {text[:80]!r}")
        except Exception as e:
            print(f"extract_text_from_file RAISED: {e!r}")


asyncio.run(main())
