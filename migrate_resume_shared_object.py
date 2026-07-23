"""数据迁移 — 收敛简历「双份存储」为单一共享对象。

背景：
    历史上一份简历会在 MinIO 存两份独立对象：
      - Candidate.resume_file      → resumes/<uuid>.<ext>（简历筛选用）
      - KnowledgeDocument.object_key → rag/<uuid>.<ext>（知识库管理用）
    两份互相独立，任一份失踪都会造成「一边能看、一边 404」。
    新代码已改为共享同一个对象；本脚本修复存量数据。

处理逻辑（幂等，可重复运行）：
    遍历所有有 resume_file 的候选人，找到其关联的「简历」知识库文档
    （source_type='resume' AND source_id=candidate_id）：

      A) resume_file 对象已丢失，但 KB 文档对象还在
         → 把 resume_file 回指到 KB 文档的 object_key（救活简历筛选预览）
      B) 两份都在且不是同一个对象
         → 让 KB 文档 object_key 收敛指向 resume_file，并删除多余的 rag/ 副本
         （统一为单一对象；删除仅针对 rag/ 前缀的冗余副本，绝不删 resumes/）
      C) 已经是同一个对象 / 无法配对 / 两份都丢失
         → 跳过并记录

使用方式:
    python migrate_resume_shared_object.py            # 实际执行
    python migrate_resume_shared_object.py --dry-run  # 仅打印将要做的改动
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DRY_RUN = "--dry-run" in sys.argv


def _obj_available(stored: str) -> bool:
    """判断 stored 指向的对象是否可用（本地绝对路径 or MinIO key）。"""
    if not stored:
        return False
    if os.path.isabs(stored):
        return os.path.exists(stored)
    from app.infrastructure import minio_storage
    try:
        return minio_storage.object_exists(stored)
    except Exception:
        return False


async def main():
    from sqlalchemy import select
    from app.database import async_session_factory
    from app.models.recruitment import Candidate
    from app.models.knowledge import KnowledgeDocument
    from app.infrastructure import minio_storage

    stats = {"scanned": 0, "revived": 0, "converged": 0, "already": 0,
             "unpairable": 0, "both_lost": 0, "deleted_dupes": 0}

    async with async_session_factory() as db:
        cand_rows = (await db.execute(
            select(Candidate.id, Candidate.resume_file)
            .where(Candidate.resume_file.is_not(None))
            .where(Candidate.resume_file != "")
        )).all()

        for cid, resume_file in cand_rows:
            stats["scanned"] += 1
            cid = str(cid)

            doc = (await db.execute(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.source_type == "resume")
                .where(KnowledgeDocument.source_id == cid)
                .limit(1)
            )).scalar_one_or_none()

            resume_ok = _obj_available(resume_file)
            doc_key = doc.object_key if doc else ""
            doc_ok = _obj_available(doc_key) if doc_key else False

            # C1) 已经共享同一个对象
            if doc and doc_key == resume_file:
                stats["already"] += 1
                continue

            # 无配对文档：只能看 resume_file 本身，无法救
            if not doc:
                if not resume_ok:
                    stats["both_lost"] += 1
                    print(f"[孤立-简历丢失] candidate={cid} resume_file={resume_file}")
                else:
                    stats["unpairable"] += 1
                continue

            # A) 简历丢失、KB 还在 → 回指救活
            if not resume_ok and doc_ok:
                print(f"[救活] candidate={cid}: resume_file {resume_file!r} → {doc_key!r}")
                if not DRY_RUN:
                    cand = (await db.execute(
                        select(Candidate).where(Candidate.id == cid)
                    )).scalar_one_or_none()
                    if cand:
                        cand.resume_file = doc_key
                stats["revived"] += 1
                continue

            # B) 两份都在且不同 → KB 收敛到 resume_file，删多余 rag/ 副本
            if resume_ok and doc_ok:
                print(f"[收敛] candidate={cid}: KB object_key {doc_key!r} → {resume_file!r}")
                if not DRY_RUN:
                    old_key = doc.object_key
                    doc.object_key = resume_file
                    # 仅删除冗余的 rag/ 前缀副本，绝不删 resumes/ 原件
                    if old_key and old_key != resume_file and old_key.startswith("rag/"):
                        try:
                            await asyncio.to_thread(minio_storage.delete_object, old_key)
                            stats["deleted_dupes"] += 1
                        except Exception as e:
                            print(f"    删除冗余副本失败(忽略) {old_key}: {e}")
                stats["converged"] += 1
                continue

            # 简历还在、KB 丢失 → 让 KB 指回 resume_file（顺带修复知识库预览）
            if resume_ok and not doc_ok:
                print(f"[修复KB] candidate={cid}: KB object_key {doc_key!r} → {resume_file!r}")
                if not DRY_RUN:
                    doc.object_key = resume_file
                stats["converged"] += 1
                continue

            # 两份都丢失
            stats["both_lost"] += 1
            print(f"[双丢失] candidate={cid} resume_file={resume_file} doc_key={doc_key}")

        if not DRY_RUN:
            await db.commit()

    print("\n===== 迁移结果" + ("（DRY-RUN，未写库）" if DRY_RUN else "") + " =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
