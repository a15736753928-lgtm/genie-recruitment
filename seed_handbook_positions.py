"""将招聘手册中6个技术岗位强制写入数据库（幂等：已存在则更新）。"""
import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from app.database import async_session_factory
    from app.models.recruitment import Position, PositionQuestion
    from sqlalchemy import select
    from app.services.recruitment.seed_data_v2 import (
        POSITIONS_DATA,
        QUESTION_BANKS,
        SCREENING_CRITERIA_COMMON,
        INTERVIEW_CRITERIA_R1_COMMON,
        LATER_WEEK_SCORING_COMMON,
        CONVERSION_CRITERIA_COMMON,
    )

    async with async_session_factory() as db:
        for pos_data in POSITIONS_DATA:
            result = await db.execute(
                select(Position).where(Position.name == pos_data["name"])
            )
            existing = result.scalar_one_or_none()

            if existing:
                # 更新已有岗位的字段
                for key, value in pos_data.items():
                    if key == "name":
                        continue
                    setattr(existing, key, value)
                existing.screening_criteria = SCREENING_CRITERIA_COMMON
                existing.interview_criteria_r1 = INTERVIEW_CRITERIA_R1_COMMON
                existing.later_week_scoring = LATER_WEEK_SCORING_COMMON
                existing.conversion_criteria = CONVERSION_CRITERIA_COMMON
                position = existing
                print(f"[更新] {pos_data['name']}")
            else:
                # 新建岗位
                position = Position(
                    name=pos_data["name"],
                    **{k: v for k, v in pos_data.items() if k != "name"},
                    screening_criteria=SCREENING_CRITERIA_COMMON,
                    interview_criteria_r1=INTERVIEW_CRITERIA_R1_COMMON,
                    later_week_scoring=LATER_WEEK_SCORING_COMMON,
                    conversion_criteria=CONVERSION_CRITERIA_COMMON,
                )
                db.add(position)
                await db.flush()
                await db.refresh(position)
                print(f"[创建] {pos_data['name']}")

            # 一面题库（幂等：已有则不重复写入）
            qb = QUESTION_BANKS.get(pos_data["name"], {})
            for round_key in ("first",):
                if round_key not in qb:
                    continue
                existing_qs = await db.execute(
                    select(PositionQuestion).where(
                        PositionQuestion.position_id == position.id,
                        PositionQuestion.round == round_key,
                    )
                )
                if existing_qs.scalars().first():
                    print(f"  题库已存在（跳过）")
                    continue

                for q_data in qb[round_key]:
                    db.add(PositionQuestion(
                        position_id=position.id,
                        round=round_key,
                        index_num=q_data["index"],
                        content=q_data["content"],
                        category=q_data["category"],
                        difficulty=q_data["difficulty"],
                    ))
                print(f"  写入 {len(qb[round_key])} 道面试题")

        await db.commit()
    print("\n✅ 完成：6 个手册岗位已写入数据库")

if __name__ == "__main__":
    asyncio.run(main())
