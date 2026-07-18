"""数据导出 — 按系统设置 exportFormat 生成 xlsx/csv。"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, List, Sequence

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.recruitment import Candidate
from app.models.interview import InterviewEvaluation
from app.models.performance import PerformanceRecord
from app.models.probation import Employee
from app.services.system.system_settings import get_system_setting

router = APIRouter(tags=["数据导出"])

SUPPORTED_MODULES = {"resumes", "interviews", "probation", "performance"}


def _to_csv(headers: Sequence[str], rows: List[List[Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def _to_xlsx(headers: Sequence[str], rows: List[List[Any]]) -> bytes:
    try:
        from openpyxl import Workbook
    except ImportError:
        # 无 openpyxl 时回退 CSV
        return _to_csv(headers, rows)

    wb = Workbook()
    ws = wb.active
    ws.title = "export"
    ws.append(list(headers))
    for row in rows:
        ws.append(list(row))
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


@router.get("/export/{module}")
async def export_module(module: str, db: AsyncSession = Depends(get_db)):
    if module not in SUPPORTED_MODULES:
        return {
            "code": 400,
            "message": f"不支持的导出模块，可选：{', '.join(sorted(SUPPORTED_MODULES))}",
            "data": None,
        }

    export_format = str(await get_system_setting(db, "exportFormat", "xlsx") or "xlsx").lower()
    if export_format not in ("xlsx", "csv"):
        export_format = "xlsx"

    if module == "resumes":
        result = await db.execute(
            select(Candidate).options(selectinload(Candidate.position)).order_by(Candidate.upload_time.desc())
        )
        items = result.scalars().all()
        headers = ["id", "name", "position", "status", "score", "phone", "email", "uploadTime"]
        rows = [
            [
                str(c.id),
                c.name or "",
                c.position.name if c.position else "",
                c.status or "",
                c.score or 0,
                c.phone or "",
                c.email or "",
                c.upload_time.isoformat() if c.upload_time else "",
            ]
            for c in items
        ]
    elif module == "interviews":
        result = await db.execute(select(InterviewEvaluation))
        items = result.scalars().all()
        headers = ["id", "candidateId", "round", "questionId", "aiScore", "hrScore", "status"]
        rows = [
            [
                str(e.id),
                str(e.candidate_id),
                e.round or "",
                str(e.question_id) if e.question_id else "",
                e.ai_score if e.ai_score is not None else "",
                e.hr_score if e.hr_score is not None else "",
                e.status or "",
            ]
            for e in items
        ]
    elif module == "probation":
        result = await db.execute(select(Employee).order_by(Employee.created_at.desc()))
        items = result.scalars().all()
        headers = ["id", "name", "department", "joinDate", "probationEnd", "status", "week1Score"]
        rows = [
            [
                str(e.id),
                e.name or "",
                e.department or "",
                e.join_date.isoformat() if e.join_date else "",
                e.probation_end.isoformat() if e.probation_end else "",
                e.status or "",
                e.week1_score if e.week1_score is not None else "",
            ]
            for e in items
        ]
    else:  # performance
        result = await db.execute(
            select(PerformanceRecord).options(selectinload(PerformanceRecord.employee))
        )
        items = result.scalars().all()
        headers = ["id", "name", "department", "quarter", "totalScore", "grade", "bonus"]
        rows = [
            [
                str(r.id),
                r.employee.name if r.employee else "",
                r.employee.department if r.employee else "",
                r.quarter or "",
                r.total_score or 0,
                r.grade or "",
                float(r.bonus) if r.bonus is not None else "",
            ]
            for r in items
        ]

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if export_format == "csv":
        content = _to_csv(headers, rows)
        media = "text/csv; charset=utf-8"
        filename = f"{module}_{ts}.csv"
    else:
        content = _to_xlsx(headers, rows)
        # 若 openpyxl 缺失会回退 CSV
        if content[:2] != b"PK":
            media = "text/csv; charset=utf-8"
            filename = f"{module}_{ts}.csv"
        else:
            media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{module}_{ts}.xlsx"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
