"""
Startup dependency health-check system.

Runs before the server accepts requests. Every check returns a CheckResult.
Critical checks (DB, upload dir) fail-fast and prevent startup.
Optional checks (RAG stack) and config checks run concurrently and warn on failure.

Usage:
    from app.core.startup_checks import run_startup_checks, print_check_summary
    results = await run_startup_checks(settings)
    print_check_summary(results)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable
from urllib.parse import urlparse

from app.config import Settings

logger = logging.getLogger("genie.startup")


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass
class CheckResult:
    name: str             # human-readable, e.g. "PostgreSQL (async)"
    category: str         # "critical", "optional", "config"
    status: CheckStatus
    detail: str = ""      # success message, or error/warning detail
    elapsed_ms: float = 0.0


CheckFn = Callable[[Settings], Awaitable[CheckResult]]


# ═══════════════════════════════════════════════════════════════════
# Critical checks (fail-fast — app does NOT start)
# ═══════════════════════════════════════════════════════════════════

async def check_database_async(settings: Settings) -> CheckResult:
    """Verify async PostgreSQL connection + table creation + migrations."""
    t0 = time.perf_counter()
    try:
        from app.database import init_db
        await init_db()
        return CheckResult(
            "PostgreSQL (async)", "critical", CheckStatus.PASS,
            "Connected, tables created, migrations applied",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "PostgreSQL (async)", "critical", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


async def check_database_sync(settings: Settings) -> CheckResult:
    """Verify sync PostgreSQL engine (used by background threads)."""
    t0 = time.perf_counter()
    try:
        from app.database import SyncSessionFactory
        from sqlalchemy import text
        db = SyncSessionFactory()
        db.execute(text("SELECT 1"))
        db.close()
        return CheckResult(
            "PostgreSQL (sync)", "critical", CheckStatus.PASS,
            "Sync engine verified",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "PostgreSQL (sync)", "critical", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


async def check_upload_dir(settings: Settings) -> CheckResult:
    """Ensure the upload directory exists and is writable."""
    t0 = time.perf_counter()
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        if not os.access(settings.upload_dir, os.W_OK):
            raise PermissionError(f"Directory not writable: {settings.upload_dir}")
        return CheckResult(
            "Upload directory", "critical", CheckStatus.PASS,
            f"Writable: {settings.upload_dir}",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "Upload directory", "critical", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════
# Optional / RAG stack checks (non-blocking — WARN on failure)
# ═══════════════════════════════════════════════════════════════════

async def check_milvus_lite(settings: Settings) -> CheckResult:
    """Verify Milvus Lite vector store is accessible."""
    t0 = time.perf_counter()
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(settings.milvus_db_path)
        collections = client.list_collections()
        return CheckResult(
            "Milvus Lite", "optional", CheckStatus.PASS,
            f"Connected, {len(collections)} collection(s)",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "Milvus Lite", "optional", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


async def check_embedding_model(settings: Settings) -> CheckResult:
    """Preload the BGE-M3 embedding model (replaces background preload)."""
    t0 = time.perf_counter()
    try:
        from app.core.model_loader import preload_models
        await asyncio.to_thread(preload_models)
        return CheckResult(
            "BGE-M3 Embedding", "optional", CheckStatus.PASS,
            "Model loaded (ONNX or PyTorch)",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "BGE-M3 Embedding", "optional", CheckStatus.FAIL,
            f"Preload failed, will retry on first use: {e}",
            (time.perf_counter() - t0) * 1000,
        )


async def check_reranker_model(settings: Settings) -> CheckResult:
    """Verify the BGE reranker model can be loaded."""
    t0 = time.perf_counter()
    try:
        def _load():
            from sentence_transformers import CrossEncoder
            CrossEncoder(settings.reranker_model, trust_remote_code=True, local_files_only=True)
        await asyncio.to_thread(_load)
        return CheckResult(
            "BGE Reranker", "optional", CheckStatus.PASS,
            f"Model loaded: {settings.reranker_model}",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "BGE Reranker", "optional", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


async def check_kuzu_graph(settings: Settings) -> CheckResult:
    """Verify the Kuzu graph database library is available."""
    t0 = time.perf_counter()
    try:
        import kuzu
        with tempfile.TemporaryDirectory() as tmpdir:
            db = kuzu.Database(os.path.join(tmpdir, "health_check"))
            conn = kuzu.Connection(db)
            conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS _health_check("
                "id INT64, PRIMARY KEY(id))"
            )
            conn.execute("DROP TABLE _health_check")
        return CheckResult(
            "Kuzu Graph DB", "optional", CheckStatus.PASS,
            "Library available, create/query OK",
            (time.perf_counter() - t0) * 1000,
        )
    except ImportError:
        return CheckResult(
            "Kuzu Graph DB", "optional", CheckStatus.FAIL,
            "kuzu package not installed",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "Kuzu Graph DB", "optional", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


async def check_rapidocr(settings: Settings) -> CheckResult:
    """Verify the OCR engine can be initialized."""
    t0 = time.perf_counter()
    try:
        from rapidocr_onnxruntime import RapidOCR
        RapidOCR()
        return CheckResult(
            "RapidOCR", "optional", CheckStatus.PASS,
            "OCR engine loaded",
            (time.perf_counter() - t0) * 1000,
        )
    except ImportError:
        return CheckResult(
            "RapidOCR", "optional", CheckStatus.FAIL,
            "rapidocr-onnxruntime not installed",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "RapidOCR", "optional", CheckStatus.FAIL,
            str(e),
            (time.perf_counter() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════
# Config validation checks (non-blocking — WARN on issues)
# ═══════════════════════════════════════════════════════════════════

async def check_deepseek_api_key(settings: Settings) -> CheckResult:
    """Verify DeepSeek API key is configured."""
    key = settings.deepseek_api_key
    if not key or key == "sk-your-api-key-here":
        return CheckResult(
            "DeepSeek API Key", "config", CheckStatus.FAIL,
            "Not configured; AI agent features will not work",
        )
    return CheckResult(
        "DeepSeek API Key", "config", CheckStatus.PASS,
        f"Configured ({key[:8]}...)",
    )


async def check_deepseek_api_reachable(settings: Settings) -> CheckResult:
    """Verify DeepSeek API endpoint is reachable."""
    key = settings.deepseek_api_key
    if not key or key == "sk-your-api-key-here":
        return CheckResult(
            "DeepSeek API", "config", CheckStatus.PASS,
            "Skipped (no key configured)",
        )

    t0 = time.perf_counter()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.deepseek_base_url}/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                return CheckResult(
                    "DeepSeek API", "config", CheckStatus.PASS,
                    f"Reachable ({resp.status_code})",
                    elapsed,
                )
            else:
                return CheckResult(
                    "DeepSeek API", "config", CheckStatus.FAIL,
                    f"HTTP {resp.status_code}: {resp.text[:100]}",
                    elapsed,
                )
    except Exception as e:
        return CheckResult(
            "DeepSeek API", "config", CheckStatus.FAIL,
            f"Unreachable: {e}",
            (time.perf_counter() - t0) * 1000,
        )


async def check_cors_origins(settings: Settings) -> CheckResult:
    """Validate CORS origin URLs."""
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    bad = []
    for origin in origins:
        try:
            parsed = urlparse(origin)
            if not parsed.scheme or not parsed.netloc:
                bad.append(origin)
        except Exception:
            bad.append(origin)

    if bad:
        return CheckResult(
            "CORS Origins", "config", CheckStatus.FAIL,
            f"Invalid entries: {', '.join(bad)}",
        )
    return CheckResult(
        "CORS Origins", "config", CheckStatus.PASS,
        f"{len(origins)} origin(s) configured",
    )


async def check_database_url_format(settings: Settings) -> CheckResult:
    """Validate the DATABASE_URL format."""
    url = settings.database_url
    if not url:
        return CheckResult(
            "Database URL", "config", CheckStatus.FAIL,
            "DATABASE_URL is empty",
        )

    try:
        parsed = urlparse(url)
        if parsed.scheme != "postgresql+asyncpg":
            return CheckResult(
                "Database URL", "config", CheckStatus.FAIL,
                f"Unexpected scheme: {parsed.scheme} (expected postgresql+asyncpg)",
            )
        if not parsed.hostname:
            return CheckResult(
                "Database URL", "config", CheckStatus.FAIL,
                "No hostname in DATABASE_URL",
            )
        return CheckResult(
            "Database URL", "config", CheckStatus.PASS,
            f"Format OK ({parsed.scheme}://{parsed.hostname}:{parsed.port or '?'}/{parsed.path.lstrip('/')})",
        )
    except Exception as e:
        return CheckResult(
            "Database URL", "config", CheckStatus.FAIL,
            f"Parse error: {e}",
        )


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def _ensure_imports() -> None:
    """Pre-import packages used by concurrent check threads.

    Without this, ``asyncio.to_thread(…)`` calls in phase 2 can race on the
    Python import lock and produce ``cannot import name 'AutoTokenizer' from
    'transformers'`` (the module is only partially initialised in the losing
    thread).
    """
    try:
        __import__("transformers")
    except Exception:
        pass
    try:
        __import__("sentence_transformers")
    except Exception:
        pass
    try:
        __import__("onnxruntime")
    except Exception:
        pass


# Per-category timeout (seconds). A check that exceeds this is reported as
# timed out so a single stuck dependency can't block startup forever.
_CHECK_TIMEOUTS = {
    "critical": 15.0,   # DB — fail-fast, refuse to start
    "optional": 60.0,   # RAG models — slow to load, warn and continue
    "config": 10.0,     # config validation — warn and continue
}


async def _run_check(check_fn: CheckFn, settings: Settings, category: str) -> CheckResult:
    """Run one check with progress logging + a per-category timeout.

    Logs the check name *before* it runs, so a hang shows exactly which check
    is stuck as the last ``▶`` line. On timeout, critical checks FAIL (refuse
    startup); optional/config checks WARN (let the app start anyway).
    """
    name = check_fn.__name__
    timeout = _CHECK_TIMEOUTS.get(category)
    logger.info("▶ %s ...", name)
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(check_fn(settings), timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        if result.elapsed_ms == 0.0:
            result.elapsed_ms = elapsed
        logger.info("  %s %s — %s (%.0fms)",
                    _STATUS_ICON[result.status], result.name,
                    result.detail[:60], result.elapsed_ms)
        return result
    except asyncio.TimeoutError:
        elapsed = (time.perf_counter() - t0) * 1000
        status = CheckStatus.FAIL if category == "critical" else CheckStatus.WARN
        detail = f"Timed out after {timeout:.0f}s"
        logger.warning("  %s %s — %s (%.0fms)",
                       _STATUS_ICON[status], name, detail, elapsed)
        return CheckResult(name=name, category=category, status=status,
                           detail=detail, elapsed_ms=elapsed)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("  ✗ %s — crashed: %s (%.0fms)", name, e, elapsed)
        return CheckResult(name=name, category=category, status=CheckStatus.FAIL,
                           detail=f"Check crashed: {e}", elapsed_ms=elapsed)


async def run_startup_checks(settings: Settings) -> list[CheckResult]:
    """Run all startup checks. Every check runs — none skipped on early failure.

    Phase 1: Infrastructure checks (sequential — DB before everything else)
    Phase 2: RAG stack checks (concurrent)
    Phase 3: Config validation (concurrent)

    All phases always run to completion so the user sees the full picture
    of what's broken in one restart cycle.
    """
    checks_phase1: list[CheckFn] = [
        check_database_url_format,
        check_database_async,
        check_database_sync,
        check_upload_dir,
    ]
    checks_phase2: list[CheckFn] = [
        check_milvus_lite,
        check_embedding_model,
        check_reranker_model,
        check_kuzu_graph,
        check_rapidocr,
    ]
    checks_phase3: list[CheckFn] = [
        check_deepseek_api_key,
        check_deepseek_api_reachable,
        check_cors_origins,
    ]

    # Pre-import heavy modules to avoid import race conditions when
    # concurrent check threads import from the same package (transformers).
    _ensure_imports()

    total = len(checks_phase1) + len(checks_phase2) + len(checks_phase3)
    logger.info("开始启动健康检查（共 %d 项）...", total)

    results: list[CheckResult] = []

    # Phase 1: Infrastructure (sequential — DB before everything else)
    logger.info("── Phase 1: 基础设施（critical，超时即拒绝启动）──")
    for check_fn in checks_phase1:
        results.append(await _run_check(check_fn, settings, "critical"))

    # Phase 2: RAG stack (concurrent)
    logger.info("── Phase 2: RAG 栈（optional，超时降级为 WARN）──")
    phase2_results = await asyncio.gather(
        *(_run_check(fn, settings, "optional") for fn in checks_phase2),
    )
    results.extend(phase2_results)

    # Phase 3: Config validation (concurrent)
    logger.info("── Phase 3: 配置校验（config，超时降级为 WARN）──")
    config_results = await asyncio.gather(
        *(_run_check(fn, settings, "config") for fn in checks_phase3),
    )
    results.extend(config_results)

    return results


# ═══════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════

_STATUS_ICON = {
    CheckStatus.PASS: "✓",
    CheckStatus.FAIL: "✗",
    CheckStatus.WARN: "⚠",
}


def print_check_summary(results: list[CheckResult]) -> None:
    """Print a formatted summary table of all check results."""
    logger.info("=" * 70)
    logger.info("  启动健康检查报告")
    logger.info("=" * 70)

    for r in results:
        icon = _STATUS_ICON[r.status]
        elapsed = f" ({r.elapsed_ms:.0f}ms)" if r.elapsed_ms > 0 else ""
        logger.info(
            "  %s %-35s [%s] %s%s",
            icon, r.name, r.category, r.detail[:80], elapsed,
        )

    passed = sum(1 for r in results if r.status == CheckStatus.PASS)
    warned = sum(1 for r in results if r.status == CheckStatus.WARN)
    failed = sum(1 for r in results if r.status == CheckStatus.FAIL)

    logger.info("-" * 70)
    logger.info("  总计: %d 通过 | %d 警告 | %d 失败", passed, warned, failed)
    logger.info("=" * 70)
