"""
Startup dependency health-check system.

Runs before the server accepts requests. Every check returns a CheckResult.
Critical checks (DB, upload dir) fail-fast and prevent startup.
Optional checks (RAG stack) and config checks run concurrently and warn on failure.

Usage:
    from app.infrastructure.startup_checks import run_startup_checks, print_check_summary
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


def _find_port_owner(port: int) -> str:
    """Best-effort lookup of the PID holding a port, for a friendly error hint.

    Parses ``netstat -ano`` output (Windows). Distinguishes a live process
    (suggest ``taskkill``) from a dead-process ghost socket (needs a reboot).
    Returns an empty string if the owner can't be determined.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return ""
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    if not pids:
        return ""

    live, ghost = [], []
    for pid in sorted(pids):
        # tasklist exit code 0 means the PID is still a running process.
        ret = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        if ret.returncode == 0 and pid in ret.stdout:
            live.append(pid)
        else:
            ghost.append(pid)
    if live:
        return (f" — held by PID {', '.join(live)}; "
                f"run: taskkill /F /PID <pid>")
    return (" — held by ghost socket from dead PID "
            f"{', '.join(ghost)} (taskkill won't help; reboot to clear)")


async def check_port_available(settings: Settings) -> CheckResult:
    """Fail fast if the app port is already in use by ANOTHER process.

    Runs as a preflight BEFORE model loading, so a port conflict surfaces in
    milliseconds rather than after ~20s of model loading.

    Implementation note: we use connect() (try to reach whoever is listening)
    rather than bind(). Under uvicorn --reload the reloader process binds the
    socket and passes it to the worker; the lifespan startup check runs inside
    the worker, so a bind()-based check would always collide with uvicorn's own
    inherited socket and falsely report "already in use". connect() only
    succeeds when some OTHER process is actually accepting connections on the
    port, which is exactly the conflict we want to catch.

    Windows ghost-socket handling: on Windows, a killed process may leave a
    LISTEN socket that still accepts connections (ghost socket). To distinguish
    a real conflict from a reclaimable ghost, we attempt bind() with
    SO_REUSEADDR after connect() succeeds — if bind works, the port can be
    reclaimed by uvicorn (which also uses SO_REUSEADDR), so we WARN instead
    of FAIL.
    """
    import socket
    port = settings.app_port
    t0 = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        # If connect succeeds, someone else is already listening → conflict.
        sock.connect(("127.0.0.1", port))
        sock.close()

        # Try SO_REUSEADDR bind to see if the port is reclaimable (ghost socket)
        reclaimable = False
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_sock.bind((settings.app_host, port))
            test_sock.listen(1)
            reclaimable = True
        except OSError:
            pass
        finally:
            try:
                test_sock.close()
            except OSError:
                pass

        if reclaimable:
            hint = await asyncio.to_thread(_find_port_owner, port)
            return CheckResult(
                f"Port {port}", "critical", CheckStatus.PASS,
                f"Reclaimable (SO_REUSEADDR) — ghost socket will be overridden{hint}",
                (time.perf_counter() - t0) * 1000,
            )

        hint = await asyncio.to_thread(_find_port_owner, port)
        return CheckResult(
            f"Port {port}", "critical", CheckStatus.FAIL,
            f"Already in use{hint}",
            (time.perf_counter() - t0) * 1000,
        )
    except (OSError, ConnectionRefusedError, socket.timeout):
        # Connection refused / timed out → port is free for us.
        return CheckResult(
            f"Port {port}", "critical", CheckStatus.PASS,
            "Available",
            (time.perf_counter() - t0) * 1000,
        )
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Optional / RAG stack checks (non-blocking — WARN on failure)
# ═══════════════════════════════════════════════════════════════════

async def check_milvus_lite(settings: Settings) -> CheckResult:
    """Verify Milvus Lite vector store is accessible."""
    t0 = time.perf_counter()
    try:
        from pymilvus import MilvusClient
        from app.infrastructure.milvus_manager import _MILVUS_GRPC_OPTIONS
        client = MilvusClient(
            settings.milvus_db_path,
            grpc_options=_MILVUS_GRPC_OPTIONS,
        )
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
        from app.infrastructure.model_loader import preload_models
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


async def check_funasr(settings: Settings) -> CheckResult:
    """Verify the FunASR speech-to-text pipeline is available.

    Critical — interview audio transcription is a core feature. The server
    refuses to start if the model or its dependencies are missing so the
    user gets a clear error at startup rather than a runtime mystery.
    """
    t0 = time.perf_counter()
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import funasr  # noqa: F401
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        pipe = pipeline(
            task=Tasks.auto_speech_recognition,
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        return CheckResult(
            "FunASR (语音转写)", "critical", CheckStatus.PASS,
            f"Paraformer model ready (pipeline: {type(pipe).__name__})",
            (time.perf_counter() - t0) * 1000,
        )
    except ImportError as e:
        return CheckResult(
            "FunASR (语音转写)", "optional", CheckStatus.WARN,
            f"依赖缺失（语音转写不可用）— 运行 asr_init.py 安装: {e}",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "FunASR (语音转写)", "optional", CheckStatus.WARN,
            f"模型加载失败（语音转写不可用）— 运行 asr_init.py 下载模型: {e}",
            (time.perf_counter() - t0) * 1000,
        )


async def check_minio(settings: Settings) -> CheckResult:
    """Verify MinIO is reachable and the bucket is ready.

    Critical — MinIO is a hard dependency for all document uploads. If it is
    unreachable the server refuses to start (fail-fast), rather than limping
    along with broken upload/preview/download endpoints.
    """
    t0 = time.perf_counter()
    try:
        from app.infrastructure import minio_storage
        if not settings.minio_enabled:
            return CheckResult(
                "MinIO", "critical", CheckStatus.FAIL,
                "MINIO_ENABLED=false — but MinIO is a required dependency; "
                "set MINIO_ENABLED=true and start MinIO before launching",
                (time.perf_counter() - t0) * 1000,
            )
        ok = await asyncio.to_thread(minio_storage.ensure_bucket)
        if ok:
            return CheckResult(
                "MinIO", "critical", CheckStatus.PASS,
                f"Connected, bucket '{settings.minio_bucket}' ready",
                (time.perf_counter() - t0) * 1000,
            )
        return CheckResult(
            "MinIO", "critical", CheckStatus.FAIL,
            f"Unreachable at {settings.minio_endpoint} — start MinIO before "
            f"launching the backend (mc alias set myminio http://{settings.minio_endpoint} admin 12345678)",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "MinIO", "critical", CheckStatus.FAIL,
            f"Init error: {e}",
            (time.perf_counter() - t0) * 1000,
        )


async def check_portrait_gender(settings: Settings) -> CheckResult:
    """Verify the portrait gender-recognition capability is available.

    Critical — resume parsing relies on this to infer gender from the headshot
    when the resume text doesn't state it. Requires ``cv2`` and ``onnxruntime``
    plus the LCNet ONNX gender model (auto-downloaded from the HF mirror on
    first start). If any piece is missing/unloadable the server refuses to
    start, rather than silently producing "未知" genders.
    """
    t0 = time.perf_counter()
    try:
        import cv2  # noqa: F401
    except ImportError:
        return CheckResult(
            "Portrait Gender", "critical", CheckStatus.FAIL,
            "opencv-python (cv2) not installed — needed for headshot gender inference",
            (time.perf_counter() - t0) * 1000,
        )
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return CheckResult(
            "Portrait Gender", "critical", CheckStatus.FAIL,
            "onnxruntime not installed — needed to run the gender ONNX model",
            (time.perf_counter() - t0) * 1000,
        )

    def _obtain_and_load():
        from app.services.recruitment.portrait_gender import _ensure_gender_model, _load_gender_classifier
        _ensure_gender_model()          # download the ONNX file if missing
        _load_gender_classifier()        # load + cache the InferenceSession

    try:
        await asyncio.to_thread(_obtain_and_load)
        return CheckResult(
            "Portrait Gender", "critical", CheckStatus.PASS,
            "cv2 + onnxruntime ready, LCNet gender model loaded",
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "Portrait Gender", "critical", CheckStatus.FAIL,
            f"Failed to obtain/load gender model: {e}",
            (time.perf_counter() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════
# Config validation checks (non-blocking — WARN on issues)
# ═══════════════════════════════════════════════════════════════════

async def check_deepseek_api_key(settings: Settings) -> CheckResult:
    """Verify at least one LLM API key is configured (env bootstrap)."""
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    secrets = [k for k in keys if k and k != "sk-your-api-key-here"]
    if not secrets:
        return CheckResult(
            "LLM API Key（env bootstrap）", "config", CheckStatus.FAIL,
            "Not configured; AI agent features will not work",
        )
    return CheckResult(
        "LLM API Key（env bootstrap）", "config", CheckStatus.PASS,
        f"Configured ({len(secrets)} key(s): {secrets[0][:8]}...)",
    )


def _format_exc_chain(exc: BaseException) -> str:
    """Format exception + cause chain; httpx ConnectError often has empty str()."""
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen and len(parts) < 6:
        seen.add(id(cur))
        text = str(cur).strip() or repr(cur)
        parts.append(f"{type(cur).__name__}: {text}")
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)


async def check_deepseek_api_reachable(settings: Settings) -> CheckResult:
    """Verify LLM API endpoint is reachable (first configured key).

    DeepSeek 的 TLS 握手在本机偶发 ``SSLEOFError: UNEXPECTED_EOF_WHILE_READING``，
    被 httpx 包成空消息的 ``ConnectError('')``。这是瞬时网络/TLS 抖动，不是 key
    失效——所以对连接类错误做有限重试；HTTP 4xx/5xx 不重试（那是真实配置问题）。
    """
    keys = [k.strip() for k in settings.deepseek_api_key.split(",") if k.strip()]
    secrets = [k for k in keys if k and k != "sk-your-api-key-here"]
    if not secrets:
        return CheckResult(
            "LLM API Reachable", "config", CheckStatus.PASS,
            "Skipped (no key configured)",
        )

    import httpx

    base = settings.deepseek_base_url.rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    headers = {"Authorization": f"Bearer {secrets[0]}"}
    t0 = time.perf_counter()
    last_error = ""
    attempts = 3

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                detail = f"Reachable ({resp.status_code})"
                if attempt > 1:
                    detail += f" after {attempt} attempts"
                return CheckResult(
                    "LLM API Reachable", "config", CheckStatus.PASS,
                    detail,
                    elapsed,
                )
            # 非 200：key/权限/网关问题，重试无意义
            return CheckResult(
                "LLM API Reachable", "config", CheckStatus.FAIL,
                f"HTTP {resp.status_code}: {resp.text[:100]}",
                elapsed,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = _format_exc_chain(e)
            logger.warning(
                "LLM API probe attempt %d/%d failed: %s",
                attempt, attempts, last_error,
            )
            if attempt < attempts:
                await asyncio.sleep(0.4 * attempt)
                continue
        except Exception as e:
            last_error = _format_exc_chain(e)
            break

    return CheckResult(
        "LLM API Reachable", "config", CheckStatus.FAIL,
        f"Unreachable after {attempts} attempts: {last_error}",
        (time.perf_counter() - t0) * 1000,
    )


async def check_llm_config_store(settings: Settings) -> CheckResult:
    """Verify LLM config can be loaded from DB (or env bootstrap fallback)."""
    t0 = time.perf_counter()
    try:
        from app.database import SyncSessionFactory
        from app.services.ai.config_store import load_llm_config as load_llm_config_sync
        import asyncio

        async def _check():
            from app.database import async_session_factory
            async with async_session_factory() as db:
                config = await load_llm_config_sync(db)
                providers = config.get("providers", [])
                enabled = [p for p in providers if p.get("enabled", True)]
                if not enabled:
                    return False, "No enabled providers (env bootstrap failed?)"
                total_keys = sum(
                    len([k for k in p.get("apiKeys", []) if k.get("enabled", True)])
                    for p in enabled
                )
                return True, f"{len(enabled)} provider(s), {total_keys} key(s) active"

        ok, detail = await _check()
        return CheckResult(
            "LLM Config Store", "config",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            detail,
            (time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "LLM Config Store", "config", CheckStatus.FAIL,
            f"Cannot load: {e}",
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

    Preflight: port availability (fail-fast — aborts before model loading).
    Phase 1: Infrastructure checks (sequential — DB before everything else)
    Phase 2: RAG stack checks (concurrent)
    Phase 3: Config validation (concurrent)

    All phases always run to completion so the user sees the full picture
    of what's broken in one restart cycle — EXCEPT the port preflight, which
    aborts immediately on failure (a port conflict is a hard blocker; loading
    ~20s of models first would be pure waste).
    """
    checks_phase1: list[CheckFn] = [
        check_database_url_format,
        check_database_async,
        check_database_sync,
        check_upload_dir,
        check_minio,
        check_portrait_gender,
    ]
    checks_phase2: list[CheckFn] = [
        check_milvus_lite,
        check_embedding_model,
        check_reranker_model,
        check_rapidocr,
        check_funasr,
    ]
    checks_phase3: list[CheckFn] = [
        check_deepseek_api_key,
        check_deepseek_api_reachable,
        check_llm_config_store,
        check_cors_origins,
    ]

    results: list[CheckResult] = []

    # Preflight: port availability — BEFORE _ensure_imports and model loading,
    # so a port conflict fails in milliseconds instead of after ~20s.
    port_result = await _run_check(check_port_available, settings, "critical")
    results.append(port_result)
    if port_result.status == CheckStatus.FAIL:
        logger.error("端口被占用，跳过其余检查以快速失败。")
        return results

    # Pre-import heavy modules to avoid import race conditions when
    # concurrent check threads import from the same package (transformers).
    _ensure_imports()

    total = (len(checks_phase1) + len(checks_phase2) + len(checks_phase3)) + 1
    logger.info("开始启动健康检查（共 %d 项，含端口预检）...", total)

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
