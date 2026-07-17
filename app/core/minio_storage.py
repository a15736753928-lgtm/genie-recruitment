"""MinIO object storage for uploaded documents.

All uploaded files (PDF / Word / Markdown / text / images) are stored in MinIO
instead of the local filesystem. The DB columns that previously held local file
paths (e.g. ``object_key``, ``resume_file``, ``file_path``) now hold MinIO
object keys such as ``rag/<uuid>.pdf``.

This module is the single integration point with the ``minio`` SDK. Callers
treat object keys as opaque strings and use:

  - ``upload_bytes(key, data, content_type)``  — store a file
  - ``download_to_temp(key, suffix)``          — get a local temp copy (caller cleans up)
  - ``resolved_local_path(stored)``            — context manager: yields a local
                                                 path for either a MinIO key or a
                                                 legacy on-disk path, cleaning up
                                                 any temp file on exit
  - ``get_object_stream(key)``                 — raw streaming response (for HTTP)
  - ``delete_object(key)``                      — remove a file
  - ``object_exists(key)``                      — existence check
  - ``ensure_bucket()``                         — create the bucket if missing
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Generator, Optional

from minio import Minio
from minio.error import S3Error

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_client: Optional[Minio] = None


def _get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def is_enabled() -> bool:
    """Whether MinIO storage is active (config + reachability)."""
    if not settings.minio_enabled:
        return False
    return ensure_bucket()


def ensure_bucket() -> bool:
    """Create the configured bucket if it doesn't exist. Returns True on success."""
    try:
        client = _get_client()
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
            logger.info("Created MinIO bucket: %s", settings.minio_bucket)
        return True
    except S3Error as e:
        logger.error("MinIO ensure_bucket failed: %s", e)
        return False
    except Exception as e:  # noqa: BLE001 — never let storage init crash the app
        logger.error("MinIO init error: %s", e)
        return False


def upload_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload raw bytes to MinIO under ``object_name``."""
    client = _get_client()
    client.put_object(
        settings.minio_bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def object_exists(object_name: str) -> bool:
    if not object_name:
        return False
    try:
        _get_client().stat_object(settings.minio_bucket, object_name)
        return True
    except S3Error:
        return False
    except Exception:  # noqa: BLE001
        return False


def download_to_temp(object_name: str, suffix: str = "") -> str:
    """Download an object to a temp file and return its path. Caller must remove it.

    When ``suffix`` is empty, it is derived from the object key's extension so
    that the temp file keeps its ``.pdf``/``.docx``/... suffix — many parsers
    (and our own ``extract_text`` helpers) branch on the file extension.
    """
    if not object_name:
        raise ValueError("empty object name")
    if not suffix:
        suffix = os.path.splitext(object_name)[1]
    response = _get_client().get_object(settings.minio_bucket, object_name)
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix or "")
        with os.fdopen(fd, "wb") as f:
            for chunk in response.stream(amt=64 * 1024):
                f.write(chunk)
        return tmp_path
    finally:
        response.close()
        response.release_conn()


def get_object_stream(object_name: str):
    """Return the raw MinIO response object for streaming to an HTTP client.

    The caller is responsible for closing the returned response (use it as a
    FastAPI StreamingResponse body; FastAPI/Starlette will close it).
    """
    return _get_client().get_object(settings.minio_bucket, object_name)


def delete_object(object_name: str) -> bool:
    if not object_name:
        return False
    try:
        _get_client().remove_object(settings.minio_bucket, object_name)
        return True
    except S3Error as e:
        logger.warning("MinIO delete failed for %s: %s", object_name, e)
        return False
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def resolved_local_path(stored: str, suffix: str = "") -> Generator[str, None, None]:
    """Yield a local file path for a stored value.

    Handles two cases transparently:
      1. Legacy absolute local path that still exists on disk → use directly,
         no cleanup (old data before the MinIO migration).
      2. Anything else → treated as a MinIO object key, downloaded to a temp
         file which is removed on context exit.

    Yields "" when ``stored`` is empty.
    """
    if not stored:
        yield ""
        return
    # Legacy on-disk path (pre-MinIO data) — use in place, no cleanup.
    if os.path.isabs(stored) and os.path.exists(stored):
        yield stored
        return
    # MinIO object key — download to a temp file and clean up afterwards.
    tmp_path = download_to_temp(stored, suffix)
    try:
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
