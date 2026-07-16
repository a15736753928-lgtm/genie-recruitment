"""RAG utilities — file validation and kb_id security.

Mirrors patterns from the reference RAG system:
  - kb_id format validation (kb_{uuid8})
  - File extension/size checks
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import get_settings

settings = get_settings()

# ── kb_id format validation ────────────────────────────────

# Format: kb_ followed by exactly 8 hex characters
_KB_ID_PATTERN = re.compile(r"^kb_[a-f0-9]{8}$")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}


def validate_kb_id(kb_id: str) -> bool:
    """Check kb_id format is valid to prevent filter expression injection."""
    return bool(_KB_ID_PATTERN.match(kb_id))


def check_extension(filename: str) -> bool:
    """Check if file extension is supported for ingestion."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def check_size(file_size: int) -> bool:
    """Check if file size is within the configured limit."""
    return file_size <= settings.max_file_size


def get_file_size_mb(file_size: int) -> float:
    """Convert bytes to megabytes."""
    return file_size / (1024 * 1024)


def get_file_type(filename: str) -> str:
    """Normalize file extension to canonical type name."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in ("doc", "docx"):
        return "docx"
    if ext in ("jpg", "jpeg"):
        return "jpg"
    if ext in ("markdown",):
        return "md"
    return ext or "txt"
