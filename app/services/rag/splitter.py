"""Semantic-aware text chunking with context bridging.

Blueprint alignment: Section 7.4
Two strategies:
  - sentence_split(): Sentence-boundary-aware splitting for structured docs
  - semantic_split(): Similarity-based splitting for long documents

Both include _bridge_context() for overlap via preceding context.
"""

import re
from typing import List, Optional
from app.config import get_settings

settings = get_settings()

# Sentence boundary pattern (Chinese + English)
_SENT_PATTERN = re.compile(
    r'(?<=[。！？.!?\n])\s*'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries."""
    raw = _SENT_PATTERN.split(text)
    # Re-attach the punctuation to each segment
    sentences = []
    buf = ""
    for part in raw:
        buf += part
        if buf.rstrip() and (buf.rstrip()[-1] in '。！？.!?\n'):
            sentences.append(buf)
            buf = ""
    if buf.strip():
        sentences.append(buf)
    return [s.strip() for s in sentences if s.strip()]


def _bridge_context(
    chunks: list[str],
    overlap: int = 100,
) -> list[str]:
    """Prepend tail of previous chunk as context bridge.

    Algorithm from blueprint:
    - For each chunk (except first), prepend ~overlap chars from prev chunk's tail
    - Align cut point to nearest sentence boundary
    - Prevents context from inflating across multiple bridges
    """
    if not chunks or overlap <= 0:
        return chunks

    bridged = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        current = chunks[i]

        # Take tail of previous chunk
        ideal_start = max(0, len(prev) - overlap)
        # Search for best sentence boundary near ideal_start (±margin)
        margin = min(50, overlap // 2)
        search_start = max(0, ideal_start - margin)
        search_end = min(len(prev), ideal_start + margin)

        # Find the nearest sentence boundary
        best_pos = ideal_start
        for sent_pat in [r'[。！？.!?]', r'\n', r'；;', r'，,']:
            matches = list(re.finditer(sent_pat, prev[search_start:search_end]))
            if matches:
                # Pick the boundary closest to ideal_start
                best = min(matches, key=lambda m: abs((search_start + m.end()) - ideal_start))
                best_pos = search_start + best.end()
                break

        bridge = prev[best_pos:].lstrip()
        if bridge:
            bridged.append(bridge + "\n" + current)
        else:
            bridged.append(current)

    return bridged


def split_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Sentence-boundary-aware chunking.

    Splits at sentence boundaries, merges short sentences up to chunk_size,
    then applies context bridging.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current.strip():
                chunks.append(current.strip())
            # If a single sentence exceeds chunk_size, force-split it
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size - chunk_overlap):
                    piece = sent[i:i + chunk_size]
                    if piece.strip():
                        chunks.append(piece.strip())
                current = ""
            else:
                current = sent

    if current.strip():
        chunks.append(current.strip())

    # Merge very short chunks (< 40 chars) into previous
    min_size = settings.min_chunk_size
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] += chunk
        else:
            merged.append(chunk)

    # Apply context bridging
    return _bridge_context(merged, chunk_overlap)


def split_text_simple(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Fixed-size chunking with overlap (fallback for when sentence split fails)."""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= text_len:
            break
        if start <= 0:
            start = chunk_size  # safety

    return chunks


def chunk_document(
    text: str,
    file_type: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Main entry point: chunk cleaned text for ingestion.

    Uses sentence-boundary split as primary strategy,
    falls back to fixed-size for very short or unstructured text.
    """
    if not text or not text.strip():
        return []

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # For very short text, return as single chunk
    if len(text) <= chunk_size:
        return [text.strip()]

    # Primary: sentence-boundary split
    try:
        chunks = split_text(text, chunk_size, chunk_overlap)
        if chunks and len(chunks) > 0:
            return chunks
    except Exception:
        pass

    # Fallback: fixed-size split
    return split_text_simple(text, chunk_size, chunk_overlap)
