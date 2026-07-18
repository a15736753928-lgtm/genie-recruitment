"""
Semantic-aware text chunking with context bridging.

Blueprint alignment: Section 7.4

Two strategies:
  - split_text_sentence(): Sentence-boundary-aware for structured docs
  - split_text_semantic(): Embedding-similarity-based for long documents

Both include _bridge_context() for overlap via preceding context.
"""

import re
from typing import List, Optional
from app.config import get_settings

settings = get_settings()

_SENT_PATTERN = re.compile(r'(?<=[。！？.!?\n])\s*')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries."""
    raw = _SENT_PATTERN.split(text)
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


def _bridge_context(chunks: list[str], overlap: int = 100) -> list[str]:
    """Prepend tail of previous chunk as context bridge.

    For each chunk (except first), prepend ~overlap chars from prev chunk's tail,
    aligned to nearest sentence boundary to avoid mid-sentence cuts.
    """
    if not chunks or overlap <= 0:
        return chunks

    bridged = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        current = chunks[i]

        ideal_start = max(0, len(prev) - overlap)
        margin = min(50, overlap // 2)
        search_start = max(0, ideal_start - margin)
        search_end = min(len(prev), ideal_start + margin)

        best_pos = ideal_start
        for boundary in [r'[。！？.!?]', r'\n', r'；;', r'，,']:
            matches = list(re.finditer(boundary, prev[search_start:search_end]))
            if matches:
                best = min(matches, key=lambda m: abs((search_start + m.end()) - ideal_start))
                best_pos = search_start + best.end()
                break

        bridge = prev[best_pos:].lstrip()
        if bridge:
            bridged.append(bridge + "\n" + current)
        else:
            bridged.append(current)

    return bridged


def split_text_sentence(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Sentence-boundary-aware chunking (primary strategy for structured docs)."""
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

    min_size = settings.min_chunk_size
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] += chunk
        else:
            merged.append(chunk)

    return _bridge_context(merged, chunk_overlap)


def split_text_semantic(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Semantic splitting via embedding-similarity breakpoints.

    Uses LlamaIndex SemanticSplitterNodeParser to find natural
    topic boundaries where adjacent sentence embeddings diverge.

    Falls back to sentence splitting if LlamaIndex is unavailable.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        # Single sentence or empty → use sentence split
        return split_text_sentence(text, chunk_size, chunk_overlap)

    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from app.infrastructure.model_loader import _get_model as _get_embedding

        # Get the embedding model (SentenceTransformer compatible)
        embed_model = _get_embedding()
        if embed_model is None:
            raise RuntimeError("Embedding model not loaded")

        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95,
        )

        # LlamaIndex expects Document objects
        from llama_index.core import Document
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])

        chunks = []
        for node in nodes:
            node_text = node.get_content()
            if node_text.strip():
                chunks.append(node_text.strip())

        if not chunks:
            return split_text_sentence(text, chunk_size, chunk_overlap)

        # Merge short chunks
        merged = []
        for chunk in chunks:
            if merged and len(chunk) < settings.min_chunk_size:
                merged[-1] += "\n" + chunk
            else:
                merged.append(chunk)

        return _bridge_context(merged, chunk_overlap)

    except ImportError:
        # LlamaIndex not available → fallback
        return split_text_sentence(text, chunk_size, chunk_overlap)
    except Exception:
        return split_text_sentence(text, chunk_size, chunk_overlap)


def split_text_simple(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size chunking with overlap (fallback)."""
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
            start = chunk_size

    return chunks


def chunk_document(
    text: str,
    file_type: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[str]:
    """Main entry point: chunk cleaned text for ingestion.

    Uses semantic splitting for long documents (>2000 chars),
    sentence-boundary splitting for medium documents,
    and single-chunk for very short text.
    """
    if not text or not text.strip():
        return []

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()]

    # Semantic splitting for long documents
    if len(text) > 2000:
        try:
            chunks = split_text_semantic(text, chunk_size, chunk_overlap)
            if chunks and len(chunks) >= 1:
                return chunks
        except Exception:
            pass

    try:
        chunks = split_text_sentence(text, chunk_size, chunk_overlap)
        if chunks:
            return chunks
    except Exception:
        pass

    return split_text_simple(text, chunk_size, chunk_overlap)
