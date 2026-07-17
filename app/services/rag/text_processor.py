"""Text processor for RAG pipeline — cleaning, normalization, highlighting."""

import re


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def normalize_query(query: str) -> str:
    """Normalize a search query."""
    if not query:
        return ""
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query


def extract_highlight_terms(query: str) -> list:
    """Extract key terms from query for highlighting."""
    if not query:
        return []
    return [t.strip() for t in query.split() if len(t.strip()) > 1]


def highlighter(text: str, terms: list) -> str:
    """Highlight search terms in text."""
    if not text or not terms:
        return text or ""
    for term in terms:
        text = re.sub(
            f'({re.escape(term)})',
            r'<mark>\1</mark>',
            text,
            flags=re.IGNORECASE,
        )
    return text
