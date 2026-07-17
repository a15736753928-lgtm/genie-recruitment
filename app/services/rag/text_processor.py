"""
Text processor for RAG pipeline — 9-step cleaning pipeline.

Blueprint alignment: Section 7.3

DEFAULT_PIPELINE = [
    strip_html,              # 1. Remove HTML tags
    normalize_unicode,       # 2. NFC normalization + zero-width chars
    normalize_fullwidth,     # 3. Fullwidth→halfwidth
    remove_control_chars,    # 4. ASCII controls (keep \n\t)
    strip_markdown,          # 5. Markdown syntax (for .md files)
    filter_header_footer,    # 6. Page numbers / headers
    trim_trailing_noise,     # 7. Tail references/copyright
    normalize_whitespace,    # 8. Multi-space→1, trim lines
    normalize_empty_lines,   # 9. ≥3 empty lines→2
]
"""

import re
import unicodedata


# ── Step 1: HTML tag stripping ─────────────────────────

_HTML_TAG = re.compile(r'<[^>]+>')
_HTML_ENTITY = re.compile(r'&[a-zA-Z]+;|&#\d+;')
_BLOCK_TAGS = re.compile(r'</?(div|p|br|h[1-6]|li|tr|table|section|article)[^>]*>', re.I)


def strip_html(text: str) -> str:
    """Remove HTML tags and entities. Block-level tags become newlines."""
    text = _BLOCK_TAGS.sub('\n', text)
    text = _HTML_TAG.sub('', text)
    text = _HTML_ENTITY.sub(' ', text)
    return text


# ── Step 2: Unicode normalization ──────────────────────

_ZERO_WIDTH = re.compile(r'[​-‏ - ⁠-⁯﻿]')


def normalize_unicode(text: str) -> str:
    """NFC normalization + remove zero-width/invisible characters."""
    text = unicodedata.normalize('NFC', text)
    text = _ZERO_WIDTH.sub('', text)
    return text


# ── Step 3: Fullwidth → Halfwidth conversion ───────────

def normalize_fullwidth(text: str) -> str:
    """Convert fullwidth Latin/numbers to halfwidth, Chinese punct to ASCII."""
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # Fullwidth Latin/numbers/punctuation
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # Fullwidth space
            result.append(' ')
        elif ch in '‘’':  # Chinese single quotes → '
            result.append("'")
        elif ch in '“”':  # Chinese double quotes → "
            result.append('"')
        else:
            result.append(ch)
    return ''.join(result)


# ── Step 4: Control character removal ──────────────────

_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def remove_control_chars(text: str) -> str:
    """Remove ASCII control characters except \n and \t."""
    return _CONTROL_CHARS.sub('', text)


# ── Step 5: Markdown syntax stripping ──────────────────

_MD_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_MD_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
_MD_FORMAT = re.compile(r'(\*\*|__|\*|_|~~|`{1,3})')
_MD_HEADING = re.compile(r'^#{1,6}\s*', re.MULTILINE)
_MD_LIST = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r'^>\s?', re.MULTILINE)
_MD_HORIZ = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')


def strip_markdown(text: str, filename: str = "") -> str:
    """Strip markdown syntax (only for .md files; others pass through)."""
    if filename and not filename.lower().endswith('.md'):
        return text

    text = _MD_CODE_BLOCK.sub(lambda m: m.group(0).replace('```', ''), text)
    text = _MD_IMAGE.sub(r'\1', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_HEADING.sub('', text)
    text = _MD_LIST.sub('', text)
    text = _MD_BLOCKQUOTE.sub('', text)
    text = _MD_HORIZ.sub('', text)
    text = _MD_FORMAT.sub('', text)
    return text


# ── Step 6: Header/footer filtering ────────────────────

_PAGE_NUM = re.compile(
    r'^\s*[-–—]*\s*(第[0-9零一二三四五六七八九十百千]+页|Page\s+\d+|[0-9]+\s*[/-]\s*[0-9]+)\s*[-–—]*\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def filter_header_footer(text: str) -> str:
    """Remove standalone lines matching page number patterns."""
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped and _PAGE_NUM.match(stripped):
            continue
        filtered.append(line)
    return '\n'.join(filtered)


# ── Step 7: Trailing noise trimming ────────────────────

_NOISE_PATTERNS = [
    re.compile(r'^参考文献\s*$'),
    re.compile(r'^References\s*$', re.I),
    re.compile(r'^版权(声明|所有).*$'),
    re.compile(r'^Copyright\s.*$', re.I),
    re.compile(r'^免责声明\s*$'),
    re.compile(r'^All Rights Reserved', re.I),
    re.compile(r'^未经许可.*(不得|禁止).*$'),
]


def trim_trailing_noise(text: str) -> str:
    """Truncate at the first trailing section of references/copyright."""
    lines = text.split('\n')
    tail_start = int(len(lines) * 0.7)  # Only check last 30%

    for i in range(tail_start, len(lines)):
        line = lines[i].strip()
        for pat in _NOISE_PATTERNS:
            if pat.match(line):
                # Verify: need 2+ consecutive noise-like lines
                noise_count = 1
                for j in range(i + 1, min(i + 4, len(lines))):
                    if any(p.match(lines[j].strip()) for p in _NOISE_PATTERNS) or len(lines[j].strip()) < 10:
                        noise_count += 1
                    else:
                        break
                if noise_count >= 2:
                    return '\n'.join(lines[:i])
                break

    return text


# ── Step 8: Whitespace normalization ───────────────────

_MULTI_SPACE = re.compile(r'[ \t]+')


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one, strip each line."""
    lines = [_MULTI_SPACE.sub(' ', line).strip() for line in text.split('\n')]
    return '\n'.join(lines)


# ── Step 9: Empty line compression ─────────────────────

_MULTI_NEWLINE = re.compile(r'\n{3,}')


def normalize_empty_lines(text: str) -> str:
    """Compress 3+ consecutive empty lines into 2."""
    return _MULTI_NEWLINE.sub('\n\n', text)


# ── Pipeline Runner ────────────────────────────────────

DEFAULT_PIPELINE = [
    strip_html,
    normalize_unicode,
    normalize_fullwidth,
    remove_control_chars,
    strip_markdown,          # Note: pass filename for conditional markdown stripping
    filter_header_footer,
    trim_trailing_noise,
    normalize_whitespace,
    normalize_empty_lines,
]


def clean_text(text: str, filename: str = "") -> str:
    """Run the full 9-step cleaning pipeline.

    Args:
        text: Raw text from document parsing.
        filename: Optional filename — used for conditional markdown stripping.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    for step in DEFAULT_PIPELINE:
        try:
            if step is strip_markdown:
                text = step(text, filename)
            else:
                text = step(text)
        except Exception:
            continue

    return text.strip()


def normalize_query(query: str) -> str:
    """Normalize a search query (lighter pipeline)."""
    if not query:
        return ""
    query = normalize_unicode(query)
    query = normalize_fullwidth(query)
    query = normalize_whitespace(query)
    return query.strip()


def extract_highlight_terms(query: str, text: str = "") -> list[str]:
    """Extract key terms from query for highlighting.

    Splits by non-letter/non-Chinese boundaries and returns
    terms that appear in the text (or all terms if text is empty).
    """
    if not query:
        return []

    # Split on non-word boundaries for mixed Chinese/English
    terms = re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', query)
    terms = [t for t in terms if len(t) > 1]

    if text:
        text_lower = text.lower()
        terms = [t for t in terms if t.lower() in text_lower]

    return terms


def highlighter(text: str, terms: list[str]) -> str:
    """Wrap matching terms in <mark> tags (case-insensitive)."""
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
