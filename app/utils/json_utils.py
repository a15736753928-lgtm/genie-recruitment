"""LLM JSON extraction — single source of truth.

Replaces multiple ad-hoc implementations:
- ``resumes.py:_extract_json_from_llm``
- ``agent_chat.py:_generate_plan`` fence-removal
- Various other JSON-from-LLM extraction snippets.

Usage::

    from app.utils.json_utils import extract_json_from_text
    clean = extract_json_from_text(llm_output)
    parsed = json.loads(clean)
"""

from __future__ import annotations

import re


def extract_json_from_text(content: str, *, fix_trailing_comma: bool = True) -> str:
    """Extract a JSON string from potentially noisy LLM output.

    Handles:
    - Markdown code fences (`` ```json ... ``` ``)
    - Prose before/after the JSON object
    - Trailing commas (the most common JSON formatting error)
    """
    content = _extract_bracket(content, "{", "}")
    if fix_trailing_comma:
        content = re.sub(r",\s*([}\]])", r"\1", content)
    return content


def extract_json_array_from_text(content: str, *, fix_trailing_comma: bool = True) -> str:
    """Like ``extract_json_from_text`` but for JSON arrays ``[...]``.

    Useful when the LLM returns a top-level JSON array.
    """
    content = _extract_bracket(content, "[", "]")
    if fix_trailing_comma:
        content = re.sub(r",\s*([\]])", r"\1", content)
    return content

def _extract_bracket(content: str, open_bracket: str, close_bracket: str) -> str:
    """Extract the outermost bracket-delimited block from LLM output."""
    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    start = content.find(open_bracket)
    if start == -1:
        return content
    end = content.rfind(close_bracket)
    if end > start:
        content = content[start:end + 1]

    return content
