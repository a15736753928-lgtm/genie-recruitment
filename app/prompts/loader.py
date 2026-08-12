"""提示词文件读取与严格变量渲染。"""

from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import Mapping

PROMPTS_DIR = Path(__file__).resolve().parent
_SECTION_RE = re.compile(
    r"^##\s+(Prompt\s+\d+|System|User|Intro|Output)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_VARIABLE_RE = re.compile(r"\[\[([A-Za-z_][A-Za-z0-9_]*)\]\]")
_CACHE: dict[Path, tuple[int, str]] = {}
_LOCK = RLock()


class PromptError(ValueError):
    """提示词路径、分段或变量不合法。"""


def _resolve(relative_path: str) -> Path:
    candidate = (PROMPTS_DIR / relative_path).resolve()
    if candidate == PROMPTS_DIR or PROMPTS_DIR not in candidate.parents:
        raise PromptError(f"提示词路径越界: {relative_path}")
    if candidate.suffix.lower() != ".md":
        raise PromptError(f"提示词必须是 .md 文件: {relative_path}")
    if not candidate.is_file():
        raise PromptError(f"提示词文件不存在: {relative_path}")
    return candidate


def _read(path: Path) -> str:
    mtime = path.stat().st_mtime_ns
    with _LOCK:
        cached = _CACHE.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        content = path.read_text(encoding="utf-8").strip()
        _CACHE[path] = (mtime, content)
        return content


def _section(content: str, name: str, relative_path: str) -> str:
    matches = list(_SECTION_RE.finditer(content))
    for index, match in enumerate(matches):
        if match.group(1).strip().casefold() == name.strip().casefold():
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            return content[start:end].strip()
    raise PromptError(f"提示词分段不存在: {relative_path}#{name}")


def load_prompt(relative_path: str, section: str | None = None) -> str:
    """读取 app/prompts 下的提示词；section 对应 Markdown 二级标题。"""
    content = _read(_resolve(relative_path))
    return _section(content, section, relative_path) if section else content


def render_prompt(
    relative_path: str,
    variables: Mapping[str, object],
    section: str | None = None,
) -> str:
    """严格替换 ``[[name]]``；缺失、额外或残留变量均报错。"""
    template = load_prompt(relative_path, section)
    expected = set(_VARIABLE_RE.findall(template))
    supplied = set(variables)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"缺少变量: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"未使用变量: {', '.join(sorted(extra))}")
        raise PromptError(f"提示词变量不匹配 {relative_path}: {'; '.join(details)}")
    rendered = _VARIABLE_RE.sub(lambda match: str(variables[match.group(1)]), template)
    residual = _VARIABLE_RE.findall(rendered)
    if residual:
        raise PromptError(f"提示词仍有未渲染变量 {relative_path}: {', '.join(sorted(set(residual)))}")
    return rendered
