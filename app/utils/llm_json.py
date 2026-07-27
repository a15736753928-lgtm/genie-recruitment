"""从 LLM 回复里稳健地抽取 JSON。

全仓此前有十余处都在用 `text.find("{")` + `text.rfind("}")` 截取再 `json.loads`，
这个写法在三种常见情况下必然失败：

1. 模型在 JSON 前后写了带花括号的说明文字 → 截出来的片段括号不配对；
2. 模型输出了多段 JSON（例如先给示例再给结果）→ 首个 `{` 与末个 `}` 跨越了两段；
3. 模型在字符串值里直接换行、或结尾多写一个逗号 → `Expecting ',' delimiter` 之类的报错。

这里用括号配对扫描（能正确跳过字符串字面量与转义）定位候选片段，
失败时再做几项保守修复后重试，全部失败才放弃。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LINE_COMMENT = re.compile(r"(^|\s)//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)

_OPEN = {"{": "}", "[": "]"}


def _strip_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = _FENCE.sub("", out)
    return out.strip()


def _scan_balanced(text: str, start: int) -> Optional[str]:
    """从 text[start] 处的括号开始，返回配对完整的片段；扫到结尾仍未闭合则返回 None。"""
    opener = text[start]
    closer = _OPEN[opener]
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _escape_raw_newlines(fragment: str) -> str:
    """把字符串字面量内部的裸换行/制表符转义掉（模型经常直接换行写多行文本）。"""
    out = []
    in_string = False
    escaped = False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
                continue
            if ch == "\\":
                escaped = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _repair(fragment: str) -> str:
    out = _BLOCK_COMMENT.sub("", fragment)
    out = _LINE_COMMENT.sub(r"\1", out)
    out = _escape_raw_newlines(out)
    out = _TRAILING_COMMA.sub(r"\1", out)
    return out


def extract_json(text: str, expect: str = "any") -> Optional[Any]:
    """抽取并解析 LLM 回复中的 JSON。

    expect: "object" 只取 dict，"array" 只取 list，"any" 都接受。
    解析不出来返回 None —— 调用方需要自行处理，不要假设一定成功。
    """
    if not text:
        return None

    body = _strip_fences(text)

    # 整段就是合法 JSON 是最常见的情况，先直接试一次
    for candidate in (body, _repair(body)):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            pass
        else:
            if _matches(parsed, expect):
                return parsed

    openers = "{" if expect == "object" else "[" if expect == "array" else "{["
    for idx, ch in enumerate(body):
        if ch not in openers:
            continue
        fragment = _scan_balanced(body, idx)
        if fragment is None:
            continue
        for candidate in (fragment, _repair(fragment)):
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if _matches(parsed, expect):
                return parsed
    return None


def _matches(parsed: Any, expect: str) -> bool:
    if expect == "object":
        return isinstance(parsed, dict)
    if expect == "array":
        return isinstance(parsed, list)
    return isinstance(parsed, (dict, list))


def extract_json_object(text: str) -> Optional[dict]:
    result = extract_json(text, expect="object")
    return result if isinstance(result, dict) else None


def extract_json_array(text: str) -> Optional[list]:
    result = extract_json(text, expect="array")
    return result if isinstance(result, list) else None
