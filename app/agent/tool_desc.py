"""Standard tool description builder for LLM tool selection."""


def tool_desc(purpose: str, when: str, avoid: str, example: str = "") -> str:
    """Build a consistent tool description: 用途 | 何时用 | 勿用 | 例."""
    parts = [f"用途：{purpose}", f"何时用：{when}", f"勿用：{avoid}"]
    if example:
        parts.append(f"例：{example}")
    return " ".join(parts)
