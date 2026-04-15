from __future__ import annotations


def normalize_text(raw_text: str) -> str:
    """Normalize newlines and collapse excess blank lines."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    compact_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            compact_lines.append(line)
            continue
        blank_run += 1
        if blank_run <= 1:
            compact_lines.append("")

    return "\n".join(compact_lines).strip()
