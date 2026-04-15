from __future__ import annotations


def paragraph_boundaries(text: str) -> list[str]:
    """Split normalized text into paragraph-like segments."""
    return [segment.strip() for segment in text.split("\n\n") if segment.strip()]
