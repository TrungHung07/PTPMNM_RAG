from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer for deterministic testable chunking."""
    return [token for token in text.split() if token]


def count_tokens(text: str) -> int:
    return len(tokenize(text))
