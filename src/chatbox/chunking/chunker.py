from __future__ import annotations

from hashlib import sha256

from chatbox.chunking.tokenizer import tokenize
from chatbox.domain.models import Chunk


def _stable_chunk_id(document_id: str, chunk_order: int, text: str) -> str:
    digest = sha256(f"{document_id}|{chunk_order}|{text}".encode()).hexdigest()
    return f"{document_id}-{digest[:16]}"


def _stable_checksum(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    document_id: str,
    max_tokens: int = 300,
    overlap_tokens: int = 30,
) -> list[Chunk]:
    tokens = tokenize(text)
    if not tokens:
        return []

    if overlap_tokens >= max_tokens:
        overlap_tokens = max_tokens - 1

    chunks: list[Chunk] = []
    start = 0
    order = 0
    stride = max_tokens - overlap_tokens

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = " ".join(chunk_tokens)
        chunks.append(
            Chunk(
                chunk_id=_stable_chunk_id(document_id, order, chunk_text_value),
                document_id=document_id,
                chunk_order=order,
                text=chunk_text_value,
                token_count=len(chunk_tokens),
                overlap_tokens=overlap_tokens if order > 0 else 0,
                boundary_type="token_window",
                checksum=_stable_checksum(chunk_text_value),
            )
        )
        order += 1
        if end >= len(tokens):
            break
        start += stride

    return chunks
