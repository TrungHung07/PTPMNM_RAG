"""
src/rag/rerank.py — Cross-encoder re-ranking utilities.

Mục tiêu:
- Load cross-encoder model **một lần** (singleton) để dùng lại giữa các request.
- Re-rank danh sách Document đã retrieve theo điểm relevance (query, chunk).
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.documents import Document


def _rerank_enabled(default: bool = False) -> bool:
    raw = os.getenv("RERANK_ENABLED")
    if raw is None:
        return default
    return raw.strip().lower() == "true"


@lru_cache(maxsize=1)
def get_reranker():
    """
    Trả về CrossEncoder singleton (load 1 lần).

    Lưu ý: import sentence-transformers lazy để tránh overhead khi rerank tắt.
    """
    from sentence_transformers import CrossEncoder

    model_name = os.getenv(
        "RERANK_MODEL",
        # Default nhẹ, đủ để validate pipeline/latency trước khi thay model khác
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    # Nếu muốn ép device: cpu/cuda
    device = os.getenv("RERANK_DEVICE")
    if device:
        return CrossEncoder(model_name, device=device)
    return CrossEncoder(model_name)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def rerank_documents(
    query: str,
    docs: list[Document], # top ung cu vien
    *,
    top_k: int,
    max_chars: int,
    batch_size: int | None = None,
    enabled: bool | None = None,
) -> tuple[list[Document], list[float] | None]:
    """
    Re-rank docs bằng cross-encoder.

    Returns:
        (docs_sorted_topk, scores_topk_or_none)
    """
    if enabled is None:
        enabled = _rerank_enabled(default=False)
    if not enabled:
        return docs[: max(0, top_k)], None

    if not docs or top_k <= 0:
        return [], []

    reranker = get_reranker()

    # Truncate để ổn định latency + tránh vượt input limit
    pairs: list[tuple[str, str]] = [
        (query, _truncate(d.page_content or "", max_chars)) for d in docs
    ]

    if batch_size is None:
        batch_size = int(os.getenv("RERANK_BATCH_SIZE", "16"))

    scores: list[float] = list(reranker.predict(pairs, batch_size=batch_size))

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ranked = ranked[: min(top_k, len(ranked))]

    docs_top = [d for d, _ in ranked]
    scores_top = [float(s) for _, s in ranked]
    return docs_top, scores_top


def warmup_reranker() -> None:
    """
    Warm-up model để tránh spike request đầu tiên.
    Không làm gì nếu rerank đang tắt.
    """
    if not _rerank_enabled(default=False):
        return
    model = get_reranker()
    # chạy 1 batch dummy ngắn để load weights/caches
    _ = model.predict([("warmup", "warmup")], batch_size=1)


__all__ = [
    "get_reranker",
    "rerank_documents",
    "warmup_reranker",
]
