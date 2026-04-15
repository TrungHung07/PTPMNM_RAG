from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from chatbox.domain.models import EmbeddingRecord
from chatbox.rag.embeddings import EmbeddingService
from chatbox.storage.ports import RagHit


@dataclass(slots=True)
class _StoredVector:
    chunk_id: str
    vector: list[float]
    text: str


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: list[_StoredVector] = []

    def upsert_embeddings(self, records: list[EmbeddingRecord | dict]) -> None:
        for record in records:
            if isinstance(record, dict):
                chunk_id = record["chunk_id"]
                vector = list(record["vector"])
                text = record.get("text", "")
            else:
                chunk_id = record.chunk_id
                vector = list(record.vector)
                text = ""
            self._vectors = [item for item in self._vectors if item.chunk_id != chunk_id]
            self._vectors.append(_StoredVector(chunk_id=chunk_id, vector=vector, text=text))

    def attach_text(self, chunk_id: str, text: str) -> None:
        for item in self._vectors:
            if item.chunk_id == chunk_id:
                item.text = text
                break

    def search(self, query_vector: list[float], top_k: int, filters: dict | None = None) -> list[RagHit]:
        _ = filters
        scored: list[tuple[str, float, str]] = []
        for item in self._vectors:
            score = _cosine_similarity(query_vector, item.vector)
            scored.append((item.chunk_id, score, item.text))
        ranked = sorted(scored, key=lambda row: row[1], reverse=True)[:top_k]
        return [RagHit(chunk_id=chunk_id, score=score, text=text) for chunk_id, score, text in ranked]


class RagIndexer:
    def __init__(self, embedding_service: EmbeddingService, vector_store: InMemoryVectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_chunks(self, chunks: list[tuple[str, str]], model_id: str) -> None:
        records: list[EmbeddingRecord] = []
        for index, (chunk_id, text) in enumerate(chunks):
            vector = self._embedding_service.embed_text(text)
            records.append(
                EmbeddingRecord(
                    embedding_id=f"{model_id}-{index}",
                    chunk_id=chunk_id,
                    model_id=model_id,
                    vector=vector,
                    dimension=len(vector),
                )
            )
        self._vector_store.upsert_embeddings(records)
        for chunk_id, text in chunks:
            self._vector_store.attach_text(chunk_id, text)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = sqrt(sum(value * value for value in left[:size]))
    right_norm = sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
