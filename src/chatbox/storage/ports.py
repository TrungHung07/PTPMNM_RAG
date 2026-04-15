from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from chatbox.domain.models import Chunk, Document, EmbeddingRecord, GraphEdge, GraphNode


@dataclass(slots=True)
class RagHit:
    chunk_id: str
    score: float
    text: str


@dataclass(slots=True)
class GraphHit:
    path_id: str
    score: float
    evidence_chunk_ids: list[str]


class VectorStorePort(Protocol):
    def upsert_embeddings(self, records: list[EmbeddingRecord]) -> None: ...

    def search(self, query_vector: list[float], top_k: int, filters: dict | None = None) -> list[RagHit]: ...


class GraphStorePort(Protocol):
    def upsert_graph(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None: ...

    def search_paths(self, query_entities: list[str], top_k: int, max_hops: int) -> list[GraphHit]: ...


class MetadataStorePort(Protocol):
    def save_document(self, document: Document) -> None: ...

    def save_chunks(self, chunks: list[Chunk]) -> None: ...

    def get_document(self, document_id: str) -> Document | None: ...
