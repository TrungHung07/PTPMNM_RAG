from __future__ import annotations

from chatbox.rag.embeddings import EmbeddingService
from chatbox.rag.indexer import InMemoryVectorStore
from chatbox.storage.ports import RagHit


class RagRetriever:
    def __init__(self, embedding_service: EmbeddingService, vector_store: InMemoryVectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RagHit]:
        query_vector = self._embedding_service.embed_text(query_text)
        return self._vector_store.search(query_vector=query_vector, top_k=top_k)
