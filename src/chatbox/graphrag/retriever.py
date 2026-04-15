from __future__ import annotations

from chatbox.graphrag.builder import InMemoryGraphStore
from chatbox.graphrag.entity_extractor import extract_entities
from chatbox.storage.ports import GraphHit


class GraphRetriever:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self._graph_store = graph_store

    def retrieve(self, query_text: str, top_k: int = 5, max_hops: int = 2) -> list[GraphHit]:
        entities = extract_entities(query_text)
        if not entities:
            entities = [token.lower() for token in query_text.split() if token]
        return self._graph_store.search_paths(query_entities=entities, top_k=top_k, max_hops=max_hops)
