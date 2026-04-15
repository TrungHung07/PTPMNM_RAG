from __future__ import annotations

from dataclasses import dataclass

from chatbox.domain.models import GraphEdge, GraphNode
from chatbox.graphrag.entity_extractor import extract_entities
from chatbox.graphrag.relation_extractor import extract_relations
from chatbox.storage.ports import GraphHit


@dataclass(slots=True)
class _StoredPath:
    path_id: str
    entities: set[str]
    evidence_chunk_ids: list[str]


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._paths: list[_StoredPath] = []

    def upsert_graph(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        for node in nodes:
            self._nodes[node.node_id] = node
        for edge in edges:
            self._edges[edge.edge_id] = edge
            from_label = self._nodes.get(edge.from_node_id, GraphNode(node_id=edge.from_node_id, label=edge.from_node_id, node_type="entity")).label.lower()
            to_label = self._nodes.get(edge.to_node_id, GraphNode(node_id=edge.to_node_id, label=edge.to_node_id, node_type="entity")).label.lower()
            self._paths.append(
                _StoredPath(
                    path_id=edge.edge_id,
                    entities={from_label, to_label},
                    evidence_chunk_ids=edge.evidence_chunk_ids,
                )
            )

    def search_paths(self, query_entities: list[str], top_k: int, max_hops: int) -> list[GraphHit]:
        _ = max_hops
        query = {entity.lower() for entity in query_entities}
        scored: list[tuple[float, _StoredPath]] = []
        for path in self._paths:
            overlap = len(query.intersection(path.entities))
            if overlap == 0:
                continue
            score = overlap / max(1, len(query))
            scored.append((score, path))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            GraphHit(path_id=item.path_id, score=score, evidence_chunk_ids=item.evidence_chunk_ids)
            for score, item in ranked
        ]


class GraphBuilder:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self._graph_store = graph_store

    def build_from_chunks(self, chunks: list[tuple[str, str]]) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        for chunk_id, text in chunks:
            entities = extract_entities(text)
            for entity in entities:
                node_id = entity.lower()
                nodes[node_id] = GraphNode(node_id=node_id, label=entity, node_type="entity")

            if len(entities) == 1:
                entity = entities[0]
                edges.append(
                    GraphEdge(
                        edge_id=f"{chunk_id}-edge-self",
                        from_node_id=entity.lower(),
                        to_node_id=entity.lower(),
                        relation_type="mentions",
                        evidence_chunk_ids=[chunk_id],
                        weight=1.0,
                        confidence=0.7,
                    )
                )

            for relation_index, (source, target, relation) in enumerate(extract_relations(entities)):
                edge = GraphEdge(
                    edge_id=f"{chunk_id}-edge-{relation_index}",
                    from_node_id=source.lower(),
                    to_node_id=target.lower(),
                    relation_type=relation,
                    evidence_chunk_ids=[chunk_id],
                    weight=1.0,
                    confidence=0.8,
                )
                edges.append(edge)

        self._graph_store.upsert_graph(list(nodes.values()), edges)
        return list(nodes.values()), edges
