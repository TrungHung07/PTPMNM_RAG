"""
src/graph_rag/types.py — Data types cho Graph RAG module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Triple:
    """
    Một knowledge triple (Subject, Relation, Object) trích xuất từ chunk.

    Attributes:
        s: Subject — thực thể chủ thể.
        r: Relation — quan hệ / động từ kết nối S và O.
        o: Object — thực thể đối tượng.
    """
    s: str
    r: str
    o: str

    def __str__(self) -> str:
        return f"({self.s}) --[{self.r}]--> ({self.o})"

    def entities(self) -> tuple[str, str]:
        """Trả về (subject, object) để thêm vào graph."""
        return self.s, self.o


@dataclass
class GraphRAGIndex:
    """
    Container lưu kết quả indexing theo dạng knowledge graph.

    Attributes:
        triples: Danh sách tất cả triple đã trích xuất.
        evidence_by_triple: Mapping từ Triple → list của (content, metadata) gốc.
            Dùng để build citations khi retrieve.
        entity_to_triples: Mapping từ tên entity (lowercase) → list Triple liên quan.
            Dùng để lookup nhanh khi query.
    """
    triples: list[Triple] = field(default_factory=list)
    evidence_by_triple: dict[Triple, list[dict[str, Any]]] = field(default_factory=dict)
    entity_to_triples: dict[str, list[Triple]] = field(default_factory=dict)

    def add_triple(self, triple: Triple, content: str, metadata: dict[str, Any]) -> None:
        """Thêm triple + evidence gốc vào index."""
        if triple not in self.evidence_by_triple:
            self.triples.append(triple)
            self.evidence_by_triple[triple] = []

        self.evidence_by_triple[triple].append({"content": content, "metadata": metadata})

        # Index theo cả 2 entity
        for entity in (triple.s.lower(), triple.o.lower()):
            self.entity_to_triples.setdefault(entity, [])
            if triple not in self.entity_to_triples[entity]:
                self.entity_to_triples[entity].append(triple)
