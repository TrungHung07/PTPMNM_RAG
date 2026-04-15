from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    document_id: str
    source_uri: str
    file_type: str
    title: str | None = None
    language: str | None = None
    raw_text: str
    sections: list[dict[str, Any]] = Field(default_factory=list)
    ingest_status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("file_type")
    @classmethod
    def _validate_file_type(cls, value: str) -> str:
        allowed = {"pdf", "docx"}
        if value not in allowed:
            raise ValueError(f"unsupported file_type {value}")
        return value


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_order: int
    text: str
    token_count: int
    overlap_tokens: int = 0
    boundary_type: str = "paragraph"
    source_pointer: dict[str, Any] = Field(default_factory=dict)
    checksum: str

    @field_validator("chunk_order", "token_count", "overlap_tokens")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value


class EmbeddingRecord(BaseModel):
    embedding_id: str
    chunk_id: str
    model_id: str
    vector: list[float]
    dimension: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GraphNode(BaseModel):
    node_id: str
    label: str
    node_type: str
    aliases: list[str] = Field(default_factory=list)
    document_refs: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class GraphEdge(BaseModel):
    edge_id: str
    from_node_id: str
    to_node_id: str
    relation_type: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    weight: float = 1.0
    confidence: float = 1.0


class RetrievalContext(BaseModel):
    query_id: str
    query_text: str
    rag_hits: list[dict[str, Any]] = Field(default_factory=list)
    graph_hits: list[dict[str, Any]] = Field(default_factory=list)
    merged_evidence: list[dict[str, Any]] = Field(default_factory=list)
    degraded_mode: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
