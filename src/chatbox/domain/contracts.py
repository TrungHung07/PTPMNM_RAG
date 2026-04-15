from __future__ import annotations

from pydantic import BaseModel, Field

from chatbox.domain.models import RetrievalContext


class IngestRequest(BaseModel):
    source_uri: str
    file_type: str
    title: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    ingest_status: str


class QueryRequest(BaseModel):
    query_text: str
    mode: str = "hybrid"
    top_k: int = 5
    stream: bool = False


class QueryContextResponse(BaseModel):
    query_id: str
    retrieval_context: RetrievalContext


class QueryAnswerResponse(BaseModel):
    query_id: str
    answer_text: str
    citations: list[dict[str, str]] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, str] = Field(default_factory=dict)
    retrieval_context: RetrievalContext


class GenerationRequest(BaseModel):
    query_id: str
    user_query: str
    retrieval_context: RetrievalContext
    response_style: str = "balanced"
    max_tokens: int = 512
    temperature: float = 0.2


class GenerationResponse(BaseModel):
    response_id: str
    query_id: str
    answer_text: str
    citations: list[dict[str, str]] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, str] = Field(default_factory=dict)
    latency_ms: int
