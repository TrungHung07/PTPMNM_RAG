from datetime import UTC, datetime

import pytest

from chatbox.domain.contracts import GenerationRequest, GenerationResponse
from chatbox.domain.models import Chunk, Document, RetrievalContext


def test_document_model_accepts_supported_type() -> None:
    document = Document(
        document_id="doc-1",
        source_uri="fixtures/sample.pdf",
        file_type="pdf",
        raw_text="hello",
    )
    assert document.file_type == "pdf"


def test_document_model_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError):
        Document(
            document_id="doc-1",
            source_uri="fixtures/sample.txt",
            file_type="txt",
            raw_text="hello",
        )


def test_chunk_model_non_negative_fields() -> None:
    chunk = Chunk(
        chunk_id="c-1",
        document_id="doc-1",
        chunk_order=0,
        text="hello",
        token_count=5,
        checksum="abc",
    )
    assert chunk.token_count == 5


def test_generation_contracts_round_trip() -> None:
    ctx = RetrievalContext(query_id="q-1", query_text="what")
    request = GenerationRequest(query_id="q-1", user_query="what", retrieval_context=ctx)
    response = GenerationResponse(
        response_id="r-1",
        query_id=request.query_id,
        answer_text="answer",
        citations=[{"chunk_id": "c-1", "document_id": "doc-1"}],
        latency_ms=120,
    )
    assert response.query_id == request.query_id
    assert response.citations[0]["chunk_id"] == "c-1"


def test_datetime_fields_are_serializable() -> None:
    now = datetime.now(UTC)
    document = Document(
        document_id="doc-2",
        source_uri="fixtures/sample.docx",
        file_type="docx",
        raw_text="hello",
        created_at=now,
        updated_at=now,
    )
    dumped = document.model_dump(mode="json")
    assert isinstance(dumped["created_at"], str)
