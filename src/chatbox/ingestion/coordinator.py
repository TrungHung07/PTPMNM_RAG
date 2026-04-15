from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from chatbox.chunking.chunker import chunk_text
from chatbox.domain.contracts import IngestResponse
from chatbox.domain.errors import IngestionError
from chatbox.domain.models import Document
from chatbox.ingestion.normalizer import normalize_text
from chatbox.storage.ports import MetadataStorePort


class IngestionCoordinator:
    def __init__(self, metadata_store: MetadataStorePort, parsers: dict[str, object]) -> None:
        self._metadata_store = metadata_store
        self._parsers = parsers

    def ingest(self, file_path: Path, file_type: str) -> IngestResponse:
        parser = self._parsers.get(file_type)
        if parser is None:
            raise IngestionError(f"no parser configured for file_type={file_type}")

        parsed = parser.parse(file_path)
        normalized_text = normalize_text(parsed.text)
        document_id = f"doc-{uuid4()}"
        document = Document(
            document_id=document_id,
            source_uri=str(file_path),
            file_type=file_type,
            title=file_path.stem,
            raw_text=normalized_text,
            sections=[parsed.metadata],
            ingest_status="indexed",
        )
        chunks = chunk_text(normalized_text, document_id=document_id)

        self._metadata_store.save_document(document)
        self._metadata_store.save_chunks(chunks)

        return IngestResponse(document_id=document_id, chunk_count=len(chunks), ingest_status="indexed")
