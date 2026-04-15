from pathlib import Path

from chatbox.chunking.chunker import chunk_text
from chatbox.ingestion.coordinator import IngestionCoordinator
from chatbox.ingestion.parsers.docx_parser import ParsedText
from chatbox.storage.sqlite_metadata import SqliteMetadataStore


class _StubParser:
    def __init__(self, file_type: str) -> None:
        self.file_type = file_type

    def parse(self, file_path: Path) -> ParsedText:
        return ParsedText(
            file_type=self.file_type,
            text=f"content from {file_path.name}",
            metadata={"source": file_path.name},
        )


def test_ingest_pipeline_persists_document_and_chunks(tmp_path: Path) -> None:
    metadata_store = SqliteMetadataStore(tmp_path / "chatbox.db")
    coordinator = IngestionCoordinator(
        metadata_store=metadata_store,
        parsers={"pdf": _StubParser("pdf"), "docx": _StubParser("docx")},
    )

    response = coordinator.ingest(file_path=tmp_path / "sample.pdf", file_type="pdf")
    saved = metadata_store.get_document(response.document_id)

    assert response.chunk_count >= 1
    assert saved is not None
    assert saved.file_type == "pdf"
    assert len(chunk_text(saved.raw_text, saved.document_id)) >= 1
