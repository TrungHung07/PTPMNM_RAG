from __future__ import annotations

from pathlib import Path

import typer

from chatbox.app.bootstrap import create_app_container
from chatbox.ingestion.coordinator import IngestionCoordinator
from chatbox.ingestion.parsers.docx_parser import DocxParser
from chatbox.ingestion.parsers.pdf_parser import PdfParser

app = typer.Typer(help="Import PDF/DOCX documents into local metadata storage")


@app.command("run")
def run_import(file_path: Path, file_type: str, sqlite_path: Path = Path(".chatbox/chatbox.db")) -> None:
    container = create_app_container(sqlite_path)
    coordinator = IngestionCoordinator(
        metadata_store=container.metadata_store,
        parsers={"pdf": PdfParser(), "docx": DocxParser()},
    )
    response = coordinator.ingest(file_path=file_path, file_type=file_type)
    typer.echo(
        f"ingested document_id={response.document_id} chunks={response.chunk_count} status={response.ingest_status}"
    )
