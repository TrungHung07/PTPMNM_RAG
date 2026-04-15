from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from chatbox.domain.contracts import IngestRequest, IngestResponse
from chatbox.ingestion.coordinator import IngestionCoordinator


def create_documents_router(coordinator: IngestionCoordinator) -> APIRouter:
    router = APIRouter(prefix="/v1/documents", tags=["documents"])

    @router.post(":ingest", response_model=IngestResponse)
    def ingest_document(request: IngestRequest) -> IngestResponse:
        return coordinator.ingest(file_path=Path(request.source_uri), file_type=request.file_type)

    return router
