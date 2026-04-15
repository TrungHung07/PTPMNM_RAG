from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chatbox.observability.logging import configure_logging
from chatbox.observability.metrics import InMemoryMetrics
from chatbox.observability.tracing import TraceFactory
from chatbox.storage.sqlite_metadata import SqliteMetadataStore


@dataclass(slots=True)
class AppContainer:
    metadata_store: SqliteMetadataStore
    metrics: InMemoryMetrics
    trace_factory: TraceFactory


def create_app_container(sqlite_path: str | Path) -> AppContainer:
    """Create runtime dependencies for local execution."""
    configure_logging()
    metadata_store = SqliteMetadataStore(Path(sqlite_path))
    return AppContainer(
        metadata_store=metadata_store,
        metrics=InMemoryMetrics(),
        trace_factory=TraceFactory(),
    )
