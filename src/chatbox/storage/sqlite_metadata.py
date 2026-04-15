from __future__ import annotations

import sqlite3
from pathlib import Path

from chatbox.domain.models import Chunk, Document


class SqliteMetadataStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    title TEXT,
                    language TEXT,
                    raw_text TEXT NOT NULL,
                    ingest_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    overlap_tokens INTEGER NOT NULL,
                    boundary_type TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
                """
            )

    def save_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, source_uri, file_type, title, language,
                    raw_text, ingest_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.source_uri,
                    document.file_type,
                    document.title,
                    document.language,
                    document.raw_text,
                    document.ingest_status,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                ),
            )

    def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, chunk_order, text,
                    token_count, overlap_tokens, boundary_type, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_order,
                        chunk.text,
                        chunk.token_count,
                        chunk.overlap_tokens,
                        chunk.boundary_type,
                        chunk.checksum,
                    )
                    for chunk in chunks
                ],
            )

    def get_document(self, document_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, source_uri, file_type, title, language,
                       raw_text, ingest_status, created_at, updated_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(
            document_id=row[0],
            source_uri=row[1],
            file_type=row[2],
            title=row[3],
            language=row[4],
            raw_text=row[5],
            ingest_status=row[6],
            created_at=row[7],
            updated_at=row[8],
        )
