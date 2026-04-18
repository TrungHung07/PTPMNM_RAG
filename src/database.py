import os
import asyncpg
from contextlib import asynccontextmanager

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpass@localhost:5432/ragdb"
)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn():
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ── Session / documents (khớp db/migrations/001_init.sql) ─────

async def db_insert_session(session_id: str) -> None:
    """Tạo một dòng sessions — bắt buộc trước khi INSERT messages (FK)."""
    async with get_conn() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id) VALUES ($1::uuid) ON CONFLICT DO NOTHING",
            session_id,
        )


async def db_insert_document(
    doc_id: str, session_id: str, file_name: str, file_type: str
) -> None:
    """Một dòng documents = một file đã upload trong phiên."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (doc_id, session_id, file_name, file_type)
            VALUES ($1::uuid, $2::uuid, $3, $4)
            ON CONFLICT DO NOTHING
            """,
            doc_id,
            session_id,
            file_name,
            file_type,
        )


async def db_get_all_sessions() -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch("""
            SELECT
                s.session_id,
                s.created_at,
                (
                    SELECT d.file_name FROM documents d
                    WHERE d.session_id = s.session_id
                    ORDER BY d.uploaded_at ASC NULLS LAST
                    LIMIT 1
                ) AS filename,
                COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.created_at
            ORDER BY s.created_at DESC
        """)
        return [dict(r) for r in rows]


async def db_delete_session(session_id: str) -> bool:
    async with get_conn() as conn:
        result = await conn.execute(
            "DELETE FROM sessions WHERE session_id = $1::uuid", session_id
        )
        return result == "DELETE 1"


async def db_delete_all_sessions() -> None:
    async with get_conn() as conn:
        await conn.execute("DELETE FROM sessions")


# ── Message queries ───────────────────────────────────────────

async def db_append_message(session_id: str, question: str, answer: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, question, answer) VALUES ($1, $2, $3)",
            session_id, question, answer
        )


async def db_get_messages(session_id: str) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            "SELECT question, answer, created_at FROM messages WHERE session_id = $1 ORDER BY created_at ASC",
            session_id,
        )
        return [dict(r) for r in rows]


async def db_get_recent_messages(session_id: str, limit: int = 5) -> list[dict]:
    """Lấy N tin nhắn gần nhất - dùng cho Conversational RAG context."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT question, answer FROM (
                SELECT question, answer, created_at
                FROM messages WHERE session_id = $1
                ORDER BY created_at DESC LIMIT $2
            ) sub ORDER BY created_at ASC
            """,
            session_id, limit
        )
        return [dict(r) for r in rows]
