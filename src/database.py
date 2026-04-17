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


# ── Session queries ───────────────────────────────────────────

async def db_create_session(file_id: str, filename: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "INSERT INTO sessions (file_id, filename) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            file_id, filename
        )


async def db_get_all_sessions() -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch("""
            SELECT s.file_id, s.filename, s.created_at, COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON s.file_id = m.file_id
            GROUP BY s.file_id, s.filename, s.created_at
            ORDER BY s.created_at DESC
        """)
        return [dict(r) for r in rows]


async def db_delete_session(file_id: str) -> bool:
    async with get_conn() as conn:
        result = await conn.execute("DELETE FROM sessions WHERE file_id = $1", file_id)
        return result == "DELETE 1"


async def db_delete_all_sessions() -> None:
    async with get_conn() as conn:
        await conn.execute("DELETE FROM sessions")


# ── Message queries ───────────────────────────────────────────

async def db_append_message(file_id: str, question: str, answer: str) -> None:
    async with get_conn() as conn:
        await conn.execute(
            "INSERT INTO messages (file_id, question, answer) VALUES ($1, $2, $3)",
            file_id, question, answer
        )


async def db_get_messages(file_id: str) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            "SELECT question, answer, created_at FROM messages WHERE file_id = $1 ORDER BY created_at ASC",
            file_id
        )
        return [dict(r) for r in rows]


async def db_get_recent_messages(file_id: str, limit: int = 5) -> list[dict]:
    """Lấy N tin nhắn gần nhất - dùng cho Conversational RAG context."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT question, answer FROM (
                SELECT question, answer, created_at
                FROM messages WHERE file_id = $1
                ORDER BY created_at DESC LIMIT $2
            ) sub ORDER BY created_at ASC
            """,
            file_id, limit
        )
        return [dict(r) for r in rows]
