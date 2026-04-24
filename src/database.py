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
    """
    Trả về danh sách tất cả phiên chat, mỗi phiên kèm:
    - Danh sách tên file (file_names)
    - Số lượng file (file_count)
    - Số tin nhắn (message_count)
    """
    async with get_conn() as conn:
        rows = await conn.fetch("""
            SELECT
                s.session_id,
                s.created_at,
                COALESCE(
                    ARRAY_AGG(d.file_name ORDER BY d.uploaded_at ASC) FILTER (WHERE d.doc_id IS NOT NULL),
                    ARRAY[]::text[]
                ) AS file_names,
                COUNT(DISTINCT d.doc_id) AS file_count,
                COUNT(m.id)             AS message_count
            FROM sessions s
            LEFT JOIN documents d ON d.session_id = s.session_id
            LEFT JOIN messages  m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.created_at
            ORDER BY s.created_at DESC
        """)
        return [dict(r) for r in rows]


async def db_get_documents_by_session(session_id: str) -> list[dict]:
    """Trả về danh sách tất cả documents (files) thuộc một session."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT doc_id, file_name, file_type, uploaded_at
            FROM documents
            WHERE session_id = $1::uuid
            ORDER BY uploaded_at ASC
            """,
            session_id,
        )
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

async def db_append_message(
    session_id: str,
    question: str,
    answer: str,
    file_ids: list[str],
    search_mode: str = "unknown",
    citations: list[dict] | None = None,
) -> None:
    """
    Lưu một cặp hỏi-đáp vào DB kèm file_ids, search_mode và citations.

    Args:
        session_id: UUID của phiên chat.
        question: Câu hỏi của user.
        answer: Câu trả lời của AI.
        file_ids: Danh sách doc_id (uuid) đã dùng để trả lời.
        search_mode: Chiến lược retrieval: 'vector', 'hybrid',
                     'compare_vector', 'compare_hybrid', hoặc 'unknown'.
        citations: Danh sách citation nguồn. Mỗi phần tử là dict có
                   keys: content (str), metadata (dict), score (float|None).
    """
    import json
    uuid_list = [str(fid) for fid in file_ids]
    citations_json = json.dumps(citations or [], ensure_ascii=False)
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO messages (session_id, question, answer, file_ids, search_mode, citations)
            VALUES ($1::uuid, $2, $3, $4::uuid[], $5, $6::jsonb)
            """,
            session_id, question, answer, uuid_list, search_mode, citations_json,
        )


async def db_get_messages(session_id: str) -> list[dict]:
    import json
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT question, answer, created_at, file_ids, search_mode, citations
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            """,
            session_id,
        )
        result = []
        for r in rows:
            row = dict(r)
            # Normalize file_ids: asyncpg uuid[] → list[str]
            row["file_ids"] = [str(fid) for fid in (row["file_ids"] or [])]
            # Normalize citations: asyncpg trả về str hoặc list tùy driver version
            raw_cit = row.get("citations")
            if raw_cit is None:
                row["citations"] = []
            elif isinstance(raw_cit, str):
                row["citations"] = json.loads(raw_cit)
            else:
                # asyncpg jsonb → Python object trực tiếp
                row["citations"] = list(raw_cit) if raw_cit else []
            # Normalize search_mode: fallback nếu NULL (row cũ trước migration)
            row["search_mode"] = row.get("search_mode") or "unknown"
            result.append(row)
        return result


async def db_get_recent_messages(
    session_id: str,
    file_ids: list[str],
    limit: int = 5,
) -> list[dict]:
    """
    Lấy N tin nhắn gần nhất có overlap với file_ids đưa vào.

    Logic lọc (Intersection mode):
      messages.file_ids && $file_ids::uuid[]
      → giữ lại message có ít nhất 1 file trùng với request.
      → tự động bỏ qua message cũ có file_ids = '{}' (không overlap với bất kỳ uuid nào).

    Trade-off so với exact-match (file_ids = $file_ids):
      - Intersection cho phép tái sử dụng history khi một request gồm nhiều file A+B
        và các lượt trước chỉ dùng A hoặc B đơn lẻ.
      - Exact-match thành chặt hơn nhưng mất history khi file_ids thay đổi thứ tự.
    """
    uuid_list = [str(fid) for fid in file_ids]
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT question, answer FROM (
                SELECT question, answer, created_at
                FROM messages
                WHERE session_id = $1::uuid
                  AND file_ids && $2::uuid[]
                ORDER BY created_at DESC
                LIMIT $3
            ) sub
            ORDER BY created_at ASC
            """,
            session_id, uuid_list, limit,
        )
        return [dict(r) for r in rows]
