"""
Test: search_mode và citations lưu đúng vào DB; /compare lưu 2 messages.

Chay:
    venv\\Scripts\\python.exe tests/test_history_search_mode_citations.py

Yêu cầu: DB đang chạy (docker compose up postgres -d) + migration 003 đã apply.
"""
import asyncio
import json
import uuid
import asyncpg
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_URL = "postgresql://raguser:ragpass@localhost:5432/ragdb"

# Fixture citations mẫu (cùng shape với CitationSource)
SAMPLE_CITATIONS = [
    {"content": "Đoạn văn 1 về chủ đề X.", "metadata": {"source": "doc.pdf", "page": 1}, "score": 0.92},
    {"content": "Đoạn văn 2 về chủ đề Y.", "metadata": {"source": "doc.pdf", "page": 2}, "score": 0.75},
]


async def run():
    conn = await asyncpg.connect(DB_URL)

    session_id = str(uuid.uuid4())
    file_a = str(uuid.uuid4())

    print(f"Session : {session_id}")
    print(f"File A  : {file_a}")
    print()

    # Tạo session
    await conn.execute(
        "INSERT INTO sessions (session_id) VALUES ($1::uuid)",
        session_id,
    )

    # ── Helper insert dùng cùng logic với db_append_message ──────────────────
    async def insert_msg(question, answer, file_ids, search_mode, citations):
        citations_json = json.dumps(citations, ensure_ascii=False)
        await conn.execute(
            """
            INSERT INTO messages (session_id, question, answer, file_ids, search_mode, citations)
            VALUES ($1::uuid, $2, $3, $4::uuid[], $5, $6::jsonb)
            """,
            session_id, question, answer, [str(f) for f in file_ids],
            search_mode, citations_json,
        )

    # ── Insert 3 messages: /ask vector, /ask hybrid, /compare (2 rows) ───────
    await insert_msg("Câu hỏi vector", "Trả lời vector", [file_a], "vector", SAMPLE_CITATIONS)
    await insert_msg("Câu hỏi hybrid", "Trả lời hybrid", [file_a], "hybrid", [SAMPLE_CITATIONS[0]])
    await insert_msg("Câu hỏi compare", "Trả lời compare_vector", [file_a], "compare_vector", SAMPLE_CITATIONS)
    await insert_msg("Câu hỏi compare", "Trả lời compare_hybrid", [file_a], "compare_hybrid", SAMPLE_CITATIONS)

    # ── Đọc lại toàn bộ ──────────────────────────────────────────────────────
    rows = await conn.fetch(
        """
        SELECT question, answer, search_mode, citations, file_ids
        FROM messages
        WHERE session_id = $1
        ORDER BY created_at ASC
        """,
        session_id,
    )
    msgs = []
    for r in rows:
        row = dict(r)
        raw = row["citations"]
        row["citations"] = json.loads(raw) if isinstance(raw, str) else list(raw or [])
        row["file_ids"] = [str(fid) for fid in (row["file_ids"] or [])]
        msgs.append(row)

    # ── TEST 1: Số lượng messages ─────────────────────────────────────────────
    print("TEST 1 - Số lượng messages (4 expected):")
    print(f"  count = {len(msgs)}")
    assert len(msgs) == 4, f"Expected 4 messages, got {len(msgs)}"
    print("  => PASSED\n")

    # ── TEST 2: search_mode được lưu đúng ────────────────────────────────────
    print("TEST 2 - search_mode lưu đúng:")
    modes = [m["search_mode"] for m in msgs]
    print(f"  modes = {modes}")
    assert modes == ["vector", "hybrid", "compare_vector", "compare_hybrid"], \
        f"search_mode không khớp: {modes}"
    print("  => PASSED\n")

    # ── TEST 3: citations lưu và đọc đúng shape ───────────────────────────────
    print("TEST 3 - citations lưu và đọc đúng:")
    cit0 = msgs[0]["citations"]
    print(f"  citations[0] count = {len(cit0)}")
    assert len(cit0) == 2, f"Expected 2 citations, got {len(cit0)}"
    assert cit0[0]["content"] == SAMPLE_CITATIONS[0]["content"], "content không khớp"
    assert cit0[0]["score"] == SAMPLE_CITATIONS[0]["score"], "score không khớp"
    assert cit0[0]["metadata"]["page"] == 1, "metadata.page không khớp"
    print("  => PASSED\n")

    # ── TEST 4: citations rỗng list ───────────────────────────────────────────
    print("TEST 4 - citations = [] khi không có nguồn:")
    # Insert thêm 1 message không có citations
    await insert_msg("No citation Q", "No citation A", [file_a], "vector", [])
    rows2 = await conn.fetch(
        "SELECT citations FROM messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1",
        session_id,
    )
    raw2 = rows2[0]["citations"]
    cit_empty = json.loads(raw2) if isinstance(raw2, str) else list(raw2 or [])
    print(f"  citations = {cit_empty}")
    assert cit_empty == [], f"Expected [], got {cit_empty}"
    print("  => PASSED\n")

    # ── TEST 5: /compare tạo đúng 2 messages cùng câu hỏi ────────────────────
    print("TEST 5 - /compare lưu 2 messages riêng biệt:")
    compare_msgs = [m for m in msgs if m["search_mode"].startswith("compare_")]
    print(f"  compare messages count = {len(compare_msgs)}")
    assert len(compare_msgs) == 2, f"Expected 2, got {len(compare_msgs)}"
    assert compare_msgs[0]["search_mode"] == "compare_vector"
    assert compare_msgs[1]["search_mode"] == "compare_hybrid"
    assert compare_msgs[0]["question"] == compare_msgs[1]["question"], \
        "2 compare messages phải có cùng câu hỏi"
    print("  => PASSED\n")

    # ── TEST 6: back-compat — message cũ không có search_mode / citations ────
    print("TEST 6 - Back-compat: message cũ (không có search_mode/citations):")
    await conn.execute(
        "INSERT INTO messages (session_id, question, answer, file_ids)"
        " VALUES ($1::uuid, $2, $3, $4::uuid[])",
        session_id, "Old Q", "Old A", [file_a],
    )
    old_row = await conn.fetchrow(
        "SELECT search_mode, citations FROM messages WHERE session_id = $1 AND question = 'Old Q'",
        session_id,
    )
    print(f"  search_mode = {old_row['search_mode']!r}  (expected 'unknown')")
    raw_old = old_row["citations"]
    cit_old = json.loads(raw_old) if isinstance(raw_old, str) else list(raw_old or [])
    print(f"  citations   = {cit_old!r}        (expected [])")
    assert old_row["search_mode"] == "unknown", "DEFAULT search_mode phải là 'unknown'"
    assert cit_old == [], "DEFAULT citations phải là []"
    print("  => PASSED\n")

    # Cleanup
    await conn.execute("DELETE FROM sessions WHERE session_id = $1::uuid", session_id)
    await conn.close()
    print("=" * 55)
    print("Tất cả test PASSED ✓")


asyncio.run(run())
