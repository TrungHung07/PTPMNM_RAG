"""
Test: file_ids filtering cho conversational history.
Chay: venv\\Scripts\\python.exe test_history_file_ids.py
"""
import asyncio
import uuid
import asyncpg
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_URL = "postgresql://raguser:ragpass@localhost:5432/ragdb"


async def fetch_history(conn, session_id, file_ids):
    return await conn.fetch(
        """
        SELECT question FROM (
            SELECT question, created_at
            FROM messages
            WHERE session_id = $1::uuid
              AND file_ids && $2::uuid[]
            ORDER BY created_at DESC
            LIMIT 10
        ) sub ORDER BY created_at ASC
        """,
        session_id,
        file_ids,
    )


async def run():
    conn = await asyncpg.connect(DB_URL)

    session_id = str(uuid.uuid4())
    file_a = str(uuid.uuid4())
    file_b = str(uuid.uuid4())
    file_c = str(uuid.uuid4())

    print(f"Session: {session_id}")
    print(f"  File A: {file_a}")
    print(f"  File B: {file_b}")
    print(f"  File C: {file_c}")
    print()

    # Tao session
    await conn.execute(
        "INSERT INTO sessions (session_id) VALUES ($1::uuid)",
        session_id,
    )

    # Msg 1: chi file A
    await conn.execute(
        "INSERT INTO messages (session_id, question, answer, file_ids)"
        " VALUES ($1::uuid, $2, $3, $4::uuid[])",
        session_id, "Q about A", "Ans A", [file_a],
    )
    # Msg 2: chi file B
    await conn.execute(
        "INSERT INTO messages (session_id, question, answer, file_ids)"
        " VALUES ($1::uuid, $2, $3, $4::uuid[])",
        session_id, "Q about B", "Ans B", [file_b],
    )
    # Msg 3: ca A+B
    await conn.execute(
        "INSERT INTO messages (session_id, question, answer, file_ids)"
        " VALUES ($1::uuid, $2, $3, $4::uuid[])",
        session_id, "Q about A+B", "Ans A+B", [file_a, file_b],
    )
    # Msg 4: message cu khong co file_ids (back-compat, file_ids = default {})
    await conn.execute(
        "INSERT INTO messages (session_id, question, answer)"
        " VALUES ($1::uuid, $2, $3)",
        session_id, "Old message", "Old ans",
    )

    # ------ TEST 1: query [B] -> chi Msg2 va Msg3 ------
    r1 = await fetch_history(conn, session_id, [file_b])
    q1 = [r["question"] for r in r1]
    print("TEST 1 - Query file_ids=[B]:")
    for q in q1:
        print(f"  -> {q!r}")
    assert "Q about A" not in q1, "BUG: history file A dinh vao query file B!"
    assert "Old message" not in q1, "BUG: message cu khong nen xuat hien!"
    assert "Q about B" in q1, "ERR: Thieu message cua B!"
    assert "Q about A+B" in q1, "ERR: Thieu message A+B (overlap)!"
    assert len(q1) == 2, f"Expected 2, got {len(q1)}"
    print("  => PASSED\n")

    # ------ TEST 2: query [A] -> chi Msg1 va Msg3 ------
    r2 = await fetch_history(conn, session_id, [file_a])
    q2 = [r["question"] for r in r2]
    print("TEST 2 - Query file_ids=[A]:")
    for q in q2:
        print(f"  -> {q!r}")
    assert "Q about B" not in q2, "BUG: history file B dinh vao query file A!"
    assert "Old message" not in q2, "BUG: message cu khong nen xuat hien!"
    assert len(q2) == 2, f"Expected 2, got {len(q2)}"
    print("  => PASSED\n")

    # ------ TEST 3: query [C] -> empty list ------
    r3 = await fetch_history(conn, session_id, [file_c])
    q3 = [r["question"] for r in r3]
    print("TEST 3 - Query file_ids=[C] (chua co message):")
    print(f"  -> count={len(q3)}")
    assert len(q3) == 0, f"Expected 0, got {len(q3)}"
    print("  => PASSED\n")

    # ------ TEST 4: query [] -> no crash, 0 rows ------
    r4 = await fetch_history(conn, session_id, [])
    print(f"TEST 4 - Query file_ids=[] (empty): count={len(r4)}")
    print("  => PASSED (no crash)\n")

    # ------ TEST 5: multi-file ask [A, B] -> lay ca 3 message ------
    r5 = await fetch_history(conn, session_id, [file_a, file_b])
    q5 = [r["question"] for r in r5]
    print("TEST 5 - Query file_ids=[A, B]:")
    for q in q5:
        print(f"  -> {q!r}")
    assert len(q5) == 3, f"Expected 3, got {len(q5)}"
    assert "Old message" not in q5
    print("  => PASSED\n")

    # Cleanup
    await conn.execute(
        "DELETE FROM sessions WHERE session_id = $1::uuid", session_id
    )
    await conn.close()
    print("=" * 50)
    print("Tat ca test PASSED")


asyncio.run(run())
