from chatbox.chunking.chunker import chunk_text


def test_chunker_respects_max_tokens_and_overlap() -> None:
    text = " ".join(f"tok{i}" for i in range(25))
    chunks = chunk_text(text, document_id="doc-1", max_tokens=10, overlap_tokens=2)

    assert len(chunks) == 3
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert chunks[0].chunk_order == 0
    assert chunks[1].chunk_order == 1


def test_chunker_is_deterministic() -> None:
    text = "alpha beta gamma delta"
    first = chunk_text(text, document_id="doc-1", max_tokens=3, overlap_tokens=1)
    second = chunk_text(text, document_id="doc-1", max_tokens=3, overlap_tokens=1)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.checksum for chunk in first] == [chunk.checksum for chunk in second]
