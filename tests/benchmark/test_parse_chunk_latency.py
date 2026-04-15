from time import perf_counter

from chatbox.chunking.chunker import chunk_text
from chatbox.ingestion.normalizer import normalize_text


def test_parse_chunk_latency_budget() -> None:
    source = "Paragraph one.\n\n" + " ".join(["token"] * 6000)

    start = perf_counter()
    normalized = normalize_text(source)
    chunks = chunk_text(normalized, document_id="doc-benchmark", max_tokens=256, overlap_tokens=32)
    elapsed = perf_counter() - start

    assert chunks
    assert elapsed < 1.0
