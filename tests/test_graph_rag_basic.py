"""
tests/test_graph_rag_basic.py — Unit tests cho Graph RAG module.

Dùng Mock LLM để tránh phụ thuộc Ollama/network.
Chạy:  venv\\Scripts\\python.exe -m pytest tests/test_graph_rag_basic.py -v
"""
import sys
import io
import json
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    Document(
        page_content="Hệ thống Smart Task Manager cho phép tạo dự án và giao việc cho nhân viên.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_index": 0},
    ),
    Document(
        page_content="Smart Task Manager tích hợp AI sử dụng Ollama hoặc GPT API để gợi ý người thực hiện.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_index": 1},
    ),
    Document(
        page_content="Hệ thống Asset Tracking theo dõi vòng đời tài sản bằng mã QR Code.",
        metadata={"source": "doc.pdf", "page": 2, "chunk_index": 0},
    ),
]

FIXED_TRIPLES_JSON = json.dumps({
    "triples": [
        {"s": "Smart Task Manager", "r": "cho phép", "o": "tạo dự án"},
        {"s": "Smart Task Manager", "r": "tích hợp", "o": "AI"},
        {"s": "Asset Tracking", "r": "theo dõi", "o": "vòng đời tài sản"},
    ]
})

FIXED_ENTITIES_JSON = json.dumps({"entities": ["Smart Task Manager"]})


def make_mock_llm(extract_response: str = FIXED_TRIPLES_JSON,
                  entity_response: str = FIXED_ENTITIES_JSON):
    """Tạo Mock LLM trả về JSON cố định cho mọi invoke()."""
    mock = MagicMock()
    # Luân phiên: lần đầu extract_response, lần sau entity_response
    mock.invoke.side_effect = lambda prompt: (
        entity_response if '"entities"' in prompt else extract_response
    )
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildGraphIndex:
    """Test build_graph_index không crash và tạo đúng structure."""

    def test_build_creates_triples(self):
        from src.graph_rag.indexer import build_graph_index

        llm = make_mock_llm()
        index = build_graph_index(SAMPLE_CHUNKS, llm=llm)

        assert len(index.triples) > 0, "Phải có ít nhất 1 triple"
        print(f"  triples count = {len(index.triples)}")

    def test_build_creates_entity_index(self):
        from src.graph_rag.indexer import build_graph_index

        llm = make_mock_llm()
        index = build_graph_index(SAMPLE_CHUNKS, llm=llm)

        assert len(index.entity_to_triples) > 0, "entity_to_triples phải không rỗng"
        print(f"  unique entities = {len(index.entity_to_triples)}")

    def test_build_stores_evidence(self):
        from src.graph_rag.indexer import build_graph_index

        llm = make_mock_llm()
        index = build_graph_index(SAMPLE_CHUNKS, llm=llm)

        for triple in index.triples:
            ev_list = index.evidence_by_triple.get(triple, [])
            assert len(ev_list) > 0, f"Triple {triple} phải có evidence"
            assert "content" in ev_list[0], "evidence phải có key 'content'"
            assert "metadata" in ev_list[0], "evidence phải có key 'metadata'"

    def test_build_empty_chunks_no_crash(self):
        from src.graph_rag.indexer import build_graph_index

        llm = make_mock_llm()
        index = build_graph_index([], llm=llm)
        assert index.triples == [], "Không có chunk → triples rỗng"


class TestGraphRetrieve:
    """Test graph_retrieve trả về docs đúng và citations có shape tốt."""

    def _build_index(self):
        from src.graph_rag.indexer import build_graph_index
        llm = make_mock_llm()
        return build_graph_index(SAMPLE_CHUNKS, llm=llm)

    def test_retrieve_returns_docs(self):
        from src.graph_rag.retriever import graph_retrieve

        index = self._build_index()
        docs, citations = graph_retrieve(
            graph_index=index,
            question="Smart Task Manager làm gì?",
            entities=["Smart Task Manager"],
            top_k=5,
        )
        assert len(docs) > 0, "Phải trả về ít nhất 1 doc"
        print(f"  docs returned = {len(docs)}")

    def test_retrieve_citations_shape(self):
        from src.graph_rag.retriever import graph_retrieve

        index = self._build_index()
        _, citations = graph_retrieve(
            graph_index=index,
            question="Smart Task Manager làm gì?",
            entities=["Smart Task Manager"],
        )

        assert len(citations) > 0, "Phải có citations"
        for cit in citations:
            assert "content" in cit, "citation phải có key 'content'"
            assert "metadata" in cit, "citation phải có key 'metadata'"
            assert "score" in cit, "citation phải có key 'score'"
        print(f"  citations count = {len(citations)}, shape OK")

    def test_retrieve_no_entity_fallback(self):
        """Khi không tìm được entity, fallback keyword search không crash."""
        from src.graph_rag.retriever import graph_retrieve

        index = self._build_index()
        docs, _ = graph_retrieve(
            graph_index=index,
            question="QR Code theo dõi tài sản",
            entities=[],  # Không có entity
        )
        # Có thể rỗng hoặc có docs từ keyword fallback — không crash là đủ
        assert isinstance(docs, list), "Phải trả về list"

    def test_retrieve_unknown_entity_empty(self):
        from src.graph_rag.retriever import graph_retrieve

        index = self._build_index()
        docs, _ = graph_retrieve(
            graph_index=index,
            question="Blockchain cryptocurrency NFT",
            entities=["blockchain_xyz_unknown_12345"],
        )
        # Không crash, trả về rỗng hoặc minimal
        assert isinstance(docs, list)


class TestRunGraphRag:
    """Test run_graph_rag trả về SearchResult hợp lệ."""

    def _build_index(self):
        from src.graph_rag.indexer import build_graph_index
        llm = make_mock_llm()
        return build_graph_index(SAMPLE_CHUNKS, llm=llm)

    def test_run_returns_search_result(self):
        from src.graph_rag.pipeline import run_graph_rag
        from src.models import SearchResult

        index = self._build_index()

        # Mock LLM cho pipeline: trả entity rồi trả answer
        pipeline_llm = MagicMock()
        call_count = [0]
        def side_effect(prompt):
            call_count[0] += 1
            if '"entities"' in prompt:
                return FIXED_ENTITIES_JSON
            return "Smart Task Manager cho phép tạo dự án và tích hợp AI."
        pipeline_llm.invoke.side_effect = side_effect

        result = run_graph_rag(
            graph_index=index,
            question="Smart Task Manager làm gì?",
            chat_history=[],
            llm=pipeline_llm,
        )

        assert isinstance(result, SearchResult), "Phải trả về SearchResult"
        assert result.answer, "answer không được rỗng"
        print(f"  answer[:60] = {result.answer[:60]!r}")

    def test_run_no_info_when_empty_index(self):
        from src.graph_rag.pipeline import run_graph_rag
        from src.graph_rag.types import GraphRAGIndex

        empty_index = GraphRAGIndex()
        llm = make_mock_llm(entity_response=json.dumps({"entities": ["unknown_xyz"]}))

        result = run_graph_rag(
            graph_index=empty_index,
            question="Câu hỏi về chủ đề không có trong graph",
            chat_history=[],
            llm=llm,
            no_info_phrase="Không tìm thấy thông tin trong tài liệu.",
        )

        assert "Không tìm thấy" in result.answer, "Phải báo không có thông tin"
        assert result.citations == [], "citations phải rỗng khi không có thông tin"
        print(f"  no_info answer = {result.answer!r}")


class TestMergeGraphIndices:
    """Test merge_graph_indices gộp đúng union triples."""

    def test_merge_two_indices(self):
        from src.graph_rag.indexer import build_graph_index, merge_graph_indices

        llm = make_mock_llm()
        idx1 = build_graph_index(SAMPLE_CHUNKS[:1], llm=llm)
        idx2 = build_graph_index(SAMPLE_CHUNKS[2:], llm=llm)

        merged = merge_graph_indices([idx1, idx2])
        # Merged phải có ≥ max(len(idx1.triples), len(idx2.triples)) triples
        assert len(merged.triples) >= max(len(idx1.triples), len(idx2.triples)), \
            "Merged phải ≥ mỗi index riêng lẻ"
        print(f"  merged triples = {len(merged.triples)}")

    def test_merge_single_index(self):
        from src.graph_rag.indexer import build_graph_index, merge_graph_indices

        llm = make_mock_llm()
        idx = build_graph_index(SAMPLE_CHUNKS, llm=llm)
        merged = merge_graph_indices([idx])
        assert merged is idx, "Merge 1 index → trả về chính nó"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestBuildGraphIndex,
        TestGraphRetrieve,
        TestRunGraphRag,
        TestMergeGraphIndices,
    ]

    total = 0
    passed = 0
    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for method in methods:
            total += 1
            try:
                getattr(obj, method)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method}")
            except Exception:
                print(f"  ✗ {cls.__name__}.{method}")
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Kết quả: {passed}/{total} PASSED")
    if passed < total:
        sys.exit(1)
