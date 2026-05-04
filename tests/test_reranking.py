"""
Test cho module reranking (src/rag/rerank.py).

Kiểm tra:
  - rerank_documents() sắp xếp lại docs theo cross-encoder score
  - Hành vi khi rerank bị tắt (enabled=False)
  - Threshold lọc bỏ docs có điểm thấp
  - Truncation không làm mất document
  - Edge cases: empty docs, top_k=0
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.rag.rerank import rerank_documents, _truncate


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs() -> list[Document]:
    """5 Document mẫu để test reranking."""
    return [
        Document(page_content="Chạy bộ giúp tăng cường sức khỏe tim mạch.", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="Dinh dưỡng hợp lý giúp cải thiện thể lực.", metadata={"page": 2, "source": "test.pdf"}),
        Document(page_content="Giày chạy bộ cần có đế êm và nhẹ.", metadata={"page": 3, "source": "test.pdf"}),
        Document(page_content="Khởi động trước khi chạy giúp tránh chấn thương.", metadata={"page": 4, "source": "test.pdf"}),
        Document(page_content="Python là ngôn ngữ lập trình phổ biến.", metadata={"page": 5, "source": "test.pdf"}),
    ]


@pytest.fixture
def mock_reranker():
    """Mock CrossEncoder để không cần load model thật."""
    mock = MagicMock()
    # Score giảm dần: doc đầu tiên liên quan nhất
    mock.predict.return_value = [0.95, 0.80, 0.60, 0.40, 0.05]
    return mock


# ── Test _truncate ─────────────────────────────────────────────────────────────

class TestTruncate:
    """Kiểm tra hàm _truncate()."""

    def test_no_truncation_when_within_limit(self):
        """Text ngắn hơn max_chars không bị cắt."""
        text = "Hello world"
        assert _truncate(text, 100) == text

    def test_truncation_at_limit(self):
        """Text dài hơn max_chars bị cắt đúng vị trí."""
        text = "abcdefghij"
        assert _truncate(text, 5) == "abcde"

    def test_no_truncation_when_zero(self):
        """max_chars=0 trả về toàn bộ text."""
        text = "Hello world"
        assert _truncate(text, 0) == text

    def test_empty_text(self):
        """Text rỗng trả về rỗng."""
        assert _truncate("", 100) == ""


# ── Test rerank_documents ──────────────────────────────────────────────────────

class TestRerankDocuments:
    """Kiểm tra hàm rerank_documents()."""

    def test_disabled_returns_original_order(self, sample_docs):
        """Khi enabled=False, trả về docs theo thứ tự gốc (không rerank)."""
        docs, scores = rerank_documents(
            query="chạy bộ",
            docs=sample_docs,
            top_k=3,
            max_chars=500,
            enabled=False,
        )
        assert len(docs) == 3
        assert scores is None
        # Thứ tự gốc được giữ nguyên
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    @patch("src.rag.rerank.get_reranker")
    def test_enabled_reorders_by_score(self, mock_get, sample_docs, mock_reranker):
        """Khi enabled=True, docs được sắp xếp theo cross-encoder score giảm dần."""
        mock_get.return_value = mock_reranker

        docs, scores = rerank_documents(
            query="chạy bộ tim mạch",
            docs=sample_docs,
            top_k=3,
            max_chars=500,
            enabled=True,
        )
        assert len(docs) == 3
        assert scores is not None
        # Scores phải giảm dần
        assert scores == sorted(scores, reverse=True)

    @patch("src.rag.rerank.get_reranker")
    def test_top_k_limits_output(self, mock_get, sample_docs, mock_reranker):
        """Chỉ trả về tối đa top_k documents."""
        mock_get.return_value = mock_reranker

        docs, scores = rerank_documents(
            query="chạy bộ",
            docs=sample_docs,
            top_k=2,
            max_chars=500,
            enabled=True,
        )
        assert len(docs) == 2
        assert len(scores) == 2

    @patch("src.rag.rerank.get_reranker")
    def test_threshold_filters_low_scores(self, mock_get, sample_docs, mock_reranker):
        """Threshold lọc bỏ docs có score thấp hơn ngưỡng."""
        mock_get.return_value = mock_reranker

        docs, scores = rerank_documents(
            query="chạy bộ",
            docs=sample_docs,
            top_k=5,
            max_chars=500,
            threshold=0.5,
            enabled=True,
        )
        # Chỉ 3 docs có score >= 0.5 (0.95, 0.80, 0.60)
        assert len(docs) == 3
        assert all(s >= 0.5 for s in scores)

    @patch("src.rag.rerank.get_reranker")
    def test_metadata_preserved_after_rerank(self, mock_get, sample_docs, mock_reranker):
        """Metadata phải được giữ nguyên sau reranking."""
        mock_get.return_value = mock_reranker

        docs, _ = rerank_documents(
            query="chạy bộ",
            docs=sample_docs,
            top_k=3,
            max_chars=500,
            enabled=True,
        )
        for doc in docs:
            assert "page" in doc.metadata
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "test.pdf"

    def test_empty_docs_returns_empty(self):
        """Danh sách docs rỗng trả về rỗng."""
        docs, scores = rerank_documents(
            query="test",
            docs=[],
            top_k=3,
            max_chars=500,
            enabled=True,
        )
        assert docs == []
        assert scores == []

    def test_top_k_zero_returns_empty(self, sample_docs):
        """top_k=0 trả về rỗng."""
        docs, scores = rerank_documents(
            query="test",
            docs=sample_docs,
            top_k=0,
            max_chars=500,
            enabled=True,
        )
        assert docs == []

    @patch("src.rag.rerank.get_reranker")
    def test_truncation_does_not_lose_docs(self, mock_get, sample_docs, mock_reranker):
        """Truncation chỉ cắt content cho model, không mất document."""
        mock_get.return_value = mock_reranker

        docs, scores = rerank_documents(
            query="chạy bộ",
            docs=sample_docs,
            top_k=5,
            max_chars=10,  # Rất ngắn, nhưng không mất doc
            enabled=True,
        )
        # Model vẫn nhận tất cả 5 docs (dù bị truncate)
        assert mock_reranker.predict.called
        pairs = mock_reranker.predict.call_args[0][0]
        assert len(pairs) == 5

    @patch("src.rag.rerank.get_reranker")
    def test_scores_are_float(self, mock_get, sample_docs, mock_reranker):
        """Scores trả về phải là list[float]."""
        mock_get.return_value = mock_reranker

        _, scores = rerank_documents(
            query="test",
            docs=sample_docs,
            top_k=3,
            max_chars=500,
            enabled=True,
        )
        assert all(isinstance(s, float) for s in scores)
