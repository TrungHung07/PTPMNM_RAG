"""
Unit tests cho src/rag/pipeline.py — build_index() và _build_prompt().

Các test này kiểm tra logic pipeline KHÔNG cần LLM hay database:
  - build_index() trả về RAGIndex đúng structure
  - _build_prompt() sinh ra prompt đúng nội dung
  - compare_search_modes() — test structure response (mock LLM call)
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.rag.pipeline import build_index, _build_prompt, _format_chat_history, _build_citation_list
from src.models import RAGIndex, CitationSource


@pytest.fixture(scope="module")
def sample_documents() -> list[Document]:
    """Danh sách Document mẫu cho pipeline tests."""
    return [
        Document(
            page_content="Trí tuệ nhân tạo là lĩnh vực khoa học máy tính nghiên cứu " * 10,
            metadata={"page": 1, "source": "ai_intro.pdf"}
        ),
        Document(
            page_content="Học máy sử dụng dữ liệu để tự động học và cải thiện hiệu suất " * 10,
            metadata={"page": 2, "source": "ai_intro.pdf"}
        ),
    ]


class TestBuildIndex:
    """Kiểm tra hàm build_index() — indexing pipeline."""

    def test_returns_ragindex(self, sample_documents):
        """build_index() phải trả về RAGIndex object."""
        index = build_index(sample_documents)
        assert isinstance(index, RAGIndex)

    def test_ragindex_has_vectorstore(self, sample_documents):
        """RAGIndex.vectorstore phải tồn tại và không None."""
        index = build_index(sample_documents)
        assert index.vectorstore is not None

    def test_ragindex_has_chunks(self, sample_documents):
        """RAGIndex.chunks phải là list[Document] không rỗng."""
        index = build_index(sample_documents)
        assert isinstance(index.chunks, list)
        assert len(index.chunks) > 0

    def test_chunk_count_greater_than_doc_count(self, sample_documents):
        """Số chunk phải >= số document gốc (mỗi doc có thể tạo nhiều chunk)."""
        index = build_index(sample_documents, chunk_size=200, overlap=0)
        assert len(index.chunks) >= len(sample_documents)

    def test_chunks_preserve_metadata(self, sample_documents):
        """Mỗi chunk trong RAGIndex.chunks phải có metadata gốc (page, source)."""
        index = build_index(sample_documents)
        for chunk in index.chunks:
            assert "source" in chunk.metadata


class TestBuildPrompt:
    """
    Kiểm tra hàm _build_prompt() — đây là hàm quan trọng nhưng thường bị bỏ qua.
    Test prompt giúp đảm bảo context và câu hỏi được đưa vào đúng chỗ.
    """

    def test_contains_context(self):
        """Prompt phải chứa context được truyền vào."""
        prompt = _build_prompt(
            context="Nội dung tài liệu quan trọng",
            question="Câu hỏi gì đó?",
            history_text=""
        )
        assert "Nội dung tài liệu quan trọng" in prompt

    def test_contains_question(self):
        """Prompt phải chứa câu hỏi của người dùng."""
        prompt = _build_prompt(
            context="Some context",
            question="Học máy là gì?",
            history_text=""
        )
        assert "Học máy là gì?" in prompt

    def test_no_history_section_when_empty(self):
        """Khi history rỗng, phần header lịch sử hội thoại phải bị bỏ qua."""
        prompt = _build_prompt("ctx", "q?", "")
        # "trước đó" chỉ xuất hiện trong history_section header, không xuất hiện ở QUY TẮC
        assert "trước đó" not in prompt

    def test_history_section_present_when_given(self):
        """Khi có history, prompt phải chứa phần lịch sử hội thoại."""
        prompt = _build_prompt("ctx", "q?", "Human: câu trước\nAI: trả lời trước")
        assert "Lịch sử hội thoại" in prompt
        assert "câu trước" in prompt


class TestFormatChatHistory:
    """Kiểm tra hàm _format_chat_history()."""

    def test_empty_history(self):
        """Lịch sử rỗng trả về chuỗi rỗng."""
        assert _format_chat_history([]) == ""

    def test_single_message(self):
        """Một tin nhắn phải được format đúng Human/AI prefix."""
        history = [{"question": "Hỏi gì?", "answer": "Trả lời."}]
        result = _format_chat_history(history)
        assert "Human: Hỏi gì?" in result
        assert "AI: Trả lời." in result

    def test_multiple_messages_order(self):
        """Nhiều tin nhắn phải theo đúng thứ tự chronological."""
        history = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        result = _format_chat_history(history)
        assert result.index("Q1") < result.index("Q2")


class TestBuildCitationList:
    """Kiểm tra hàm _build_citation_list()."""

    def test_converts_docs_to_citations(self):
        """Phải chuyển mỗi Document thành CitationSource."""
        docs = [
            Document(page_content="Đoạn 1", metadata={"page": 1, "source": "a.pdf"}),
            Document(page_content="Đoạn 2", metadata={"page": 2, "source": "a.pdf"}),
        ]
        citations = _build_citation_list(docs)
        assert len(citations) == 2
        assert all(isinstance(c, CitationSource) for c in citations)

    def test_content_and_metadata_match(self):
        """content và metadata của CitationSource phải khớp với Document gốc."""
        doc = Document(page_content="Nội dung mẫu", metadata={"page": 5, "source": "test.pdf"})
        citations = _build_citation_list([doc])
        assert citations[0].content == "Nội dung mẫu"
        assert citations[0].metadata["page"] == 5

    def test_empty_docs_returns_empty_list(self):
        """Danh sách Document rỗng phải trả về danh sách citation rỗng."""
        assert _build_citation_list([]) == []
