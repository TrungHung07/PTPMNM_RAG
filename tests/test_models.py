"""
Unit tests cho src/models.py

Kiểm tra validation của CitationSource và AskResponse Pydantic models.
"""
import pytest
from src.models import CitationSource, AskResponse


class TestCitationSource:
    """Kiểm tra CitationSource model."""

    def test_basic_creation(self):
        """CitationSource phải tạo được từ content và metadata."""
        citation = CitationSource(
            content="Đoạn văn trích dẫn",
            metadata={"page": 3, "source": "test.pdf"}
        )
        assert citation.content == "Đoạn văn trích dẫn"
        assert citation.metadata["page"] == 3
        assert citation.metadata["source"] == "test.pdf"

    def test_pdf_metadata(self):
        """CitationSource với metadata PDF phải chứa page và source."""
        citation = CitationSource(
            content="PDF content",
            metadata={"page": 1, "source": "report.pdf", "chunk_index": 0}
        )
        assert "page" in citation.metadata
        assert "source" in citation.metadata
        assert "chunk_index" in citation.metadata

    def test_docx_metadata(self):
        """CitationSource với metadata DOCX phải chứa paragraph và source."""
        citation = CitationSource(
            content="DOCX paragraph content",
            metadata={"paragraph": 5, "source": "doc.docx", "chunk_index": 1}
        )
        assert citation.metadata["paragraph"] == 5


class TestAskResponse:
    """Kiểm tra AskResponse model."""

    def test_basic_creation(self):
        """AskResponse phải tạo được với question, answer và citations rỗng."""
        response = AskResponse(
            question="Câu hỏi gì đó?",
            answer="Câu trả lời tương ứng.",
            citations=[]
        )
        assert response.question == "Câu hỏi gì đó?"
        assert response.answer == "Câu trả lời tương ứng."
        assert response.citations == []

    def test_with_citations(self):
        """AskResponse phải chấp nhận danh sách citations hợp lệ."""
        citations = [
            CitationSource(content="Đoạn 1", metadata={"page": 1, "source": "a.pdf"}),
            CitationSource(content="Đoạn 2", metadata={"page": 2, "source": "a.pdf"}),
        ]
        response = AskResponse(
            question="Test?",
            answer="Answer.",
            citations=citations
        )
        assert len(response.citations) == 2
        assert response.citations[0].metadata["page"] == 1

    def test_serializable_to_dict(self):
        """AskResponse phải có thể serialize về dict (cho JSON response)."""
        response = AskResponse(
            question="Q?",
            answer="A.",
            citations=[CitationSource(content="ctx", metadata={"page": 1, "source": "f.pdf"})]
        )
        d = response.model_dump()
        assert "question" in d
        assert "answer" in d
        assert "citations" in d
        assert d["citations"][0]["content"] == "ctx"
        assert d["citations"][0]["metadata"]["page"] == 1
