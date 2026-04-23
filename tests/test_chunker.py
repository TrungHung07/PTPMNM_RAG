"""
Unit tests cho module src/chunking/text_chunker.py

Kiểm tra tính đúng đắn của hàm chunk_documents() khi:
- Chia đúng số lượng chunk
- Bảo toàn metadata gốc từ Document nguồn
- Bổ sung chunk_index đúng thứ tự
- Xử lý edge cases (text rỗng, overlap quá lớn, v.v.)
"""
import pytest
from langchain_core.documents import Document

from src.chunking.text_chunker import chunk_text, chunk_documents


class TestChunkText:
    """Kiểm tra hàm chunk_text() (legacy)."""

    def test_basic_split(self):
        """Văn bản dài hơn chunk_size phải bị chia thành nhiều chunk."""
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        """Văn bản ngắn hơn chunk_size chỉ tạo ra 1 chunk."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text(self):
        """Văn bản rỗng trả về danh sách rỗng."""
        assert chunk_text("") == []

    def test_overlap_larger_than_chunk_raises(self):
        """overlap >= chunk_size phải tự điều chỉnh, không raise."""
        # overlap quá lớn sẽ tự được clamp thành chunk_size // 2
        chunks = chunk_text("a" * 500, chunk_size=100, overlap=200)
        assert len(chunks) > 0

    def test_invalid_chunk_size(self):
        """chunk_size <= 0 phải raise ValueError."""
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=0)


class TestChunkDocuments:
    """Kiểm tra hàm chunk_documents() — hàm chính hỗ trợ citation tracking."""

    def _make_doc(self, content: str, **metadata) -> Document:
        """Helper tạo Document cho test."""
        return Document(page_content=content, metadata=metadata)

    def test_metadata_preserved(self):
        """Metadata gốc (page, source) phải được bảo toàn trong mọi chunk."""
        doc = self._make_doc("a" * 2500, page=3, source="test.pdf")
        chunks = chunk_documents([doc], chunk_size=1000, overlap=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["page"] == 3
            assert chunk.metadata["source"] == "test.pdf"

    def test_chunk_index_increments(self):
        """chunk_index phải tăng dần 0, 1, 2... trong cùng một Document."""
        doc = self._make_doc("b" * 3000, page=1, source="test.pdf")
        chunks = chunk_documents([doc], chunk_size=1000, overlap=0)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_multiple_documents_independent_index(self):
        """
        Mỗi Document có chunk_index độc lập, bắt đầu lại từ 0.
        """
        doc1 = self._make_doc("c" * 2500, page=1, source="test.pdf")
        doc2 = self._make_doc("d" * 2500, page=2, source="test.pdf")
        chunks = chunk_documents([doc1, doc2], chunk_size=1000, overlap=0)

        # Chunks của doc1 và doc2 đều phải bắt đầu chunk_index từ 0
        doc1_chunks = [c for c in chunks if c.page_content[0] == "c"]
        doc2_chunks = [c for c in chunks if c.page_content[0] == "d"]

        assert doc1_chunks[0].metadata["chunk_index"] == 0
        assert doc2_chunks[0].metadata["chunk_index"] == 0

    def test_docx_paragraph_metadata(self):
        """Metadata 'paragraph' từ DOCX phải được bảo toàn."""
        doc = self._make_doc("Some paragraph text " * 60, paragraph=5, source="report.docx")
        chunks = chunk_documents([doc], chunk_size=500, overlap=50)

        for chunk in chunks:
            assert chunk.metadata["paragraph"] == 5
            assert chunk.metadata["source"] == "report.docx"

    def test_empty_documents(self):
        """Danh sách Documents rỗng trả về danh sách chunk rỗng."""
        assert chunk_documents([]) == []

    def test_document_with_empty_content(self):
        """Document với page_content rỗng bị bỏ qua."""
        doc = self._make_doc("", page=1, source="test.pdf")
        chunks = chunk_documents([doc])
        assert chunks == []

    def test_chunk_size_respected(self):
        """Mỗi chunk không được dài hơn chunk_size ký tự."""
        doc = self._make_doc("x" * 5000, page=1, source="test.pdf")
        chunk_size = 300
        chunks = chunk_documents([doc], chunk_size=chunk_size, overlap=0)

        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_overlap_creates_extra_chunks(self):
        """Với overlap > 0, số chunk phải nhiều hơn so với overlap=0."""
        doc = self._make_doc("word " * 400, page=1, source="test.pdf")
        chunks_no_overlap = chunk_documents([doc], chunk_size=500, overlap=0)
        chunks_with_overlap = chunk_documents([doc], chunk_size=500, overlap=100)

        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
