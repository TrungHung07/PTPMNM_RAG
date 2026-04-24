"""
Unit tests kiểm tra các parser (PDF và DOCX) trả về Document đúng format.

Các test này dùng file mẫu có sẵn trong repo hoặc tạo file tạm bằng
thư viện tương ứng để không cần kết nối LLM hay database.
"""
import pytest
from pathlib import Path
import tempfile

from langchain_core.documents import Document
from src.parsers.docx_parser import extract_documents_docx, extract_text_docx


class TestDocxParser:
    """Kiểm tra extract_documents_docx() với file DOCX tạm."""

    def _create_temp_docx(self, paragraphs: list[str]) -> Path:
        """Helper tạo file DOCX tạm với nội dung cho trước."""
        from docx import Document as DocxDoc
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            tmp_path = Path(f.name)
        doc = DocxDoc()
        for para in paragraphs:
            doc.add_paragraph(para)
        doc.save(str(tmp_path))
        return tmp_path

    def test_returns_list_of_documents(self):
        """Hàm phải trả về list[Document], không phải chuỗi."""
        tmp = self._create_temp_docx(["Hello world", "Second paragraph"])
        try:
            result = extract_documents_docx(tmp)
            assert isinstance(result, list)
            assert all(isinstance(d, Document) for d in result)
        finally:
            tmp.unlink(missing_ok=True)

    def test_paragraph_metadata(self):
        """Mỗi Document phải có metadata 'paragraph' đúng thứ tự và 'source'."""
        paragraphs = ["First paragraph", "Second paragraph", "Third paragraph"]
        tmp = self._create_temp_docx(paragraphs)
        try:
            result = extract_documents_docx(tmp)
            assert len(result) == 3

            for i, doc in enumerate(result, start=1):
                assert doc.metadata["paragraph"] == i
                assert doc.metadata["source"] == tmp.name
                assert doc.page_content == paragraphs[i - 1]
        finally:
            tmp.unlink(missing_ok=True)

    def test_empty_paragraphs_skipped(self):
        """Đoạn văn rỗng phải bị bỏ qua, không thêm vào kết quả."""
        paragraphs = ["Real content", "", "   ", "More content"]
        tmp = self._create_temp_docx(paragraphs)
        try:
            result = extract_documents_docx(tmp)
            # Chỉ có 2 đoạn có nội dung thực
            assert len(result) == 2
            assert result[0].page_content == "Real content"
            assert result[1].page_content == "More content"
        finally:
            tmp.unlink(missing_ok=True)

    def test_legacy_extract_text_docx(self):
        """Hàm legacy extract_text_docx() vẫn phải trả về chuỗi."""
        tmp = self._create_temp_docx(["Para one", "Para two"])
        try:
            result = extract_text_docx(tmp)
            assert isinstance(result, str)
            assert "Para one" in result
            assert "Para two" in result
        finally:
            tmp.unlink(missing_ok=True)

    def test_source_filename_in_metadata(self):
        """metadata['source'] phải chứa tên file, không phải path đầy đủ."""
        tmp = self._create_temp_docx(["Content here"])
        try:
            result = extract_documents_docx(tmp)
            assert len(result) == 1
            # source là path.name nghĩa là chứa ít nhất tên file
            assert result[0].metadata["source"] == tmp.name
        finally:
            tmp.unlink(missing_ok=True)


class TestSampleDocxFile:
    """Kiểm tra với file sample.docx có sẵn trong repo."""

    SAMPLE_PATH = Path("sample.docx")

    def test_sample_docx_exists_and_parses(self):
        """File mẫu sample.docx phải parse được và trả về ít nhất 1 Document."""
        if not self.SAMPLE_PATH.exists():
            pytest.skip("sample.docx không tồn tại, bỏ qua test này")

        result = extract_documents_docx(self.SAMPLE_PATH)
        assert len(result) > 0
        assert all("paragraph" in d.metadata for d in result)
        assert all("source" in d.metadata for d in result)
