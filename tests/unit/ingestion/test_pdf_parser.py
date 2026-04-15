from pathlib import Path

from chatbox.ingestion.parsers.pdf_parser import PdfParser


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [_FakePage("A"), _FakePage("B")]


def test_pdf_parser_text_first_reads_pages() -> None:
    parser = PdfParser(reader_factory=_FakeReader)
    parsed = parser.parse(Path("sample.pdf"))

    assert parsed.file_type == "pdf"
    assert parsed.text == "A\nB"
    assert parsed.metadata["page_count"] == 2
