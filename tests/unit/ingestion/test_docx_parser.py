from pathlib import Path

from chatbox.ingestion.parsers.docx_parser import DocxParser


class _FakeParagraph:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDocument:
    def __init__(self, _path: str) -> None:
        self.paragraphs = [_FakeParagraph("Heading"), _FakeParagraph("Body")]


def test_docx_parser_reads_paragraphs() -> None:
    parser = DocxParser(document_factory=_FakeDocument)
    parsed = parser.parse(Path("sample.docx"))

    assert parsed.file_type == "docx"
    assert parsed.text == "Heading\nBody"
    assert parsed.metadata["paragraph_count"] == 2
