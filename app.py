import argparse
import json
from pathlib import Path
from typing import Literal

from src.chunking.text_chunker import chunk_text
from src.parsers.docx_parser import extract_text_docx
from src.parsers.pdf_parser import extract_text_pdf

PdfBackend = Literal["pdfplumber", "pypdf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load text from PDF/DOCX files and split it into fixed-size chunks."
    )
    parser.add_argument("file", help="Path to an input .pdf or .docx file.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk (default: 1000).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlapping characters between chunks (default: 200).",
    )
    parser.add_argument(
        "--pdf-backend",
        choices=["pdfplumber", "pypdf"],
        default="pdfplumber",
        help="Backend to use when reading PDF files.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Use .json for JSON, any other suffix for text.",
    )
    return parser.parse_args()


def load_text(file_path: Path, pdf_backend: PdfBackend) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_pdf(file_path, backend=pdf_backend)
    if suffix == ".docx":
        return extract_text_docx(file_path)
    raise ValueError("Only .pdf and .docx files are supported.")


def save_chunks(chunks: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        payload = [{"id": idx, "text": chunk} for idx, chunk in enumerate(chunks)]
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    output_path.write_text("\n---\n".join(chunks), encoding="utf-8")


def main() -> None:
    args = parse_args()
    file_path = Path(args.file)

    if not file_path.exists():
        raise SystemExit(f"Input file does not exist: {file_path}")

    chunk_size = max(50, args.chunk_size)
    overlap = max(0, args.overlap)
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    try:
        text = load_text(file_path, pdf_backend=args.pdf_backend)
    except Exception as exc:
        raise SystemExit(f"Failed to read file: {exc}") from exc

    normalized_text = text.replace("\r\n", "\n").strip()
    chunks = chunk_text(normalized_text, chunk_size=chunk_size, overlap=overlap)

    print(f"Loaded file: {file_path}")
    print(f"Total characters: {len(normalized_text)}")
    print(f"Number of chunks: {len(chunks)}")

    preview_count = min(3, len(chunks))
    if preview_count:
        print("\nPreview:")
    for index in range(preview_count):
        print(f"\nChunk {index}:")
        print(chunks[index])

    if args.output:
        output_path = Path(args.output)
        save_chunks(chunks, output_path)
        print(f"\nChunks saved to: {output_path}")


if __name__ == "__main__":
    main()
