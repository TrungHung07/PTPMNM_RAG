# Future Feature Groups

## Ingestion Expansion

1. OCR pipeline for scanned PDFs
2. HTML and Markdown import adapters
3. Incremental document update detection

## Retrieval Expansion

1. Pluggable vector adapter set (Qdrant, Chroma, FAISS)
2. Advanced graph relation extraction with confidence calibration
3. Query-time reranking and citation deduplication

## Response Expansion

1. Multi-turn conversation memory
2. Response style presets and guardrails
3. Structured answer formats (JSON schema output)

## Scope Lock For Current Release

1. Keep support limited to PDF and DOCX.
2. Preserve deterministic chunk IDs for stable indexing.
3. Keep runtime local-first with no mandatory cloud services.
