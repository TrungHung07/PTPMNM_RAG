# Input Parsing And Chunking Design Guide

## Purpose

This document specifies how input parsing and chunking must work in ChatBox MVP, with extension points for future file types and chunking policies.

## Scope

1. Supported inputs in MVP: text PDF and DOCX.
2. Output contract: normalized `Document` and ordered `Chunk` records.
3. Non-goal: OCR-heavy scanned PDFs in MVP (handled as low-quality parse result).

## Parsing Pipeline

### Stage 1: Intake And Validation

1. Validate file extension and MIME hints.
2. Compute file checksum and file size.
3. Reject early if file is corrupted or unsupported.

Output:

- `IngestIntakeResult`: `document_id`, `source_uri`, `file_type`, `file_size`, `checksum`, `accepted`.

### Stage 2: Format-Specific Parsing

1. PDF parser (text-first):
   - Iterate pages in order.
   - Extract text blocks and page metadata.
   - Preserve page boundaries in source pointers.
2. DOCX parser:
   - Iterate paragraphs, runs, and heading styles.
   - Preserve section boundaries and list structure hints.

Output:

- `ParseResult`: `raw_text`, `segments`, `parse_quality`, `warnings`, `errors`.

### Stage 3: Normalization

1. Normalize Unicode spaces/newlines.
2. Remove duplicate whitespace while preserving paragraph breaks.
3. Canonicalize heading markers for boundary detection.

Output:

- `NormalizedDocument`: stable text and section map for chunking.

## Chunking Pipeline

### Policy (MVP)

1. Target size: 500 tokens per chunk.
2. Overlap: 80 tokens.
3. Boundary strategy:
   - First split by semantic boundaries (heading, paragraph).
   - If segment exceeds cap, apply token hard split.

### Chunk Generation Rules

1. Deterministic ordering by source location.
2. Deterministic `chunk_id` from (`document_id`, `chunk_order`, `checksum`).
3. Include source pointer (`page`, `section`, `offset_start`, `offset_end`).
4. Tag `boundary_type`: `heading`, `paragraph`, `token_cap`.

### Invariants

1. No empty chunks.
2. Chunk order must be contiguous.
3. Overlap must match policy unless at boundaries (first/last chunk).
4. Running chunker twice on unchanged input must yield identical chunk IDs.

## Contracts

### Parser Contract

```python
class ParserPort(Protocol):
    def parse(self, file_path: str, document_id: str) -> ParseResult: ...
```

### Chunker Contract

```python
class ChunkerPort(Protocol):
    def chunk(self, normalized_doc: NormalizedDocument, policy: ChunkPolicy) -> list[Chunk]: ...
```

### Policy Contract

```python
@dataclass(frozen=True)
class ChunkPolicy:
    target_tokens: int
    overlap_tokens: int
    prefer_semantic_boundaries: bool
```

## Error Handling

1. Parser errors return structured `ParseError` with code and location.
2. Partial parse allowed only when `parse_quality` remains above configured threshold.
3. If parsing fails, chunking must not run.
4. All failures emit traceable logs with `trace_id` and `document_id`.

## Performance Strategy

1. Use bounded worker pools for page-level extraction.
2. Avoid repeated tokenization by caching token counts per segment.
3. Stream parse output into chunker to reduce memory spikes for large files.
4. Persist normalized intermediate artifacts for retry/debug.

## Extensibility Guide

### Current MVP Wiring

1. Parser adapters are loaded through a file-type mapping used by the ingestion coordinator.
2. The coordinator normalizes parser output before chunking and persists both document and chunks.
3. Chunk IDs are deterministic and derived from stable inputs so re-ingestion is idempotent.
4. The token-window policy currently uses whitespace tokenization for deterministic local behavior.

### Add New File Type

1. Implement new parser adapter under `src/chatbox/ingestion/parsers/`.
2. Register parser in parser factory mapping.
3. Add fixtures for success, malformed, and edge-case inputs.
4. Pass parser contract test suite.

Recommended implementation checklist:

1. Return a parser result with explicit metadata counts (for example: page count, paragraph count).
2. Keep parser errors wrapped in domain `ParsingError` for consistent API/CLI behavior.
3. Ensure parser output keeps source ordering, then rely on chunker for deterministic IDs.

### Add New Chunk Policy

1. Introduce policy config in `ChunkPolicy`.
2. Implement deterministic split logic and overlap behavior.
3. Add chunk invariants tests and benchmark comparators.
4. Document migration impact on existing chunk IDs.

Policy authoring rules:

1. Always preserve monotonic `chunk_order` per document.
2. Never emit empty chunks.
3. Keep overlap bounded (`overlap_tokens < max_tokens`) and stable across runs.
4. Record any ID strategy change in release notes because it can affect re-indexing.

## Test Matrix For Parsing/Chunking

1. Happy path: text PDF and DOCX with headings and tables.
2. Edge path: empty sections, very long paragraphs, special characters.
3. Failure path: corrupted files, unsupported encoding.
4. Regression path: same file ingested twice yields same chunk IDs.

## Benchmark Baseline

1. Parse+chunk p95 <= 45s for valid 20MB file set.
2. Throughput report includes per-format distribution (PDF vs DOCX).
3. Benchmark output captures CPU, memory, and latency percentiles.
