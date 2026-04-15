# Implementation Plan: ChatBox Local-First Knowledge Assistant

**Branch**: `001-local-rag-chatbox` | **Date**: 2026-04-15 | **Spec**: `/specs/001-local-rag-chatbox/spec.md`  
**Input**: Feature specification from `/specs/001-local-rag-chatbox/spec.md`

## Summary

Build a local-first Python system that ingests PDF/DOCX, parses and chunks documents, then serves question answering through parallel RAG and GraphRAG paths merged by an LLM response layer.  
This plan keeps storage vendor-neutral by introducing explicit storage abstractions and adapter-based bindings, so backend choices can evolve without rewriting domain logic.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: pydantic v2, fastapi (local API), typer (CLI), pypdf, python-docx, tiktoken (or equivalent tokenizer), networkx (graph processing abstraction), httpx/asyncio, structlog or standard logging  
**Storage**: Storage abstraction layer with pluggable adapters for vector and graph persistence; default local file metadata store (SQLite) for MVP control-plane data  
**Testing**: pytest, pytest-asyncio, pytest-benchmark, hypothesis (optional for parser/chunk invariants)  
**Target Platform**: Local workstation (Windows/Linux/macOS), offline-first runtime  
**Project Type**: Local backend service + CLI tooling  
**Performance Goals**: parse+chunk p95 <= 45s per valid 20MB file; retrieval p95 <= 2.5s; full response p95 <= 8s  
**Constraints**: local-first, no hard vendor lock for vector/graph store, traceable citations, deterministic chunk ordering  
**Scale/Scope**: MVP for internal knowledge workflows; 1-50 concurrent local queries; thousands of chunks per workspace

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Constitution file is currently template-level and defines no enforceable constraints.
- Gate A (clarity): PASS, feature scope and SLAs are explicit.
- Gate B (testability): PASS, measurable SC and latency targets exist.
- Gate C (local-first constraints): PASS, no mandatory cloud dependencies.

## Project Structure

### Documentation (this feature)

```text
specs/001-local-rag-chatbox/
├── plan.md
├── input-parsing-chunking.md
├── data-model.md                 # optional follow-up expansion from this plan
├── contracts/                    # optional API/event contract files from this plan
└── tasks.md                      # generated in /speckit.tasks
```

### Source Code (repository root)

```text
src/chatbox/
├── app/
│   ├── api/                      # local HTTP endpoints
│   ├── cli/                      # import/query/admin commands
│   └── bootstrap.py              # dependency wiring
├── domain/
│   ├── models.py                 # Document, Chunk, EmbeddingRecord, etc.
│   ├── contracts.py              # GenerationRequest/Response and retrieval contracts
│   └── errors.py
├── ingestion/
│   ├── coordinator.py
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   └── docx_parser.py
│   └── normalizer.py
├── chunking/
│   ├── chunker.py
│   ├── boundary.py
│   └── tokenizer.py
├── rag/
│   ├── embeddings.py
│   ├── indexer.py
│   └── retriever.py
├── graphrag/
│   ├── builder.py
│   ├── entity_extractor.py
│   ├── relation_extractor.py
│   └── retriever.py
├── orchestration/
│   ├── parallel_retrieval.py
│   ├── merger.py
│   └── llm_response.py
├── storage/
│   ├── ports.py                  # abstraction interfaces
│   ├── sqlite_metadata.py
│   ├── vector_adapters/
│   │   ├── base.py
│   │   └── qdrant_adapter.py     # optional initial adapter
│   └── graph_adapters/
│       ├── base.py
│       └── neo4j_adapter.py      # optional initial adapter
└── observability/
   ├── logging.py
   ├── metrics.py
   └── tracing.py

tests/
├── unit/
├── contract/
├── integration/
└── benchmark/
```

**Structure Decision**: Single Python project with clean modular boundaries. Domain and storage abstractions are isolated from vendor adapters to preserve local-first portability and reduce migration cost.

## Module Architecture

1. `ingestion`
  Parses source files, extracts text + structure, normalizes encoding, and emits `Document` payloads for chunking.
2. `chunking`
  Applies semantic-first chunk policy (heading/paragraph first, token hard-cap fallback), emits stable `Chunk` sequences.
3. `rag`
  Builds embeddings, maintains vector index through `VectorStorePort`, retrieves semantic contexts.
4. `graphrag`
  Builds graph artifacts from chunks (`GraphNode`/`GraphEdge`) and retrieves relation paths through `GraphStorePort`.
5. `orchestration`
  Runs parallel RAG + GraphRAG retrieval with timeout budget, merges results into `RetrievalContext`, calls local LLM.
6. `observability`
  Emits structured logs, latency metrics, and trace IDs for ingest/query lifecycle.

## Data Model And Contracts

### Core Domain Models

1. `Document`
  - Fields: `document_id`, `source_uri`, `file_type`, `title`, `language`, `raw_text`, `sections`, `ingest_status`, `created_at`, `updated_at`
  - Invariants: `document_id` unique; `file_type in {pdf, docx}` for MVP.
2. `Chunk`
  - Fields: `chunk_id`, `document_id`, `chunk_order`, `text`, `token_count`, `overlap_tokens`, `boundary_type`, `source_pointer`, `checksum`
  - Invariants: strict monotonic `chunk_order` per `document_id`; deterministic `checksum` for idempotent ingest.
3. `EmbeddingRecord`
  - Fields: `embedding_id`, `chunk_id`, `model_id`, `vector`, `dimension`, `created_at`
  - Invariants: one active embedding per (`chunk_id`, `model_id`) in MVP.
4. `GraphNode`
  - Fields: `node_id`, `label`, `node_type`, `aliases`, `document_refs`, `confidence`
5. `GraphEdge`
  - Fields: `edge_id`, `from_node_id`, `to_node_id`, `relation_type`, `evidence_chunk_ids`, `weight`, `confidence`
6. `RetrievalContext`
  - Fields: `query_id`, `query_text`, `rag_hits`, `graph_hits`, `merged_evidence`, `degraded_mode`, `timings`, `trace_id`
7. `GenerationRequest`
  - Fields: `query_id`, `user_query`, `retrieval_context`, `response_style`, `max_tokens`, `temperature`
8. `GenerationResponse`
  - Fields: `response_id`, `query_id`, `answer_text`, `citations`, `uncertainty_flags`, `provider_metadata`, `latency_ms`

### Service Contracts

1. `ParserPort`
  - `parse(file_path) -> DocumentParseResult`
2. `ChunkerPort`
  - `chunk(document, policy) -> list[Chunk]`
3. `VectorStorePort`
  - `upsert_embeddings(records)`
  - `search(query_vector, top_k, filters) -> list[RagHit]`
4. `GraphStorePort`
  - `upsert_graph(nodes, edges)`
  - `search_paths(query_entities, top_k, max_hops) -> list[GraphHit]`
5. `RetrieverOrchestratorPort`
  - `retrieve_parallel(query, timeout_budget) -> RetrievalContext`
6. `LLMProviderPort`
  - `generate(request: GenerationRequest) -> GenerationResponse`

### API Contract Sketch (local)

```text
POST /v1/documents:ingest
POST /v1/query
GET  /v1/documents/{document_id}
GET  /v1/health
```

## Parallel RAG And GraphRAG Flow

1. Receive user query and allocate `query_id` + `trace_id`.
2. Build query embedding and launch RAG retrieval task.
3. In parallel, run entity extraction and graph path retrieval task.
4. Await both tasks with per-path timeout budget.
5. If one path fails/timeout, mark degraded mode and continue with successful path.
6. Merge evidence by source confidence, relevance score, and citation traceability.
7. Build `GenerationRequest` and call local LLM provider.
8. Return `GenerationResponse` with citations and uncertainty flags.

Pseudo-flow:

```python
rag_task = asyncio.create_task(rag_retrieve(query))
graph_task = asyncio.create_task(graph_retrieve(query))
rag_result, graph_result = await gather_with_budget(rag_task, graph_task, budget_ms)
ctx = merge_results(rag_result, graph_result)
response = llm_provider.generate(build_generation_request(query, ctx))
```

## Performance Strategy

### Fast File Read + PDF/DOCX Import

1. Use streaming reads and avoid loading full binary payload twice.
2. PDF path optimized for text PDFs: page-level extraction with bounded worker pool.
3. DOCX path optimized by paragraph/run iteration with style boundary hints.
4. Early reject unsupported/corrupted files before full pipeline cost.
5. Persist intermediate parse artifacts to avoid recomputation on retry.

### Parse/Chunk Throughput

1. Normalize text once, then pass immutable payload to chunker.
2. Semantic-first chunking with deterministic fallback to token cap.
3. Batch embedding/index writes to reduce adapter call overhead.
4. Cache tokenizer instances and reuse model metadata.

### Query/Response Latency

1. Parallel retrieval with strict timeout budget split (RAG and GraphRAG).
2. Non-blocking degradation when one path misses SLA.
3. Limit merged context size before generation to bound LLM latency.
4. Add warmup endpoint for local model cold-start mitigation.

## Observability Plan

1. Structured logs for ingest/query lifecycle with `trace_id`, `document_id`, `query_id`.
2. Metrics (histograms): parse latency, chunk latency, retrieval latency, generation latency.
3. Counters: parse failures by file type, fallback rate, citation coverage.
4. Optional local trace spans for parallel path timings and merge decision diagnostics.

## Testing And Benchmark Plan

### Unit Tests

1. Parser correctness for text PDF and DOCX fixtures.
2. Chunking invariants: order stability, overlap size, boundary labeling.
3. Merger logic: confidence ranking and degraded-mode behavior.

### Contract Tests

1. `VectorStorePort` adapter conformance suite.
2. `GraphStorePort` adapter conformance suite.
3. `LLMProviderPort` response schema and error mapping.

### Integration Tests

1. End-to-end ingest from file to indexed artifacts.
2. Parallel query flow with both paths healthy.
3. Degraded flow where one retrieval path fails.

### Benchmarks (minimum)

1. Parse+chunk latency benchmark on representative 20MB PDF/DOCX corpus.
2. Retrieval latency benchmark (RAG-only, GraphRAG-only, hybrid).
3. Full response latency benchmark with local LLM warm and cold states.

Benchmark thresholds:

1. parse+chunk p95 <= 45s per valid 20MB file.
2. retrieval p95 <= 2.5s.
3. full response p95 <= 8s.

## Implementation Phases

### Phase 0 - Foundations

1. Define domain models and interface ports.
2. Implement metadata persistence and ID strategy.
3. Setup observability baseline and benchmark harness.

Exit criteria:

1. Domain contracts compile and pass unit tests.
2. Adapter test harness available.

### Phase 1 - Ingestion And Chunking Core

1. Build PDF/DOCX parsers and normalizer.
2. Implement chunking policy and deterministic chunk IDs.
3. Produce parsing/chunking documentation for extensibility.

Exit criteria:

1. Ingest pipeline works end-to-end on fixture corpus.
2. parse+chunk benchmark report generated.

### Phase 2 - Retrieval Paths

1. Implement vector indexing/retrieval via abstraction.
2. Implement graph build/retrieval via abstraction.
3. Add adapter stubs and at least one working local adapter per store type.

Exit criteria:

1. RAG and GraphRAG contract tests pass.
2. retrieval benchmark report generated.

### Phase 3 - LLM Response Layer

1. Implement parallel orchestrator + merge policy.
2. Integrate local LLM provider abstraction (Ollama/vLLM adapters).
3. Add citation and uncertainty response shaping.

Exit criteria:

1. Hybrid and degraded integration tests pass.
2. full response benchmark report generated.

### Phase 4 - Hardening

1. Tune latency hotspots from benchmark evidence.
2. Improve failure messages and recovery paths.
3. Finalize operational runbook and local quickstart.

Exit criteria:

1. All SLA benchmarks meet p95 targets.
2. Release checklist signed.

## Risks And Mitigations

1. Storage backend churn risk.
  Mitigation: strict ports/adapters and contract tests before backend switch.
2. Graph extraction quality variance.
  Mitigation: confidence thresholds and fallback-safe merge policy.
3. Local model cold-start latency.
  Mitigation: warmup command, cached prompts, bounded context size.
4. Chunk quality drift across document styles.
  Mitigation: golden fixtures and chunking regression benchmarks.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Storage ports and adapters | Avoid vendor lock while keeping local-first deployment | Direct backend calls would couple domain logic to one store and block migration |
