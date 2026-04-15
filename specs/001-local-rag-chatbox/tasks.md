# Tasks: ChatBox Local-First Knowledge Assistant

**Input**: Design documents from /specs/001-local-rag-chatbox/
**Prerequisites**: plan.md, spec.md, input-parsing-chunking.md

**Tests**: Integration and benchmark tasks are included because the request explicitly requires full-pipeline integration testing and performance benchmarking.

**Organization**: Tasks are grouped by phase and user story so each story can be implemented and validated independently.

## Format: [ID] [P?] [Story] Description

- [P] means task can run in parallel (different files, no unresolved dependency).
- [Story] label is used only for user story phases.
- Each task includes file path and dependency direction.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project scaffolding and baseline tooling.

- [X] T001 Create package skeleton in src/chatbox/app/__init__.py, src/chatbox/domain/__init__.py, src/chatbox/ingestion/__init__.py, src/chatbox/chunking/__init__.py, src/chatbox/rag/__init__.py, src/chatbox/graphrag/__init__.py, src/chatbox/orchestration/__init__.py, src/chatbox/storage/__init__.py, and src/chatbox/observability/__init__.py (depends on none; before T007)
- [X] T002 Initialize project dependencies and tool config in pyproject.toml (depends on T001; before T003, T004, T007)
- [X] T003 [P] Add formatter and linter configuration in ruff.toml and .editorconfig (depends on T002; before T062)
- [X] T004 [P] Configure pytest base settings in pytest.ini (depends on T002; before T012)
- [X] T005 [P] Add local runtime environment template in .env.example (depends on T002; before T047)
- [X] T006 Create application bootstrap wiring in src/chatbox/app/bootstrap.py (depends on T002; before T013, T034, T043, T051)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared domain, ports, and observability required by all stories.

**CRITICAL**: No user story starts before this phase is complete.

- [X] T007 Define core domain models in src/chatbox/domain/models.py (Document, Chunk, EmbeddingRecord, GraphNode, GraphEdge, RetrievalContext) (depends on T001, T002; before T016, T029, T038, T047)
- [X] T008 Define request/response contracts in src/chatbox/domain/contracts.py (GenerationRequest, GenerationResponse, ingest/query payloads) (depends on T007; before T034, T043, T051)
- [X] T009 Create shared domain errors in src/chatbox/domain/errors.py (depends on T007; before T026, T035, T044, T054)
- [X] T010 Define storage abstraction ports in src/chatbox/storage/ports.py (VectorStorePort, GraphStorePort, MetadataStorePort) (depends on T007, T008; before T031, T039, T048)
- [X] T011 Implement metadata persistence in src/chatbox/storage/sqlite_metadata.py (depends on T010; before T026)
- [X] T012 [P] Add foundational model and contract unit tests in tests/unit/domain/test_models_contracts.py (depends on T007, T008, T009; before T015)
- [X] T013 Implement structured logging and trace helpers in src/chatbox/observability/logging.py and src/chatbox/observability/tracing.py (depends on T006; before T026, T035, T044, T054)
- [X] T014 [P] Implement metrics helpers in src/chatbox/observability/metrics.py (depends on T006; before T028, T037, T046, T055)
- [X] T015 Add foundational smoke test for bootstrap and ports in tests/integration/test_foundation_bootstrap.py (depends on T010, T011, T012, T013, T014; before T016)

**Checkpoint**: Foundation complete, all user stories can proceed.

---

## Phase 3: User Story 1 - Ingest + Parse + Chunk PDF/DOCX + Chunking Documentation (Priority: P1) 🎯 MVP

**Goal**: Deliver an end-to-end ingest pipeline for PDF/DOCX with deterministic chunking and extension-ready documentation.

**Independent Test**: Import sample PDF/DOCX and verify parse output, chunk ordering, metadata, and documentation accuracy without retrieval/generation.

### Tests for User Story 1

- [X] T016 [P] [US1] Add PDF parser unit tests in tests/unit/ingestion/test_pdf_parser.py (depends on T015; before T020)
- [X] T017 [P] [US1] Add DOCX parser unit tests in tests/unit/ingestion/test_docx_parser.py (depends on T015; before T021)
- [X] T018 [P] [US1] Add chunking invariant tests for token size, overlap, and boundaries in tests/unit/chunking/test_chunker_policy.py (depends on T015; before T023)
- [X] T019 [US1] Add ingestion integration test (import to persisted chunks) in tests/integration/test_ingest_parse_chunk_pipeline.py (depends on T016, T017, T018; before T028)

### Implementation for User Story 1

- [X] T020 [P] [US1] Implement text-first PDF parser in src/chatbox/ingestion/parsers/pdf_parser.py (depends on T016; before T026)
- [X] T021 [P] [US1] Implement DOCX parser in src/chatbox/ingestion/parsers/docx_parser.py (depends on T017; before T026)
- [X] T022 [P] [US1] Implement normalization pipeline in src/chatbox/ingestion/normalizer.py (depends on T020, T021; before T023, T026)
- [X] T023 [P] [US1] Implement boundary detector and tokenizer helpers in src/chatbox/chunking/boundary.py and src/chatbox/chunking/tokenizer.py (depends on T018, T022; before T024)
- [X] T024 [US1] Implement chunker with deterministic IDs in src/chatbox/chunking/chunker.py (depends on T023; before T026)
- [X] T025 [P] [US1] Add parser and chunking fixtures in tests/fixtures/ingestion/README.md and tests/fixtures/chunking/README.md (depends on T016, T017, T018; before T028)
- [X] T026 [US1] Implement ingestion coordinator workflow in src/chatbox/ingestion/coordinator.py (depends on T009, T011, T013, T020, T021, T022, T024; before T027, T028)
- [X] T027 [US1] Expose ingest API and CLI commands in src/chatbox/app/api/documents.py and src/chatbox/app/cli/import_cmd.py (depends on T026; before T028, T029)
- [X] T028 [US1] Add parse/chunk benchmark suite in tests/benchmark/test_parse_chunk_latency.py (depends on T014, T019, T025, T026, T027; before T056)

### Documentation for User Story 1

- [X] T029 [US1] Expand parsing and chunking extension guide in specs/001-local-rag-chatbox/input-parsing-chunking.md (depends on T024, T026; before T056)

**Checkpoint**: US1 is demo-ready and can be shown independently.

---

## Phase 4: User Story 2 - RAG Retrieval + Context Assembly (Priority: P2)

**Goal**: Add semantic retrieval path and produce retrieval context payload for downstream response layer.

**Independent Test**: Given indexed chunks, query returns ranked RAG context with traceable chunk citations.

### Tests for User Story 2

- [X] T030 [P] [US2] Add vector store port contract tests in tests/contract/test_vector_store_port.py (depends on T010; before T032)
- [X] T031 [P] [US2] Add RAG retriever unit tests in tests/unit/rag/test_retriever.py (depends on T015; before T034)
- [X] T032 [US2] Add RAG integration test (query to context assembly) in tests/integration/test_rag_retrieval_context.py (depends on T030, T031; before T037)

### Implementation for User Story 2

- [X] T033 [P] [US2] Implement embedding service in src/chatbox/rag/embeddings.py (depends on T007, T010; before T034)
- [X] T034 [US2] Implement RAG indexer and retriever in src/chatbox/rag/indexer.py and src/chatbox/rag/retriever.py (depends on T031, T033; before T035, T037)
- [X] T035 [US2] Implement retrieval context assembler for RAG path in src/chatbox/orchestration/merger.py (depends on T008, T009, T013, T034; before T036, T043)
- [X] T036 [US2] Expose query endpoint in RAG-only mode in src/chatbox/app/api/query.py (depends on T027, T035; before T037, T052)
- [X] T037 [US2] Add retrieval latency benchmark for RAG path in tests/benchmark/test_rag_retrieval_latency.py (depends on T014, T032, T034, T036; before T056)

**Checkpoint**: US2 works independently using only RAG path.

---

## Phase 5: User Story 3 - GraphRAG Parallel Pipeline + Context Fusion (Priority: P2)

**Goal**: Build graph extraction/retrieval and parallel fusion with RAG context.

**Independent Test**: For multi-entity queries, system runs GraphRAG in parallel with RAG and returns fused context with degradation handling.

### Tests for User Story 3

- [X] T038 [P] [US3] Add graph store port contract tests in tests/contract/test_graph_store_port.py (depends on T010; before T041)
- [X] T039 [P] [US3] Add GraphRAG builder unit tests in tests/unit/graphrag/test_graph_builder.py (depends on T015; before T041)
- [X] T040 [US3] Add parallel retrieval integration test for RAG + GraphRAG in tests/integration/test_parallel_retrieval_fusion.py (depends on T032, T038, T039; before T046)

### Implementation for User Story 3

- [X] T041 [P] [US3] Implement graph entity/relation extraction in src/chatbox/graphrag/entity_extractor.py and src/chatbox/graphrag/relation_extractor.py (depends on T039; before T042)
- [X] T042 [US3] Implement graph builder and retriever in src/chatbox/graphrag/builder.py and src/chatbox/graphrag/retriever.py (depends on T038, T041; before T043, T046)
- [X] T043 [US3] Implement parallel retrieval orchestrator in src/chatbox/orchestration/parallel_retrieval.py (depends on T034, T035, T042; before T044, T046, T051)
- [X] T044 [US3] Add fusion and degraded-mode rules in src/chatbox/orchestration/merger.py (depends on T009, T013, T043; before T045, T051)
- [X] T045 [US3] Add query API support for hybrid retrieval mode in src/chatbox/app/api/query.py (depends on T043, T044; before T046, T052)
- [X] T046 [US3] Add hybrid retrieval benchmark in tests/benchmark/test_hybrid_retrieval_latency.py (depends on T014, T040, T042, T045; before T056)

**Checkpoint**: US3 works independently with parallel retrieval and context fusion.

---

## Phase 6: User Story 4 - LLM Response Orchestration + Streaming + Fallback Modes (Priority: P3)

**Goal**: Produce final answer generation layer with streaming output and robust fallback modes.

**Independent Test**: Query returns streamed response with citations; fallback mode is activated when one retrieval path or provider fails.

### Tests for User Story 4

- [X] T047 [P] [US4] Add LLM provider port contract tests in tests/contract/test_llm_provider_port.py (depends on T008; before T049)
- [X] T048 [P] [US4] Add response layer unit tests for citations and uncertainty flags in tests/unit/orchestration/test_llm_response.py (depends on T015; before T051)
- [X] T049 [US4] Add streaming and fallback integration tests in tests/integration/test_response_streaming_fallback.py (depends on T040, T047, T048; before T055)

### Implementation for User Story 4

- [X] T050 [P] [US4] Implement Ollama and vLLM provider adapters in src/chatbox/orchestration/providers/ollama_adapter.py and src/chatbox/orchestration/providers/vllm_adapter.py (depends on T047; before T051)
- [X] T051 [US4] Implement LLM response orchestration with citation shaping in src/chatbox/orchestration/llm_response.py (depends on T043, T044, T048, T050; before T052, T055)
- [X] T052 [US4] Implement query API streaming responses and mode selection in src/chatbox/app/api/query.py (depends on T045, T051; before T053, T055)
- [X] T053 [US4] Implement CLI streaming query command in src/chatbox/app/cli/query_cmd.py (depends on T052; before T055)
- [X] T054 [US4] Add fallback mode policy config in src/chatbox/app/config.py (hybrid default, rag-only, graph-only) (depends on T009, T051; before T055)
- [X] T055 [US4] Add full response latency benchmark for warm/cold model states in tests/benchmark/test_full_response_latency.py (depends on T014, T049, T052, T053, T054; before T056)

**Checkpoint**: US4 delivers full response behavior with streaming and fallback modes.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validate whole-system behavior, performance, and release readiness.

- [X] T056 Add end-to-end full pipeline integration test in tests/integration/test_full_pipeline_e2e.py (depends on T028, T037, T046, T055; before T057, T062)
- [X] T057 [P] Consolidate benchmark runner and reporting in tests/benchmark/test_pipeline_sla_report.py (depends on T028, T037, T046, T055, T056; before T061)
- [X] T058 [P] Document runbook for local demo and troubleshooting in docs/runbook/local-first-chatbox.md (depends on T056; before T061)
- [X] T059 [P] Document API usage and examples in docs/api/chatbox-local-api.md (depends on T052, T056; before T061)
- [X] T060 [P] Document future feature groups and scope lock in docs/roadmap/future-feature-groups.md (depends on T056; before T061)
- [X] T061 Validate quickstart scenario and update specs/001-local-rag-chatbox/quickstart.md (depends on T057, T058, T059, T060; before T062)
- [X] T062 Final polish pass for lint/test/benchmark gates in pyproject.toml and .github/workflows/local-ci.yml (depends on T003, T056, T057, T061; final)

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): starts immediately.
- Foundational (Phase 2): depends on Setup; blocks all user stories.
- US1 (Phase 3): depends on Foundational; first MVP slice and first demo target.
- US2 (Phase 4): depends on Foundational and uses US1 output (chunks/index input).
- US3 (Phase 5): depends on Foundational and integrates with US2 retrieval context.
- US4 (Phase 6): depends on US2 and US3 orchestration outputs.
- Polish (Phase 7): depends on all prior stories.

### User Story Dependencies

- US1 (P1): no dependency on other stories; can be demoed immediately after completion.
- US2 (P2): depends on US1 artifacts (chunked content) but remains independently testable.
- US3 (P2): depends on US1 artifacts and fuses with US2 retrieval contracts.
- US4 (P3): depends on US2 and US3 context outputs for final answer orchestration.

### Task Dependency Rules

- Tasks marked [P] are parallel-safe only after their listed prerequisite IDs are complete.
- Integration and benchmark tasks are always after their functional path tasks.
- T056 is the global pipeline validation gate before final release tasks.

## Parallel Execution Examples

### Parallel Example: US1

- T016 and T017 and T018 can run in parallel after T015.
- T020 and T021 can run in parallel after their test stubs are available.

### Parallel Example: US2

- T030 and T031 can run in parallel.
- T033 can run in parallel with T031, then converge at T034.

### Parallel Example: US3

- T038 and T039 can run in parallel.
- T041 can start once T039 is done while T038 progresses independently.

### Parallel Example: US4

- T047 and T048 can run in parallel.
- T050 can progress in parallel with T048 before merging at T051.

## Implementation Strategy

### MVP-First (Early Demo)

1. Complete Phase 1 and Phase 2.
2. Complete US1 (Phase 3) and run T019 + T028.
3. Demo ingest/parse/chunk flow immediately after US1 checkpoint.

### Incremental Delivery

1. Ship US1 as first valuable increment.
2. Add US2 for semantic retrieval.
3. Add US3 for parallel GraphRAG fusion.
4. Add US4 for final streamed response experience.
5. Run Polish phase for pipeline-level release confidence.

### Multi-Developer Strategy

1. Team converges on Phase 1 and Phase 2.
2. Split by story owner after foundation:
   - Engineer A: US1 maintenance and benchmark.
   - Engineer B: US2 retrieval path.
   - Engineer C: US3 graph path.
   - Engineer D: US4 response orchestration.
3. Rejoin for T056-T062 final hardening.

## Notes

- Every task line follows checklist format: checkbox, Task ID, optional [P], optional [USx], clear action with file paths.
- Dependencies are embedded per task using "depends on" and "before" markers.
- The schedule favors earliest demo after US1 completion.
