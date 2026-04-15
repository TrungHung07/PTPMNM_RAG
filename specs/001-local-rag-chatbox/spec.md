# Feature Specification: ChatBox Local-First Knowledge Assistant

**Feature Branch**: `001-local-rag-chatbox`  
**Created**: 2026-04-15  
**Status**: Draft  
**Input**: User description: "Tạo đặc tả feature cho hệ thống ChatBox theo hướng local-first với LLM cục bộ; phạm vi gồm ingest/parse/chunk, RAG, GraphRAG song song, lớp phản hồi LLM, nhóm chức năng phát triển tương lai, hiệu năng MVP đọc file nhanh, và hỗ trợ PDF/DOCX."

## Clarifications

### Session 2026-04-15

- Q: Lựa chọn storage cho vector và graph trong MVP local-first là gì? → A: Option A - Vector store Qdrant local; Graph store Neo4j local; orchestration ở ứng dụng ChatBox.
- Q: Mức SLA cụ thể cho parse/chunk và query/response ở MVP là gì? → A: Option B - Parse+chunk p95 <= 45s/file 20MB; retrieval p95 <= 2.5s; full response p95 <= 8s.
- Q: Quy tắc chunking cho MVP là gì? → A: Option B - 500 tokens/chunk, overlap 80 tokens, ưu tiên semantic boundary (heading/paragraph), fallback hard cap theo token.
- Q: Chiến lược fallback giữa RAG-only, GraphRAG-only, hybrid là gì? → A: Option A - Mặc định hybrid (RAG + GraphRAG song song); timeout một path thì dùng path còn lại; hợp nhất theo độ tin cậy nguồn.
- Q: Scope của future features trong phiên bản đầu nên khóa ở mức nào? → A: Option B - Khóa 3 nhóm future sau v1: ingestion mở rộng định dạng, quản trị tri thức, và chất lượng/đánh giá phản hồi.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ingest, Parse, Chunk Tài Liệu Nguồn (Priority: P1)

Là người xây dựng kho tri thức cục bộ cho ChatBox, tôi muốn import file PDF/DOCX và nhận được các chunk đã chuẩn hóa để toàn bộ các tầng phía sau (RAG, GraphRAG, LLM response) có dữ liệu đầu vào tin cậy và nhất quán.

**Why this priority**: Đây là nền tảng bắt buộc. Nếu không có ingest/parse/chunk ổn định thì các luồng truy xuất và sinh phản hồi không có dữ liệu để vận hành.

**Independent Test**: Có thể kiểm thử độc lập bằng cách import một bộ PDF/DOCX mẫu, xác minh kết quả parse, metadata, chunking, và trạng thái xử lý mà không cần triển khai các luồng trả lời LLM.

**Acceptance Scenarios**:

1. **Given** người dùng chọn một file PDF hoặc DOCX hợp lệ, **When** hệ thống chạy ingest và parse, **Then** hệ thống tạo được văn bản chuẩn hóa cùng metadata nguồn (tên file, loại file, trang/section, thời điểm import).
2. **Given** văn bản đã parse thành công, **When** hệ thống chạy chunking, **Then** hệ thống tạo được danh sách chunk liên tiếp, có định danh duy nhất, thứ tự ổn định, và tham chiếu ngược về vị trí trong tài liệu gốc.
3. **Given** file không hợp lệ hoặc không đọc được nội dung, **When** ingest được thực thi, **Then** hệ thống đánh dấu lỗi rõ nguyên nhân, không tạo dữ liệu dở dang, và cho phép người dùng import lại sau khi chỉnh sửa tệp.

---

### User Story 2 - Truy Xuất Ngữ Nghĩa Qua RAG Path (Priority: P2)

Là người dùng ChatBox, tôi muốn truy vấn kiến thức từ các chunk đã ingest để nhận được các đoạn ngữ cảnh liên quan nhất trước khi mô hình sinh câu trả lời.

**Why this priority**: Sau khi đã có nền dữ liệu từ P1, RAG là luồng tạo giá trị sử dụng trực tiếp đầu tiên cho người dùng khi đặt câu hỏi dựa trên tài liệu của họ.

**Independent Test**: Có thể kiểm thử độc lập bằng một tập câu hỏi chuẩn và bộ tài liệu đã chunk, đo khả năng trả về ngữ cảnh liên quan mà không phụ thuộc GraphRAG.

**Acceptance Scenarios**:

1. **Given** kho chunk đã được lập chỉ mục, **When** người dùng gửi câu hỏi, **Then** hệ thống trả về danh sách ngữ cảnh có mức liên quan cao nhất kèm tham chiếu nguồn.
2. **Given** câu hỏi không có dữ liệu liên quan trong kho tri thức, **When** RAG truy xuất, **Then** hệ thống trả kết quả rỗng có kiểm soát và trạng thái "không đủ ngữ cảnh" thay vì tạo trích dẫn sai.

---

### User Story 3 - Truy Xuất Quan Hệ Qua GraphRAG Song Song (Priority: P2)

Là người dùng cần trả lời câu hỏi có quan hệ nhiều bước, tôi muốn hệ thống chạy GraphRAG song song với RAG path để bổ sung các liên kết thực thể và quan hệ, giúp truy xuất đầy đủ hơn.

**Why this priority**: GraphRAG tăng độ bao phủ tri thức cho các câu hỏi đa thực thể nhưng vẫn phụ thuộc dữ liệu đã chuẩn hóa từ P1, vì vậy phù hợp mức ưu tiên P2 cùng với RAG.

**Independent Test**: Có thể kiểm thử độc lập bằng bộ câu hỏi yêu cầu suy luận quan hệ giữa nhiều thực thể, xác minh GraphRAG sinh được đường dẫn quan hệ và nguồn gốc dữ liệu.

**Acceptance Scenarios**:

1. **Given** truy vấn có nhiều thực thể liên quan, **When** hệ thống kích hoạt song song RAG path và GraphRAG path, **Then** hệ thống tạo được cả ngữ cảnh ngữ nghĩa lẫn kết quả quan hệ có thể hợp nhất.
2. **Given** GraphRAG không trả về kết quả hợp lệ, **When** quá trình song song kết thúc, **Then** hệ thống vẫn có thể tiếp tục với kết quả từ RAG path và ghi nhận trạng thái suy giảm một phần.

---

### User Story 4 - Lớp Phản Hồi LLM Có Trích Dẫn Nguồn (Priority: P2)

Là người dùng cuối, tôi muốn nhận câu trả lời rõ ràng, bám sát ngữ cảnh truy xuất, có trích dẫn nguồn và thông báo mức độ tin cậy để dễ kiểm chứng.

**Why this priority**: Đây là lớp hiển thị giá trị trực tiếp cho người dùng sau khi đã có dữ liệu truy xuất; thiếu lớp này thì hệ thống chưa hoàn thành trải nghiệm hỏi đáp.

**Independent Test**: Có thể kiểm thử độc lập bằng cách cấp ngữ cảnh đầu vào giả lập từ RAG/GraphRAG và đo chất lượng đầu ra phản hồi, trích dẫn, cũng như hành vi khi thiếu ngữ cảnh.

**Acceptance Scenarios**:

1. **Given** hệ thống nhận được ngữ cảnh truy xuất hợp lệ, **When** LLM response layer tạo phản hồi, **Then** câu trả lời phải kèm danh sách nguồn trích dẫn tương ứng với chunk/quan hệ đã dùng.
2. **Given** ngữ cảnh truy xuất không đủ hoặc mâu thuẫn, **When** hệ thống tạo phản hồi, **Then** hệ thống phải thông báo giới hạn thông tin và tránh khẳng định chắc chắn vượt ngoài bằng chứng.

---

### User Story 5 - Nhóm Chức Năng Phát Triển Tương Lai (Priority: P3)

Là product owner, tôi muốn có nhóm chức năng tương lai được định nghĩa rõ ranh giới với MVP để mở rộng có kiểm soát mà không làm ảnh hưởng luồng cốt lõi hiện tại.

**Why this priority**: Không trực tiếp tạo giá trị MVP tức thời nhưng rất quan trọng để bảo toàn khả năng mở rộng và giảm nợ sản phẩm ở giai đoạn tiếp theo.

**Independent Test**: Có thể kiểm thử độc lập bằng cách rà soát danh mục tính năng tương lai, điều kiện kích hoạt, và tác động dự kiến mà không cần triển khai ngay.

**Acceptance Scenarios**:

1. **Given** tài liệu đặc tả MVP đã hoàn thiện, **When** nhóm sản phẩm xác định nhóm tính năng tương lai, **Then** mỗi mục phải có mô tả mục tiêu, phụ thuộc, và tiêu chí kích hoạt rõ ràng.
2. **Given** có yêu cầu mới ngoài MVP, **When** đánh giá phạm vi, **Then** hệ thống phân loại được yêu cầu vào "MVP" hoặc "Future Group" dựa trên tiêu chí thống nhất.

### Edge Cases

- File PDF là bản scan ảnh không có lớp chữ: hệ thống phải phát hiện mức độ trích xuất văn bản thấp và cảnh báo tài liệu chưa đủ dữ liệu để chunk chất lượng cao.
- File DOCX có bảng, bullet nhiều cấp, hoặc section trống: hệ thống phải giữ được thứ tự nội dung và không tạo chunk rỗng.
- Người dùng import lại cùng một tài liệu đã có trước đó: hệ thống phải xử lý idempotent theo chính sách chống trùng lặp và không nhân bản dữ liệu vô ý.
- Một phần dữ liệu parse thành công nhưng chunking lỗi giữa chừng: hệ thống phải rollback hoặc đánh dấu trạng thái nhất quán để không phục vụ dữ liệu nửa vời.
- RAG path và GraphRAG path trả về kết quả mâu thuẫn: lớp phản hồi phải ưu tiên trích dẫn được kiểm chứng và nêu rõ mâu thuẫn thay vì hợp nhất mù.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Hệ thống MUST cho phép import tài liệu từ file PDF và DOCX trong môi trường local-first.
- **FR-002**: Hệ thống MUST chuẩn hóa văn bản sau parse thành một định dạng nội bộ thống nhất để phục vụ các bước xử lý tiếp theo.
- **FR-003**: Hệ thống MUST tạo chunk tuần tự từ văn bản đã chuẩn hóa, trong đó mỗi chunk có định danh duy nhất, thứ tự, và tham chiếu nguồn gốc tài liệu.
- **FR-004**: Hệ thống MUST lưu metadata ingest/parse/chunk tối thiểu gồm mã tài liệu, loại tệp, thời điểm xử lý, trạng thái, và thông tin lỗi nếu có.
- **FR-005**: Hệ thống MUST đảm bảo ingest pipeline có thể chạy lại an toàn cho cùng một tài liệu mà không tạo dữ liệu trùng không kiểm soát.
- **FR-006**: Hệ thống MUST từ chối hoặc gắn cờ các tài liệu không parse được, đồng thời cung cấp lý do lỗi để người dùng xử lý.
- **FR-007**: Hệ thống MUST hỗ trợ truy vấn RAG dựa trên kho chunk đã xử lý và trả về danh sách ngữ cảnh liên quan kèm điểm liên quan tương đối.
- **FR-008**: Hệ thống MUST chạy GraphRAG theo một path song song với RAG path cho cùng một truy vấn khi dữ liệu quan hệ sẵn sàng.
- **FR-009**: Hệ thống MUST mặc định chạy hybrid retrieval bằng cách kích hoạt song song RAG path và GraphRAG path cho mỗi truy vấn đủ điều kiện.
- **FR-010**: Hệ thống MUST fallback có kiểm soát sang path còn lại khi một path timeout hoặc lỗi, đồng thời ghi nhận trạng thái suy giảm trong truy vết phản hồi.
- **FR-011**: Hệ thống MUST tạo phản hồi LLM dựa trên ngữ cảnh đã truy xuất và đính kèm trích dẫn nguồn ở cấp chunk hoặc quan hệ thực thể.
- **FR-012**: Hệ thống MUST thông báo rõ khi bằng chứng không đủ để trả lời chắc chắn, thay vì tạo kết luận không có cơ sở.
- **FR-013**: Hệ thống MUST lưu nhật ký truy vết cho từng lượt hỏi đáp, bao gồm truy vấn, nguồn đã dùng, và trạng thái từng path.
- **FR-014**: Hệ thống MUST duy trì danh mục "Future Feature Group" tách biệt với phạm vi MVP, giới hạn ở 3 nhóm hậu v1 gồm: ingestion mở rộng định dạng, quản trị tri thức, và chất lượng/đánh giá phản hồi; mỗi mục có mô tả giá trị, phụ thuộc, và tiêu chí đưa vào roadmap.
- **FR-015**: Hệ thống MUST cung cấp cơ chế đánh dấu ưu tiên yêu cầu mới theo nhóm P1/P2/P3 để bảo toàn trọng tâm ingest-first của MVP.
- **FR-016**: Hệ thống MUST dùng Qdrant local cho vector retrieval index và Neo4j local cho graph persistence trong phạm vi MVP local-first.
- **FR-017**: Hệ thống MUST thực hiện orchestration truy xuất và hợp nhất kết quả từ Qdrant và Neo4j tại lớp ứng dụng ChatBox.
- **FR-018**: Hệ thống MUST đáp ứng SLA parse+chunk với p95 không vượt quá 45 giây cho mỗi tệp PDF/DOCX hợp lệ có kích thước đến 20MB trong điều kiện kiểm thử MVP chuẩn.
- **FR-019**: Hệ thống MUST đáp ứng SLA truy xuất và phản hồi với retrieval p95 không vượt quá 2.5 giây và full response p95 không vượt quá 8 giây.
- **FR-020**: Hệ thống MUST áp dụng chính sách chunking mặc định 500 tokens/chunk với overlap 80 tokens, ưu tiên tách theo semantic boundary (heading/paragraph) trước khi áp hard token cap.
- **FR-021**: Hệ thống MUST hợp nhất kết quả hybrid theo thứ tự ưu tiên bằng chứng có độ tin cậy nguồn cao hơn và có khả năng truy vết rõ hơn.

### Key Entities *(include if feature involves data)*

- **SourceDocument**: Tài liệu đầu vào do người dùng import; thuộc tính chính gồm document_id, file_type (PDF/DOCX), file_name, import_time, ingest_status.
- **ParsedContent**: Nội dung đã được trích xuất và chuẩn hóa từ SourceDocument; gồm normalized_text, structural_segments, parse_quality, parse_errors.
- **ContentChunk**: Đơn vị tri thức nhỏ phục vụ truy xuất; gồm chunk_id, document_id, chunk_order, chunk_text, source_pointer, token_estimate, overlap_tokens, boundary_type.
- **KnowledgeIndexEntry**: Đại diện dữ liệu được lập chỉ mục cho RAG path; gồm reference_to_chunk, relevance_features, index_status.
- **GraphNode**: Thực thể tri thức trong đồ thị; gồm node_id, entity_label, source_references, confidence_level.
- **GraphEdge**: Quan hệ giữa các GraphNode; gồm edge_id, relation_type, direction, evidence_references.
- **RetrievalResult**: Kết quả truy xuất cho một truy vấn; gồm query_id, rag_contexts, graph_paths, merge_status, degradation_flags.
- **LLMResponseRecord**: Bản ghi phản hồi cuối cùng; gồm response_id, answer_text, citations, uncertainty_note, response_time, audit_trace.
- **FutureFeatureItem**: Mục tính năng tương lai; gồm feature_key, group_type (ingestion_extension, knowledge_governance, response_quality_evaluation), objective, dependency_list, priority_level, activation_criteria.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Trong môi trường kiểm thử MVP tiêu chuẩn nội bộ, ingest + parse + chunk cho file PDF/DOCX hợp lệ đến 20MB đạt p95 <= 45 giây.
- **SC-002**: Tối thiểu 98% lượt import tài liệu hợp lệ tạo ra bộ chunk có thứ tự đúng và tham chiếu nguồn truy vết được.
- **SC-003**: Với bộ câu hỏi đánh giá nghiệp vụ đã gán nhãn, tối thiểu 85% câu trả lời có ít nhất một trích dẫn nguồn đúng với nội dung được hỏi.
- **SC-004**: Trong các truy vấn yêu cầu quan hệ đa thực thể, việc chạy song song RAG và GraphRAG cải thiện tỷ lệ trả lời đầy đủ ngữ cảnh ít nhất 20% so với chỉ dùng RAG path.
- **SC-005**: Tối thiểu 90% phản hồi trả về cho người dùng bao gồm trạng thái độ chắc chắn (đủ bằng chứng/thiếu bằng chứng) để hỗ trợ kiểm chứng.
- **SC-006**: 100% yêu cầu ngoài MVP được ghi vào Future Feature Group với đủ mục tiêu, phụ thuộc, và tiêu chí kích hoạt trước khi đưa vào sprint kế tiếp.
- **SC-007**: Độ trễ retrieval của truy vấn tri thức đạt p95 <= 2.5 giây trên bộ benchmark MVP chuẩn.
- **SC-008**: Thời gian full response (từ lúc gửi câu hỏi đến khi nhận câu trả lời hoàn chỉnh) đạt p95 <= 8 giây trên bộ benchmark MVP chuẩn.
- **SC-009**: Tối thiểu 99% truy vấn vẫn nhận được phản hồi có trích dẫn khi ít nhất một trong hai path (RAG hoặc GraphRAG) hoàn tất thành công.
- **SC-010**: 100% mục future được tạo sau v1 được phân loại đúng vào một trong 3 nhóm future đã khóa, không phát sinh nhóm mới trong phạm vi phiên bản đầu.

## Assumptions

- Người dùng mục tiêu là nhóm vận hành tri thức nội bộ cần chạy hệ thống trong môi trường cục bộ, ưu tiên bảo toàn dữ liệu tại máy hoặc mạng nội bộ.
- MVP chỉ tập trung vào tài liệu văn bản PDF/DOCX; các định dạng khác (XLSX, HTML, ảnh thuần) nằm ngoài phạm vi hiện tại.
- Quy trình cốt lõi cần hoạt động ngay cả khi không có kết nối internet, ngoại trừ các hoạt động quản trị không ảnh hưởng chức năng chính.
- Tổ chức đã có mô hình ngôn ngữ cục bộ phù hợp để tạo phản hồi từ ngữ cảnh truy xuất.
- Bộ dữ liệu kiểm thử chuẩn nội bộ (bao gồm câu hỏi gán nhãn và tài liệu mẫu) sẽ được cung cấp để đo các chỉ số thành công.
- Profile lưu trữ của MVP được cố định: Qdrant local cho vector store và Neo4j local cho graph store.
- Phạm vi future trong phiên bản đầu được khóa ở đúng 3 nhóm hậu v1 và không mở rộng thêm nhóm mới trước khi hoàn tất MVP.
