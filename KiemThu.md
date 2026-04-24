BÁO CÁO KIỂM THỬ (TEST CASES REPORT)
Dưới đây là bảng tổng hợp các Test Case dựa trên các kịch bản kiểm thử tự động (Unit Tests & Integration Tests) đã được xây dựng trong thư mục tests/ của dự án.
1. Module Chunking (test_chunker.py)
Mục đích: Kiểm tra tính đúng đắn của việc chia nhỏ văn bản (chunking) và bảo toàn siêu dữ liệu (metadata).
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-CHK-01
Chia văn bản dài
Đưa văn bản dài hơn chunk_size vào hàm chunk_text.
Trả về danh sách gồm nhiều hơn 1 chunk.
Pass
TC-CHK-02
Chia văn bản ngắn
Đưa văn bản ngắn hơn chunk_size vào hàm chunk_text.
Trả về đúng 1 chunk chứa nguyên văn chuỗi ban đầu.
Pass
TC-CHK-03
Xử lý văn bản rỗng
Đưa chuỗi rỗng "" vào hàm chunk_text.
Trả về danh sách rỗng [].
Pass
TC-CHK-04
Xử lý Overlap quá lớn
Đưa overlap >= chunk_size vào hàm chunk_text.
Tự động điều chỉnh overlap, không báo lỗi (không raise exception).
Pass
TC-CHK-05
Xử lý Chunk Size không hợp lệ
Đưa chunk_size <= 0 vào hàm chunk_text.
Báo lỗi ValueError.
Pass
TC-CHK-06
Bảo toàn Metadata
Đưa 1 Document có metadata page và source vào chunk_documents.
Tất cả các chunk sinh ra đều giữ nguyên metadata gốc.
Pass
TC-CHK-07
Đánh chỉ mục chunk (chunk_index)
Đưa 1 Document dài vào chunk_documents.
Các chunk sinh ra có chunk_index tăng dần (0, 1, 2...).
Pass
TC-CHK-08
Chỉ mục độc lập giữa các Document
Đưa 2 Document khác nhau vào chunk_documents.
Mỗi Document có chunk_index bắt đầu lại từ 0 một cách độc lập.
Pass
TC-CHK-09
Bảo toàn Metadata DOCX
Đưa Document có metadata paragraph vào chunk_documents.
Các chunk sinh ra giữ nguyên metadata paragraph.
Pass
TC-CHK-10
Bỏ qua Document rỗng
Đưa Document có page_content rỗng vào chunk_documents.
Document bị bỏ qua, không sinh ra chunk nào.
Pass
TC-CHK-11
Tuân thủ giới hạn Chunk Size
Đưa Document rất dài vào chunk_documents.
Không có chunk nào sinh ra có độ dài vượt quá chunk_size.
Pass
TC-CHK-12
Tạo thêm chunk khi có Overlap
So sánh số lượng chunk khi overlap=0 và overlap>0.
Số lượng chunk khi có overlap phải lớn hơn hoặc bằng khi không có.
Pass

2. Module Retriever (test_retriever.py)
Mục đích: Kiểm tra các chiến lược tìm kiếm (Vector, BM25, Hybrid) và thuật toán kết hợp RRF.
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-RET-01
Giới hạn kết quả BM25 (Top K)
Gọi bm25_retriever.invoke() với tham số k=3.
Trả về tối đa 3 documents.
Pass
TC-RET-02
Độ chính xác từ khóa BM25
Tìm kiếm từ khóa "BM25" bằng BM25 Retriever.
Document chứa cụm từ "thuật toán BM25" nằm trong top kết quả.
Pass
TC-RET-03
Bảo toàn Metadata trong BM25
Truy vấn BM25 Retriever.
Các Document trả về giữ nguyên metadata gốc (page, source).
Pass
TC-RET-04
Giới hạn kết quả Vector (Top K)
Gọi vector_retriever.invoke() với tham số k=2.
Trả về tối đa 2 documents.
Pass
TC-RET-05
Khởi tạo Hybrid Retriever
Gọi build_hybrid_retriever() kết hợp Vector và BM25.
Trả về đối tượng BaseRetriever hợp lệ, truy vấn trả về >= 1 kết quả.
Pass
TC-RET-06
Xử lý Trọng số (Weight) không hợp lệ
Khởi tạo Hybrid Retriever với bm25_weight = 1.5 hoặc -0.1.
Báo lỗi ValueError.
Pass
TC-RET-07
Xử lý Trọng số biên
Khởi tạo Hybrid Retriever với bm25_weight = 0.0 và 1.0.
Khởi tạo thành công, không báo lỗi.
Pass
TC-RET-08
Ảnh hưởng của Trọng số đến kết quả
Truy vấn cùng 1 câu hỏi với bm25_weight = 0.8 và 0.2.
Trả về kết quả hợp lệ, không bị crash (thứ hạng có thể thay đổi).
Pass

3. Module Pipeline (test_pipeline.py)
Mục đích: Kiểm tra luồng xử lý chính của RAG (tạo Index, xây dựng Prompt, tạo Citation).
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-PIP-01
Tạo RAG Index
Gọi build_index() với danh sách tài liệu mẫu.
Trả về đối tượng RAGIndex chứa vectorstore và chunks không rỗng.
Pass
TC-PIP-02
Prompt chứa Context
Truyền context vào hàm _build_prompt().
Chuỗi Prompt sinh ra chứa chính xác nội dung context.
Pass
TC-PIP-03
Prompt chứa Câu hỏi
Truyền question vào hàm _build_prompt().
Chuỗi Prompt sinh ra chứa chính xác nội dung question.
Pass
TC-PIP-04
Prompt xử lý Lịch sử rỗng
Truyền history_text="" vào hàm _build_prompt().
Bỏ qua phần header "Lịch sử hội thoại trước đó".
Pass
TC-PIP-05
Prompt chứa Lịch sử chat
Truyền history_text có nội dung vào hàm _build_prompt().
Prompt chứa phần "Lịch sử hội thoại" và nội dung tương ứng.
Pass
TC-PIP-06
Format Lịch sử hội thoại
Truyền danh sách tin nhắn vào _format_chat_history().
Format đúng tiền tố "Human:" và "AI:", giữ đúng thứ tự thời gian.
Pass
TC-PIP-07
Tạo danh sách Citation
Đưa danh sách Document vào _build_citation_list().
Trả về danh sách CitationSource với content và metadata khớp hoàn toàn.
Pass

4. Module Parsers (test_parsers.py)
Mục đích: Kiểm tra khả năng trích xuất văn bản từ các định dạng tài liệu (PDF, DOCX).
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-PAR-01
Trích xuất DOCX thành Documents
Đưa file DOCX tạm vào extract_documents_docx().
Trả về danh sách đối tượng Document (không phải chuỗi).
Pass
TC-PAR-02
Đánh dấu đoạn văn (Paragraph) DOCX
Đưa file DOCX có 3 đoạn văn vào parser.
Trả về 3 Document, metadata paragraph đánh số đúng thứ tự 1, 2, 3.
Pass
TC-PAR-03
Bỏ qua đoạn văn rỗng DOCX
Đưa file DOCX chứa các đoạn văn rỗng/khoảng trắng vào parser.
Các đoạn văn rỗng bị bỏ qua, chỉ lấy các đoạn có nội dung thực.
Pass
TC-PAR-04
Tên file trong Metadata
Đưa file DOCX vào parser.
Metadata source chỉ chứa tên file, không chứa đường dẫn tuyệt đối.
Pass

5. Module Models (test_models.py)
Mục đích: Kiểm tra tính hợp lệ của các cấu trúc dữ liệu (Pydantic Models).
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-MOD-01
Khởi tạo CitationSource
Tạo CitationSource với nội dung và metadata.
Khởi tạo thành công, các thuộc tính được gán đúng giá trị.
Pass
TC-MOD-02
Khởi tạo AskResponse
Tạo AskResponse với câu hỏi, câu trả lời và citations.
Khởi tạo thành công, chấp nhận danh sách citations rỗng hoặc có phần tử.
Pass
TC-MOD-03
Serialize AskResponse
Gọi .model_dump() trên đối tượng AskResponse.
Chuyển đổi thành công sang kiểu dict (phục vụ cho JSON response).
Pass

6. Module Lịch sử & Database (test_history_file_ids.py & test_history_search_mode_citations.py)
Mục đích: Kiểm tra việc lưu trữ và truy xuất lịch sử hội thoại từ PostgreSQL.
ID
Tên Test Case (Mục đích)
Các bước thực hiện (Input)
Kết quả mong đợi (Expected Output)
Trạng thái
TC-HIS-01
Lọc lịch sử theo 1 File ID
Truy vấn lịch sử chat với file_ids=[B].
Chỉ trả về các tin nhắn của riêng file B và các tin nhắn dùng chung file (A+B). Không lẫn tin nhắn của file A.
Pass
TC-HIS-02
Lọc lịch sử với File chưa có tin nhắn
Truy vấn lịch sử chat với file_ids=[C] (file mới).
Trả về danh sách rỗng (0 rows).
Pass
TC-HIS-03
Lọc lịch sử với File IDs rỗng
Truy vấn lịch sử chat với file_ids=[].
Không bị crash, trả về danh sách rỗng.
Pass
TC-HIS-04
Lọc lịch sử nhiều File IDs
Truy vấn lịch sử chat với file_ids=[A, B].
Trả về tất cả tin nhắn liên quan đến A, B, và A+B.
Pass
TC-HIS-05
Lưu Search Mode
Insert các tin nhắn với các mode: vector, hybrid, compare_vector, compare_hybrid.
Đọc từ DB ra chính xác các mode tương ứng cho từng tin nhắn.
Pass
TC-HIS-06
Lưu Citations (JSONB)
Insert tin nhắn kèm danh sách CitationSource.
Đọc từ DB ra danh sách Citations đúng cấu trúc (content, score, metadata).
Pass
TC-HIS-07
Lưu Citations rỗng
Insert tin nhắn với citations=[].
Đọc từ DB ra danh sách rỗng [].
Pass
TC-HIS-08
Lưu lịch sử cho endpoint /compare
Gọi endpoint /compare (mô phỏng).
Hệ thống tự động tạo và lưu đúng 2 tin nhắn riêng biệt: 1 cho compare_vector và 1 cho compare_hybrid với cùng câu hỏi.
Pass
TC-HIS-09
Tương thích ngược (Backward Compatibility)
Truy vấn các tin nhắn cũ trong DB (chưa có cột search_mode và citations).
Hệ thống tự động gán search_mode = 'unknown' và citations = [], không gây lỗi.
Pass


