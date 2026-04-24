I. GIỚI THIỆU
A. Bối cảnh đề tài
B. Mục tiêu dự án
C. Phạm vi và bài toán

II. CƠ SỞ LÝ THUYẾT
A. RAG (Retrieval-Augmented Generation)
B. Text Embedding và Vector Database (FAISS)
C. Large Language Models (Qwen2.5, Ollama)
D. LangChain Framework
E. Hybrid Search, RRF, Reranking và Citation Tracking

III. THIẾT KẾ HỆ THỐNG
A. Kiến trúc tổng quan
B. Luồng xử lý tài liệu
C. Luồng xử lý câu hỏi
D. Thiết kế các thành phần mã nguồn
E. Prompt Engineering và quản lý hội thoại

IV. TRIỂN KHAI HỆ THỐNG
A. Công nghệ sử dụng
B. Cài đặt môi trường (local, Docker)
C. Cấu hình mô hình và tham số
D. Cơ sở dữ liệu và quản lý lịch sử hội thoại
E. API chính và giao diện sử dụng

V. KIỂM THỬ VÀ ĐÁNH GIÁ
A. Chiến lược kiểm thử
1. Phạm vi kiểm thử (Unit, Integration)
2. Công cụ và dữ liệu kiểm thử

B. Module Chunking
1. Tính đúng đắn chia chunk (TC-CHK-01 đến TC-CHK-05)
2. Bảo toàn metadata (TC-CHK-06 đến TC-CHK-10)
3. Giới hạn chunk size và overlap (TC-CHK-11 đến TC-CHK-12)

C. Module Retriever
1. BM25 Retriever (TC-RET-01 đến TC-RET-03)
2. Vector Retriever (TC-RET-04)
3. Hybrid Retriever và trọng số/RRF (TC-RET-05 đến TC-RET-08)

D. Module Pipeline
1. Tạo RAG Index (TC-PIP-01)
2. Prompt và lịch sử hội thoại (TC-PIP-02 đến TC-PIP-06)
3. Citation tracking (TC-PIP-07)

E. Module Parsers
1. Trích xuất DOCX/PDF và metadata (TC-PAR-01 đến TC-PAR-04)

F. Module Models
1. Kiểm thử Pydantic models (TC-MOD-01 đến TC-MOD-03)

G. Module Lịch sử và Database
1. Lọc lịch sử theo file IDs (TC-HIS-01 đến TC-HIS-04)
2. Lưu search mode và citations (TC-HIS-05 đến TC-HIS-07)
3. Tương thích ngược và luồng compare (TC-HIS-08 đến TC-HIS-09)

H. Benchmark và phân tích kết quả
1. Benchmark chunking
2. So sánh vector, hybrid, rerank
3. Hạn chế hiện tại

VI. HƯỚNG DẪN SỬ DỤNG
A. Quy trình nạp tài liệu
B. Quy trình đặt câu hỏi
C. Quản lý session và khôi phục dữ liệu

VII. ĐỊNH HƯỚNG PHÁT TRIỂN
A. Cải thiện độ chính xác truy xuất
B. Mở rộng hiệu năng và triển khai thực tế
C. Nâng cấp tính năng sản phẩm

VIII. KẾT LUẬN
A. Tổng kết kết quả đạt được
B. Giá trị ứng dụng

IX. TÀI LIỆU THAM KHẢO
A. Tài liệu học thuật
B. Tài liệu kỹ thuật và mã nguồn tham chiếu