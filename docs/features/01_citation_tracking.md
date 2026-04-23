# [Feature 1] Citation / Source Tracking

**Branch:** `feature/citation-tracking`
**Commit:** `50eaac4`
**Ngày:** 2026-04-17

### Mục tiêu

Hệ thống RAG ban đầu chỉ trả về câu trả lời thuần văn bản từ LLM mà không
có thông tin nguồn gốc. Người dùng không thể biết AI đang lấy thông tin từ
trang nào, đoạn nào của tài liệu gốc.

Tính năng này bổ sung khả năng **trích dẫn nguồn** (citation tracking):
mỗi câu trả lời đi kèm danh sách các đoạn văn gốc đã được dùng làm căn cứ,
kèm thông tin vị trí chính xác trong tài liệu.

### Kiến trúc thay đổi

```
Trước (v1):
  File → extract_text() → str → chunk_text() → list[str]
       → FAISS (không có metadata) → ask_question() → str answer

Sau (v2):
  File → extract_documents() → list[Document(content, metadata)]
       → chunk_documents() → list[Document(content, {page, source, chunk_index})]
       → FAISS (có metadata) → ask_question() → AskResponse(answer, citations)
```

### Files thay đổi

| File | Loại | Mô tả |
|---|---|---|
| `src/parsers/pdf_parser.py` | MODIFY | Thêm `extract_documents_pdf()` trả về `list[Document]` với `metadata.page` + `metadata.source` |
| `src/parsers/docx_parser.py` | MODIFY | Thêm `extract_documents_docx()` trả về `list[Document]` với `metadata.paragraph` + `metadata.source` |
| `src/parsers/__init__.py` | MODIFY | Export thêm các hàm `extract_documents_*` mới |
| `src/chunking/text_chunker.py` | MODIFY | Thêm `chunk_documents()` — chia nhỏ Document nhưng giữ nguyên toàn bộ metadata gốc + bổ sung `chunk_index` |
| `src/rag/vectorstore.py` | MODIFY | Nhận `list[Document]` trực tiếp thay vì `list[str]`, bảo toàn metadata vào FAISS index |
| `src/rag/pipeline.py` | MODIFY | Thêm `build_vectorstore_from_documents()`, `ask_question()` trả về `AskResponse` thay vì `str` |
| `src/models.py` | **NEW** | Định nghĩa `CitationSource` và `AskResponse` Pydantic models dùng chung |
| `app.py` | MODIFY | Dùng parsers mới, endpoint `/ask` trả về `AskResponse` đầy đủ citations |

### Quyết định thiết kế

- **Backward compatible:** Mọi hàm cũ (`extract_text_pdf`, `extract_text_docx`,
  `chunk_text`, `build_vectorstore_from_text`) đều được giữ nguyên để code
  không bị breaking change.
- **DOCX không có số trang:** Định dạng DOCX không có khái niệm phân trang rõ ràng.
  Thay vì dùng số trang ước tính, hệ thống lưu `metadata.paragraph` — số thứ tự
  đoạn văn *có nội dung* (bỏ qua đoạn rỗng), bắt đầu từ 1.
- **Source filename:** Cả PDF và DOCX đều lưu `metadata.source = path.name` để
  hiển thị tên file trong citation, tiện cho trường hợp sau này hỗ trợ multi-file.
- **`chunk_index`:** Khi một trang/đoạn văn bị chia thành nhiều chunk, mỗi chunk
  được đánh số `chunk_index` bắt đầu từ 0, độc lập với từng Document nguồn.

### API Response sau khi nâng cấp

```json
POST /ask
{
  "file_id": "abc-123",
  "question": "Chính sách bảo mật dữ liệu là gì?"
}

→ Response:
{
  "question": "Chính sách bảo mật dữ liệu là gì?",
  "answer": "Theo tài liệu, dữ liệu người dùng được mã hóa AES-256...",
  "citations": [
    {
      "content": "Dữ liệu người dùng được mã hóa AES-256 và lưu trữ nội bộ...",
      "metadata": { "page": 12, "source": "policy_2024.pdf", "chunk_index": 0 }
    },
    {
      "content": "Không có thông tin nào được chia sẻ với bên thứ ba...",
      "metadata": { "page": 13, "source": "policy_2024.pdf", "chunk_index": 1 }
    }
  ]
}
```

### Kiểm thử

**25 unit tests** — PASSED (0.55s), không cần LLM hay database.

| Test file | Tests | Phạm vi kiểm thử |
|---|---|---|
| `tests/test_chunker.py` | 13 | `chunk_text()` (legacy) + `chunk_documents()` (metadata preservation, chunk_index, edge cases) |
| `tests/test_parsers.py` | 6 | `extract_documents_docx()` (paragraph metadata, bỏ qua đoạn rỗng, backward compat) |
| `tests/test_models.py` | 6 | `CitationSource`, `AskResponse` (tạo object, validate, serialize sang dict) |

**Test quan trọng nhất — `test_metadata_preserved`:**
```python
# Đầu vào: Document dài (2500 ký tự), trang 3
doc = Document(page_content="a" * 2500, metadata={"page": 3, "source": "test.pdf"})

# Sau chunk_documents(chunk_size=1000) → 3 chunks
# Mỗi chunk phải biết nó từ trang 3
assert chunk.metadata["page"] == 3          # ✅
assert chunk.metadata["source"] == "test.pdf" # ✅
```
