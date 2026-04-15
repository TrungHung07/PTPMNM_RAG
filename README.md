# ChatBox Local RAG

Trợ lý tri thức **local-first** kết hợp **Parallel RAG + GraphRAG**, tối ưu cho việc demo/tái lập kết quả trên máy cá nhân và phát triển nhanh.

## Thành phần chính
- **Ingestion**: parse PDF/DOCX → chuẩn hoá → chunk.
- **Retrieval**: truy hồi vector + truy hồi graph, chạy song song và hợp nhất kết quả.
- **Orchestration**: adapter nhà cung cấp (ví dụ Ollama/vLLM) + các pattern streaming/fallback.
- **Storage**: lưu metadata bằng SQLite cho workflow local-first.

## Yêu cầu
- Python **3.11+**

## Cài đặt (dev)
```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Demo nhanh (CLI)
Repo có sẵn các file mẫu ở thư mục gốc:
- `sample.pdf`
- `sample.docx`
- `sample2.docx`

Import một tài liệu:
```bash
python -m chatbox.app.cli.import_cmd run --file-path sample.pdf --file-type pdf
```

Query (hybrid + streaming):
```bash
python -m chatbox.app.cli.query_cmd run "Summarize key points" --mode hybrid --stream true
```

Lưu ý: state local được lưu dưới `.chatbox/` và đã được **git-ignore**.

## API
Xem mô tả API tại `docs/api/chatbox-local-api.md`.

## Phát triển
Lint:
```bash
python -m ruff check src tests
```

Test:
```bash
python -m pytest -q
```

## Tài liệu
- Runbook: `docs/runbook/local-first-chatbox.md`
- Roadmap: `docs/roadmap/future-feature-groups.md`
- Specs: `specs/001-local-rag-chatbox/`
