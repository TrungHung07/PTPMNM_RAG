# Project LLMs-Rag-Agent

Project tối giản để đọc văn bản từ file PDF/DOCX và chia nội dung thành các chunk có độ dài cố định để phục vụ các thử nghiệm RAG.

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Cách chạy

Ví dụ với file PDF:

```bash
python app.py data/sample.pdf --chunk-size 1000 --overlap 200
```

Ví dụ với file DOCX:

```bash
python app.py sample.docx --chunk-size 800 --overlap 100 --output chunks.json
```

## Tham số CLI

- `file`: đường dẫn tới file `.pdf` hoặc `.docx`.
- `--chunk-size`: số ký tự tối đa của mỗi chunk, mặc định `1000`.
- `--overlap`: số ký tự overlap giữa hai chunk liên tiếp, mặc định `200`.
- `--pdf-backend`: chọn backend đọc PDF là `pdfplumber` hoặc `pypdf`.
- `--output`: đường dẫn file output. Đuôi `.json` sẽ lưu JSON, còn lại lưu text.

## Sample files

Repo giữ lại các file mẫu sau:

- `data/sample.pdf`
- `sample.docx`
- `sample2.docx`
- `sample3.docx`
