# Local Demo Runbook

## Prerequisites

1. Python 3.11+
2. Virtual environment activated
3. Project dependencies installed with development extras

## Demo Flow

1. Run tests: python -m pytest -q
2. Import a document with CLI:
   - python -m chatbox.app.cli.import_cmd run --file-path sample.pdf --file-type pdf
3. Query with hybrid mode and streaming:
   - python -m chatbox.app.cli.query_cmd run "What does the document say?" --mode hybrid --stream true

## Troubleshooting

1. If parser dependencies are missing, install pypdf and python-docx.
2. If SQLite errors occur, remove .chatbox/chatbox.db and rerun import.
3. If response generation fails, verify fallback mode is set in .env.
