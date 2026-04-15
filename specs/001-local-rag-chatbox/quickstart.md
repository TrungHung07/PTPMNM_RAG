# Quickstart: ChatBox Local-First Knowledge Assistant

## Setup

1. Create a Python 3.11 environment.
2. Install dependencies:
   - pip install -e .[dev]
3. Copy environment template:
   - copy .env.example .env

## Validate Project

1. Run unit, integration, and benchmark checks:
   - python -m pytest -q

## Import A Document

1. Run:
   - python -m chatbox.app.cli.import_cmd run --file-path sample.pdf --file-type pdf

## Query In Hybrid Mode

1. Run:
   - python -m chatbox.app.cli.query_cmd run "Summarize key points" --mode hybrid --stream true

## API Usage

1. Start API host by wiring routers with bootstrap container.
2. Call:
   - POST /v1/documents:ingest
   - POST /v1/query/context
   - POST /v1/query
