# ChatBox Local API

## Endpoints

### POST /v1/documents:ingest

Request body:

- source_uri: string
- file_type: pdf | docx
- title: optional string

Response:

- document_id
- chunk_count
- ingest_status

### POST /v1/query/context

Request body:

- query_text: string
- mode: rag | hybrid
- top_k: integer

Response:

- query_id
- retrieval_context

### POST /v1/query

Request body:

- query_text: string
- mode: rag | hybrid
- top_k: integer
- stream: boolean

Response (non-stream):

- query_id
- answer_text
- citations
- uncertainty_flags
- provider_metadata
- retrieval_context

Response (stream=true):

- text/plain streaming chunks of answer_text
