"""
src/graph_rag/extractor.py — Trích xuất knowledge triples từ chunk văn bản.

Dùng LLM để sinh JSON list triple. Fail-safe: lỗi → return [].
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document

from src.graph_rag.types import Triple

_logger = logging.getLogger("uvicorn.error")

# ─── Prompt ──────────────────────────────────────────────────────────────────

_EXTRACT_TRIPLES_PROMPT = """\
Bạn là một hệ thống trích xuất tri thức. Nhiệm vụ của bạn là đọc đoạn văn bản dưới đây \
và trích xuất tất cả các quan hệ có ý nghĩa dưới dạng triple (Subject, Relation, Object).

QUY TẮC NGHIÊM NGẶT:
1. Chỉ trả về JSON, KHÔNG giải thích, KHÔNG thêm text ngoài JSON.
2. Mỗi triple phải có đủ 3 trường: "s" (subject), "r" (relation), "o" (object).
3. Subject và Object là thực thể, khái niệm hoặc danh từ cụ thể.
4. Relation là động từ hoặc cụm động từ mô tả quan hệ giữa S và O.
5. Tối đa {max_triples} triple từ đoạn văn này.
6. Nếu không trích xuất được triple nào, trả về: {{"triples": []}}

Metadata của đoạn văn:
- Nguồn tài liệu: {source}
- Trang/đoạn: {page}

Đoạn văn bản:
\"\"\"
{chunk_text}
\"\"\"

Trả về JSON theo đúng format sau (chỉ JSON, không gì khác):
{{"triples":[{{"s":"<subject>","r":"<relation>","o":"<object>"}}]}}"""


def _parse_triples_json(raw: str) -> list[dict[str, str]]:
    """
    Parse JSON trả về từ LLM. Robust: thử json.loads trực tiếp,
    nếu fail thì tìm đoạn JSON bằng regex.
    """
    raw = raw.strip()

    # Thử parse trực tiếp
    try:
        data = json.loads(raw)
        return data.get("triples", [])
    except json.JSONDecodeError:
        pass

    # Tìm block JSON trong output bị bọc text
    match = re.search(r'\{.*"triples".*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("triples", [])
        except json.JSONDecodeError:
            pass

    return []


def extract_triples_from_chunk(
    llm: Any,
    doc: Document,
    max_triples: int = 10,
) -> list[Triple]:
    """
    Trích xuất list Triple từ một Document chunk bằng LLM.

    Args:
        llm: LLM instance (e.g. Ollama) đã được khởi tạo.
        doc: Document chunk có page_content và metadata.
        max_triples: Số triple tối đa cần trích xuất từ chunk này.

    Returns:
        List[Triple]. Trả về [] nếu LLM fail hoặc output không parse được.
    """
    metadata = doc.metadata or {}
    source = metadata.get("source", "unknown")
    page = metadata.get("page", metadata.get("paragraph", "?"))
    chunk_text = doc.page_content.strip()

    if not chunk_text:
        return []

    prompt = _EXTRACT_TRIPLES_PROMPT.format(
        chunk_text=chunk_text,
        source=source,
        page=page,
        max_triples=max_triples,
    )

    try:
        raw = llm.invoke(prompt)
        raw_str = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        items = _parse_triples_json(raw_str)
    except Exception as exc:
        _logger.warning("[GraphRAG] extract_triples_from_chunk error: %s", exc)
        return []

    triples: list[Triple] = []
    for item in items:
        s = str(item.get("s", "")).strip()
        r = str(item.get("r", "")).strip()
        o = str(item.get("o", "")).strip()
        if s and r and o:
            triples.append(Triple(s=s, r=r, o=o))

    return triples
