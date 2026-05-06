"""
Benchmark reranking via two endpoints:
  - POST /ask     (rerank enabled by default)
  - POST /ask-v2  (forces rerank_enabled=False server-side)

Goal:
  - Compare latency and retrieval quality for both retrieval modes:
      vector vs hybrid

Session / repeat semantics:
  - Trong **một** session: mỗi câu hỏi chỉ gọi **một lần** cho từng cấu hình
    (/ask + /ask-v2 × vector + hybrid = 4 request/câu).
  - `--repeats` = số **session độc lập**: mỗi lần **upload lại file** (session_id mới),
    rồi lặp lại toàn bộ bộ câu hỏi. Thống kê latency/quality gom qua các session.

Chất lượng retrieval (ưu tiên):
  - Nếu mỗi câu trong JSON có `gold_citations` (metadata chunk đúng), tính:
      - recall@k: trong top-k citation trả về, bao phủ bao nhiêu phần gold (|hit|/|gold|).
      - top1_chunk_hit: citation đầu tiên có khớp một gold (theo page/chunk_index/paragraph) không.
  - Nếu không có gold: fallback keyword trên nội dung citation so với `expected_answer`
    (proxy yếu; nên bổ sung gold theo metadata từ API sau một lần chạy thử).

No external dependencies (stdlib only), similar style to e2e_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_QUESTIONS_PATH = Path("tests/data.json")
DEFAULT_DATA_DIR = Path("data")


AskEndpoint = Literal["/ask", "/ask-v2"]
SearchMode = Literal["vector", "hybrid"]


@dataclass
class TimingStats:
    n: int
    avg_s: float
    p50_s: float
    p95_s: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def _timing_stats(samples_s: list[float]) -> TimingStats:
    if not samples_s:
        return TimingStats(n=0, avg_s=0.0, p50_s=0.0, p95_s=0.0)
    return TimingStats(
        n=len(samples_s),
        avg_s=float(statistics.fmean(samples_s)),
        p50_s=_percentile(samples_s, 0.50),
        p95_s=_percentile(samples_s, 0.95),
    )


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: int = 300,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = Request(url, data=data, method=method.upper(), headers=headers)
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def _multipart_form_data(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    filename = file_path.name
    ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    lines: list[bytes] = []
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
    )
    lines.append(f"Content-Type: {ctype}\r\n\r\n".encode())
    lines.append(file_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def upload_document(
    base_url: str,
    file_path: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    timeout_s: int = 600,
) -> tuple[str, list[str]]:
    """
    Upload tài liệu để lấy (session_id, file_ids).
    """
    url = f"{base_url}/upload?chunk_size={chunk_size}&chunk_overlap={chunk_overlap}"
    body, content_type = _multipart_form_data("files", file_path)
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    session_id = str(data.get("session_id") or "")
    files = data.get("files") or []
    file_ids = [str(x.get("file_id")) for x in files if x.get("file_id")]
    if not session_id or not file_ids:
        raise RuntimeError(f"Upload failed or unexpected response: {data}")
    return session_id, file_ids


def load_questions(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for it in items:
        q = str(it.get("question", "")).strip()
        exp = str(it.get("expected_answer", "")).strip()
        if not q:
            continue
        raw_gold = it.get("gold_citations")
        gold: list[dict[str, Any]] = []
        if isinstance(raw_gold, list):
            for g in raw_gold:
                if isinstance(g, dict):
                    gold.append(dict(g))
        out.append({"question": q, "expected_answer": exp, "gold_citations": gold})
    return out


def _meta_matches_gold(meta: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Gold chỉ ràng buộc các field có mặt và khác null."""
    for key in ("page", "paragraph", "chunk_index"):
        if key in gold and gold[key] is not None:
            if meta.get(key) != gold[key]:
                return False
    sc = gold.get("source_contains")
    if sc is not None and str(sc).strip():
        src = str(meta.get("source") or "")
        if str(sc) not in src:
            return False
    return True


def recall_at_k(
    citations: list[dict[str, Any]],
    gold_list: list[dict[str, Any]],
    *,
    k: int,
) -> float | None:
    """
    Recall@k theo chunk: tỉ lệ phần tử gold có ít nhất một citation trong top-k khớp metadata.
    None nếu không có gold.
    """
    if not gold_list:
        return None
    top = citations[: max(0, int(k))]
    metas = [(c or {}).get("metadata") or {} for c in top]
    hit = 0
    for gold in gold_list:
        if any(_meta_matches_gold(m, gold) for m in metas):
            hit += 1
    return hit / len(gold_list)


def top1_chunk_hit(
    citations: list[dict[str, Any]],
    gold_list: list[dict[str, Any]],
) -> float | None:
    """1.0 nếu citation đầu khớp một gold; None nếu không có gold."""
    if not gold_list or not citations:
        return None
    meta = (citations[0] or {}).get("metadata") or {}
    return 1.0 if any(_meta_matches_gold(meta, g) for g in gold_list) else 0.0


def dataset_has_gold(questions: list[dict[str, Any]]) -> bool:
    return any(bool(q.get("gold_citations")) for q in questions)


def _tokenize_expected(expected: str) -> list[str]:
    exp = (expected or "").lower()
    if not exp:
        return []
    cleaned = (
        exp.replace("\n", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("{", " ")
        .replace("}", " ")
        .replace("/", " ")
        .replace("\\", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    tokens = [w for w in cleaned.split() if len(w) > 3]
    if not tokens:
        tokens = [w for w in exp.split() if w]
    # Dedup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def keyword_hit_ratio(expected: str, retrieved_text: str) -> float:
    """
    Weak retrieval-quality proxy: fraction of expected keywords that appear
    in concatenated retrieved citations.
    """
    tokens = _tokenize_expected(expected)
    if not tokens:
        return 0.0
    hay = (retrieved_text or "").lower()
    hits = sum(1 for t in tokens if t in hay)
    return hits / len(tokens)


def citations_text(resp: dict[str, Any], *, k: int) -> str:
    cits = resp.get("citations") or []
    parts: list[str] = []
    for c in cits[: max(0, int(k))]:
        content = (c or {}).get("content") or ""
        if content:
            parts.append(str(content))
    return "\n".join(parts)


CONFIGS: list[dict[str, Any]] = [
    {"endpoint": "/ask", "search_mode": "vector"},
    {"endpoint": "/ask", "search_mode": "hybrid"},
    {"endpoint": "/ask-v2", "search_mode": "vector"},
    {"endpoint": "/ask-v2", "search_mode": "hybrid"},
]


def _warmup_session(
    base_url: str,
    *,
    session_id: str,
    file_ids: list[str],
    first_question: str,
    bm25_weight: float,
    rerank_threshold: float,
    warmup: int,
) -> None:
    """Đầu mỗi session: làm nóng cả 4 đường (/ask,/ask-v2 × vector,hybrid) trên câu đầu tiên."""
    if warmup <= 0:
        return
    for _ in range(warmup):
        for cfg in CONFIGS:
            endpoint: AskEndpoint = cfg["endpoint"]
            search_mode: SearchMode = cfg["search_mode"]
            url = f"{base_url}{endpoint}"
            payload = {
                "session_id": session_id,
                "file_ids": file_ids,
                "question": first_question,
                "search_mode": search_mode,
                "bm25_weight": bm25_weight,
                "rerank_enabled": True,
                "rerank_threshold": rerank_threshold,
            }
            _ = _http_json("POST", url, payload, timeout_s=600)


def run_single_session(
    *,
    base_url: str,
    session_id: str,
    file_ids: list[str],
    questions: list[dict[str, Any]],
    warmup: int,
    bm25_weight: float,
    rerank_threshold: float,
    citations_k: int,
    session_run_index: int,
) -> dict[str, Any]:
    """
    Một session: mỗi câu hỏi → đúng 4 request (không lặp lại cùng câu trong session).
    """
    if not questions:
        raise RuntimeError("questions empty")

    by_config: dict[str, Any] = {}
    per_question: list[dict[str, Any]] = []

    _warmup_session(
        base_url,
        session_id=session_id,
        file_ids=file_ids,
        first_question=questions[0]["question"],
        bm25_weight=bm25_weight,
        rerank_threshold=rerank_threshold,
        warmup=warmup,
    )

    for item in questions:
        q = item["question"]
        expected = item.get("expected_answer", "")
        gold = list(item.get("gold_citations") or [])

        per_q_row: dict[str, Any] = {
            "question": q,
            "expected_answer": expected,
            "gold_citations": gold,
            "session_run_index": session_run_index,
            "runs": [],
        }

        for cfg in CONFIGS:
            endpoint: AskEndpoint = cfg["endpoint"]
            search_mode: SearchMode = cfg["search_mode"]
            url = f"{base_url}{endpoint}"

            payload = {
                "session_id": session_id,
                "file_ids": file_ids,
                "question": q,
                "search_mode": search_mode,
                "bm25_weight": bm25_weight,
                "rerank_enabled": True,
                "rerank_threshold": rerank_threshold,
            }

            t0 = time.perf_counter()
            resp = _http_json("POST", url, payload, timeout_s=600)
            t1 = time.perf_counter()
            latency_s = t1 - t0
            cits = list(resp.get("citations") or [])
            kw_hit = keyword_hit_ratio(expected, citations_text(resp, k=citations_k))
            rec = recall_at_k(cits, gold, k=citations_k)
            t1hit = top1_chunk_hit(cits, gold)

            key = f"{endpoint}:{search_mode}"
            if key not in by_config:
                by_config[key] = {
                    "endpoint": endpoint,
                    "search_mode": search_mode,
                    "latency_s": [],
                    "hit_scores": [],
                    "recall_at_k": [],
                    "top1_chunk_hit": [],
                }
            by_config[key]["latency_s"].append(latency_s)
            by_config[key]["hit_scores"].append(kw_hit)
            if rec is not None:
                by_config[key]["recall_at_k"].append(float(rec))
            if t1hit is not None:
                by_config[key]["top1_chunk_hit"].append(float(t1hit))

            per_q_row["runs"].append(
                {
                    "endpoint": endpoint,
                    "search_mode": search_mode,
                    "latency_s": latency_s,
                    "keyword_hit_citations": kw_hit,
                    "recall_at_k": rec,
                    "top1_chunk_hit": t1hit,
                    "citations_sample": cits[: max(1, citations_k)],
                    "response_sample": resp,
                }
            )

        per_question.append(per_q_row)

    return {
        "session_run_index": session_run_index,
        "session_id": session_id,
        "file_ids": file_ids,
        "by_config": by_config,
        "per_question": per_question,
    }


def run_benchmark(
    *,
    base_url: str,
    questions_path: Path,
    session_runs: int,
    warmup: int,
    bm25_weight: float,
    rerank_threshold: float,
    citations_k: int,
    hit_threshold: float,
    file_path: Path | None,
    chunk_size: int,
    chunk_overlap: int,
    fixed_session_id: str | None,
    fixed_file_ids: list[str] | None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if not questions:
        raise RuntimeError(f"No questions found in {questions_path}")

    if fixed_session_id and fixed_file_ids is not None:
        if session_runs != 1:
            raise ValueError(
                "Khi dùng --session-id và --file-ids cố định, chỉ hỗ trợ 1 vòng; đặt --repeats 1."
            )
        sessions_payload: list[tuple[str, list[str]]] = [(fixed_session_id, fixed_file_ids)]
    else:
        if file_path is None or not file_path.exists():
            raise ValueError("Cần --file để upload khi không có --session-id/--file-ids.")
        sessions_payload = []
        for i in range(max(1, session_runs)):
            sid, fids = upload_document(
                base_url,
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            sessions_payload.append((sid, fids))

    merged_by_config: dict[str, Any] = {}
    per_session: list[dict[str, Any]] = []

    for run_idx, (session_id, file_ids) in enumerate(sessions_payload):
        part = run_single_session(
            base_url=base_url,
            session_id=session_id,
            file_ids=file_ids,
            questions=questions,
            warmup=warmup,
            bm25_weight=bm25_weight,
            rerank_threshold=rerank_threshold,
            citations_k=citations_k,
            session_run_index=run_idx,
        )
        per_session.append(
            {
                "session_run_index": run_idx,
                "session_id": session_id,
                "file_ids": file_ids,
                "per_question": part["per_question"],
            }
        )
        for key, bucket in part["by_config"].items():
            if key not in merged_by_config:
                merged_by_config[key] = {
                    "endpoint": bucket["endpoint"],
                    "search_mode": bucket["search_mode"],
                    "latency_s": [],
                    "hit_scores": [],
                    "recall_at_k": [],
                    "top1_chunk_hit": [],
                }
            merged_by_config[key]["latency_s"].extend(bucket["latency_s"])
            merged_by_config[key]["hit_scores"].extend(bucket["hit_scores"])
            merged_by_config[key]["recall_at_k"].extend(bucket.get("recall_at_k") or [])
            merged_by_config[key]["top1_chunk_hit"].extend(bucket.get("top1_chunk_hit") or [])

    has_gold = dataset_has_gold(questions)
    questions_with_gold = sum(1 for q in questions if q.get("gold_citations"))
    gold_partial = has_gold and questions_with_gold < len(questions)

    summary: dict[str, Any] = {}
    for key, bucket in merged_by_config.items():
        lat = list(bucket.get("latency_s") or [])
        hits = list(bucket.get("hit_scores") or [])
        recalls = list(bucket.get("recall_at_k") or [])
        top1s = list(bucket.get("top1_chunk_hit") or [])
        hit_rate = 0.0
        if hits:
            hit_rate = sum(1 for h in hits if float(h) >= float(hit_threshold)) / len(hits)
        recall_perfect = 0.0
        if recalls:
            recall_perfect = sum(1 for r in recalls if float(r) >= 1.0 - 1e-9) / len(recalls)
        row: dict[str, Any] = {
            "endpoint": bucket["endpoint"],
            "search_mode": bucket["search_mode"],
            "timing": _timing_stats(lat).__dict__,
            "avg_keyword_hit": float(statistics.fmean(hits)) if hits else 0.0,
            "hit_rate_keyword": float(hit_rate),
        }
        if recalls:
            row["avg_recall_at_k"] = float(statistics.fmean(recalls))
            row["recall_perfect_rate"] = float(recall_perfect)
            row["recall_eval_count"] = len(recalls)
        if top1s:
            row["avg_top1_chunk_hit"] = float(statistics.fmean(top1s))
        summary[key] = row

    return {
        "meta": {
            "base_url": base_url,
            "questions_path": str(questions_path),
            "question_count": len(questions),
            "session_runs": len(sessions_payload),
            "session_ids": [s for s, _ in sessions_payload],
            "upload_file": str(file_path) if file_path else None,
            "warmup": warmup,
            "bm25_weight": bm25_weight,
            "rerank_threshold": rerank_threshold,
            "citations_k": citations_k,
            "hit_threshold": hit_threshold,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "has_gold_citations": has_gold,
            "questions_with_gold": questions_with_gold,
            "gold_partial_coverage": gold_partial,
            "quality_primary": "recall_at_k_top1_chunk" if has_gold else "keyword_on_citations",
        },
        "by_config": merged_by_config,
        "per_session": per_session,
        "summary": summary,
    }


def write_markdown(out: dict[str, Any], path: Path) -> None:
    meta = out.get("meta") or {}
    summary = out.get("summary") or {}
    has_gold = bool(meta.get("has_gold_citations"))

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    lines: list[str] = []
    lines.append("# Rerank Endpoint Benchmark\n")
    lines.append("## Meta\n")
    lines.append(f"- base_url: `{meta.get('base_url','')}`")
    sids = meta.get("session_ids") or []
    lines.append(
        f"- session_runs: `{meta.get('session_runs', len(sids))}` (mỗi lần upload = session mới; trong session mỗi câu 4 request: /ask,/ask-v2 × vector,hybrid)"
    )
    if sids:
        lines.append(f"- session_ids: `{', '.join(str(x) for x in sids)}`")
    if meta.get("upload_file"):
        lines.append(f"- upload_file: `{meta.get('upload_file')}`")
    lines.append(f"- questions: `{meta.get('question_count', 0)}`; warmup (mỗi session): `{meta.get('warmup', 0)}`")
    if meta.get("gold_partial_coverage"):
        lines.append(
            f"- **Cảnh báo:** chỉ `{meta.get('questions_with_gold', 0)}/{meta.get('question_count', 0)}` câu có `gold_citations` — "
            "Recall@k chỉ trung bình trên các cặp (câu, request) có gold."
        )
    lines.append(f"- bm25_weight: `{meta.get('bm25_weight', 0.5)}`")
    lines.append(f"- citations_k (cho recall@k): `{meta.get('citations_k', 3)}`")
    if has_gold:
        lines.append(
            "- chất lượng: **Recall@k + top1_chunk** theo `gold_citations` (page/chunk_index/paragraph trong JSON câu hỏi)"
        )
    else:
        lines.append(
            "- chất lượng: **keyword trên citation** (proxy); thêm `gold_citations` vào JSON để đo recall chunk"
        )
    lines.append(f"- hit_threshold (keyword): `{meta.get('hit_threshold', 0.5)}`\n")

    lines.append("## Results\n")
    order = ["/ask:vector", "/ask:hybrid", "/ask-v2:vector", "/ask-v2:hybrid"]

    if has_gold:
        lines.append(
            "| Endpoint | Mode | n | avg (s) | p50 (s) | p95 (s) | avg_recall@k | recall_perfect_rate | avg_top1_chunk |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for key in order:
            row = summary.get(key) or {}
            t = row.get("timing") or {}
            lines.append(
                "| "
                + f"`{row.get('endpoint','')}` | `{row.get('search_mode','')}` | "
                + f"{t.get('n', 0)} | {fmt(float(t.get('avg_s', 0.0)))} | {fmt(float(t.get('p50_s', 0.0)))} | {fmt(float(t.get('p95_s', 0.0)))} | "
                + f"{fmt(float(row.get('avg_recall_at_k', 0.0)))} | {fmt(float(row.get('recall_perfect_rate', 0.0)))} | "
                + f"{fmt(float(row.get('avg_top1_chunk_hit', 0.0)))} |"
            )
        lines.append("\n### Keyword-on-citations (tham khảo)\n")
        lines.append("| Endpoint | Mode | avg_keyword_hit | hit_rate_keyword |")
        lines.append("|---|---|---:|---:|")
        for key in order:
            row = summary.get(key) or {}
            lines.append(
                f"| `{row.get('endpoint','')}` | `{row.get('search_mode','')}` | "
                f"{fmt(float(row.get('avg_keyword_hit', 0.0)))} | {fmt(float(row.get('hit_rate_keyword', 0.0)))} |"
            )
    else:
        lines.append("| Endpoint | Mode | n | avg (s) | p50 (s) | p95 (s) | avg_keyword_hit@k | hit_rate_keyword |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for key in order:
            row = summary.get(key) or {}
            t = row.get("timing") or {}
            lines.append(
                "| "
                + f"`{row.get('endpoint','')}` | `{row.get('search_mode','')}` | "
                + f"{t.get('n', 0)} | {fmt(float(t.get('avg_s', 0.0)))} | {fmt(float(t.get('p50_s', 0.0)))} | {fmt(float(t.get('p95_s', 0.0)))} | "
                + f"{fmt(float(row.get('avg_keyword_hit', 0.0)))} | {fmt(float(row.get('hit_rate_keyword', 0.0)))} |"
            )

    def _get_avg_s(k: str) -> float:
        return float(((summary.get(k) or {}).get("timing") or {}).get("avg_s") or 0.0)

    def _get_kw(k: str) -> float:
        return float((summary.get(k) or {}).get("avg_keyword_hit") or 0.0)

    def _get_rec(k: str) -> float:
        return float((summary.get(k) or {}).get("avg_recall_at_k") or 0.0)

    def _get_t1(k: str) -> float:
        return float((summary.get(k) or {}).get("avg_top1_chunk_hit") or 0.0)

    lines.append("\n## Deltas (ask vs ask-v2)\n")
    for mode in ["vector", "hybrid"]:
        k1 = f"/ask:{mode}"
        k2 = f"/ask-v2:{mode}"
        d_lat = _get_avg_s(k1) - _get_avg_s(k2)
        if has_gold:
            d_rec = _get_rec(k1) - _get_rec(k2)
            d_t1 = _get_t1(k1) - _get_t1(k2)
            lines.append(
                f"- `{mode}`: avg_latency Δ = **{d_lat:+.3f}s**; avg_recall@k Δ = **{d_rec:+.3f}**; avg_top1_chunk Δ = **{d_t1:+.3f}**"
            )
        else:
            d_kw = _get_kw(k1) - _get_kw(k2)
            lines.append(f"- `{mode}`: avg_latency Δ = **{d_lat:+.3f}s**; avg_keyword_hit@k Δ = **{d_kw:+.3f}**")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _pick_default_file() -> Path:
    # Prefer newest file in ./data (pdf/docx/txt), fallback to repo-wide PDF.
    candidates: list[Path] = []
    if DEFAULT_DATA_DIR.exists():
        for ext in ("*.pdf", "*.docx", "*.txt"):
            candidates.extend(sorted(DEFAULT_DATA_DIR.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True))
    if candidates:
        return candidates[0]

    root = Path(".")
    pdfs = sorted(root.glob("**/*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pdfs:
        return pdfs[0]
    raise FileNotFoundError("No document found. Provide --file <path> or put a file in ./data")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark /ask (rerank) vs /ask-v2 (no rerank). "
            "--repeats = số session độc lập (upload lại file mỗi lần); "
            "trong một session mỗi câu chỉ gọi 4 lần (vector+hybrid × 2 endpoint)."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Số lần upload + session mới (mặc định 5). Với --session-id cố định phải dùng 1.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Mỗi session: số vòng warmup (câu đầu × cả 4 cấu hình).",
    )
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rerank-threshold", type=float, default=0.0)
    parser.add_argument("--citations-k", type=int, default=3)
    parser.add_argument("--hit-threshold", type=float, default=0.5)

    # Either provide existing session/file_ids, or upload a doc.
    parser.add_argument("--session-id", default="")
    parser.add_argument("--file-ids", default="", help="Comma-separated file_ids (same session)")
    parser.add_argument("--file", default="", help="Path to PDF/DOCX/TXT to upload (if no session-id provided)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)

    args = parser.parse_args()

    base_url = str(args.base_url or DEFAULT_BASE_URL).rstrip("/")
    questions_path = Path(args.questions)

    session_id = str(args.session_id or "").strip()
    file_ids_arg = str(args.file_ids or "").strip()
    file_ids: list[str] = [x.strip() for x in file_ids_arg.split(",") if x.strip()] if file_ids_arg else []
    session_runs = max(1, int(args.repeats))
    warmup = max(0, int(args.warmup))

    try:
        file_arg = str(args.file or "").strip()
        file_path: Path | None = Path(file_arg) if file_arg else None

        if session_id:
            if not file_ids:
                raise ValueError("Khi dùng --session-id cần --file-ids (danh sách id, cách nhau bởi dấu phẩy).")
            if session_runs != 1:
                raise ValueError("Với session cố định, chỉ hỗ trợ --repeats 1 (không upload lại).")
            if file_path is not None:
                raise ValueError("Không dùng đồng thời --session-id và --file; bỏ --file khi test session có sẵn.")
            out = run_benchmark(
                base_url=base_url,
                questions_path=questions_path,
                session_runs=1,
                warmup=warmup,
                bm25_weight=float(args.bm25_weight),
                rerank_threshold=float(args.rerank_threshold),
                citations_k=max(1, int(args.citations_k)),
                hit_threshold=float(args.hit_threshold),
                file_path=None,
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
                fixed_session_id=session_id,
                fixed_file_ids=file_ids,
            )
        else:
            if file_path is None:
                file_path = _pick_default_file()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            out = run_benchmark(
                base_url=base_url,
                questions_path=questions_path,
                session_runs=session_runs,
                warmup=warmup,
                bm25_weight=float(args.bm25_weight),
                rerank_threshold=float(args.rerank_threshold),
                citations_k=max(1, int(args.citations_k)),
                hit_threshold=float(args.hit_threshold),
                file_path=file_path,
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
                fixed_session_id=None,
                fixed_file_ids=None,
            )
    except (URLError, HTTPError) as exc:
        print(f"[ERROR] Cannot reach API at {base_url}: {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] Benchmark failed: {exc}")
        return 1

    Path("rerank_endpoint_benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(out, Path("rerank_endpoint_benchmark.md"))
    print("[OK] Wrote rerank_endpoint_benchmark.json, rerank_endpoint_benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

