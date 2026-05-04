"""
End-to-end benchmark (bao gồm LLM) cho PTPMNM_RAG.

Mục tiêu:
  - Upload tài liệu 1 lần để lấy session_id + file_id
  - Gọi /compare (vector vs hybrid) và /compare-rag (standard vs graph)
  - Đo round-trip time (E2E) bằng time.perf_counter() tại client
  - Xuất:
      - e2e_results.json  (raw)
      - e2e_results.md    (human-readable)
      - e2e_autogen.tex   (macro + plot coords để report_cursor.tex tự cập nhật)

Không dùng thư viện ngoài (requests/httpx) để chạy được ngay với requirements hiện tại.
"""

from __future__ import annotations

import json
import mimetypes
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


DEFAULT_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_QUESTIONS_PATH = Path("tests/benchmark_data.json")
DEFAULT_DATA_DIR = Path("data")


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


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout_s: int = 300) -> dict[str, Any]:
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
) -> tuple[str, str]:
    """
    Upload tài liệu để lấy (session_id, file_id).
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
    if not session_id or not files or not files[0].get("file_id"):
        raise RuntimeError(f"Upload failed or unexpected response: {data}")
    file_id = str(files[0]["file_id"])
    return session_id, file_id


def load_questions(path: Path) -> list[dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for it in items:
        q = str(it.get("question", "")).strip()
        exp = str(it.get("expected_answer", "")).strip()
        if q:
            out.append({"question": q, "expected_answer": exp})
    return out


def keyword_hit_ratio(expected: str, answer: str) -> float:
    exp = (expected or "").lower()
    ans = (answer or "").lower()
    if not exp:
        return 0.0
    tokens = [w for w in exp.replace(",", " ").replace(".", " ").split() if len(w) > 3]
    if not tokens:
        tokens = [w for w in exp.split() if w]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in ans)
    return hits / len(tokens)


def run_e2e(
    *,
    base_url: str,
    file_path: Path,
    questions_path: Path,
    repeats: int = 5,
    warmup: int = 1,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    bm25_weight: float = 0.5,
    rerank_enabled: bool = True,
    rerank_threshold: float = 0.0,
) -> dict[str, Any]:
    # 0) Upload
    session_id, file_id = upload_document(
        base_url,
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    questions = load_questions(questions_path)
    if not questions:
        raise RuntimeError(f"No questions found in {questions_path}")

    compare_url = f"{base_url}/compare"
    compare_rag_url = f"{base_url}/compare-rag"

    vector_lat_s: list[float] = []
    hybrid_lat_s: list[float] = []
    std_lat_s: list[float] = []
    graph_lat_s: list[float] = []

    vector_hit: list[float] = []
    hybrid_hit: list[float] = []
    std_hit: list[float] = []
    graph_hit: list[float] = []

    per_question: list[dict[str, Any]] = []

    for qi, item in enumerate(questions):
        q = item["question"]
        expected = item.get("expected_answer", "")

        payload_common = {
            "session_id": session_id,
            "file_ids": [file_id],
            "question": q,
            "bm25_weight": bm25_weight,
            "rerank_enabled": rerank_enabled,
            "rerank_threshold": rerank_threshold,
        }

        # Warm-up: gọi nhẹ /compare-rag (nặng nhất) để nạp model và cache.
        for _ in range(max(0, warmup)):
            _ = _http_json("POST", compare_rag_url, payload_common, timeout_s=600)

        # 1) /compare: vector vs hybrid
        q_vec_s: list[float] = []
        q_hyb_s: list[float] = []
        last_compare: dict[str, Any] | None = None
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            res = _http_json("POST", compare_url, payload_common, timeout_s=600)
            t1 = time.perf_counter()
            last_compare = res
            e2e = t1 - t0
            # /compare chạy cả 2 mode trong 1 request → coi như cùng round-trip
            # nhưng để plot, ta gán cùng một giá trị cho cả 2 (đánh giá fairness E2E)
            q_vec_s.append(e2e)
            q_hyb_s.append(e2e)

        vector_lat_s.extend(q_vec_s)
        hybrid_lat_s.extend(q_hyb_s)

        # 2) /compare-rag: standard vs graph (song song)
        q_std_s: list[float] = []
        q_grp_s: list[float] = []
        last_compare_rag: dict[str, Any] | None = None
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            res = _http_json("POST", compare_rag_url, payload_common, timeout_s=600)
            t1 = time.perf_counter()
            last_compare_rag = res
            e2e = t1 - t0
            # /compare-rag cũng trả 2 kết quả trong 1 request → gán cùng E2E
            q_std_s.append(e2e)
            q_grp_s.append(e2e)

        std_lat_s.extend(q_std_s)
        graph_lat_s.extend(q_grp_s)

        # Quality (simple): keyword hit ratio on ANSWER vs expected_answer
        try:
            v_ans = ((last_compare or {}).get("vector_result") or {}).get("answer") or ""
            h_ans = ((last_compare or {}).get("hybrid_result") or {}).get("answer") or ""
            s_ans = ((last_compare_rag or {}).get("standard_result") or {}).get("answer") or ""
            g_ans = ((last_compare_rag or {}).get("graph_result") or {}).get("answer") or ""
        except Exception:
            v_ans = h_ans = s_ans = g_ans = ""

        vector_hit.append(keyword_hit_ratio(expected, v_ans))
        hybrid_hit.append(keyword_hit_ratio(expected, h_ans))
        std_hit.append(keyword_hit_ratio(expected, s_ans))
        graph_hit.append(keyword_hit_ratio(expected, g_ans))

        per_question.append(
            {
                "index": qi + 1,
                "question": q,
                "expected_answer": expected,
                "compare_sample": last_compare,
                "compare_rag_sample": last_compare_rag,
                "e2e_s_compare_samples": q_vec_s,  # same as q_hyb_s
                "e2e_s_compare_rag_samples": q_std_s,  # same as q_grp_s
            }
        )

    result: dict[str, Any] = {
        "meta": {
            "base_url": base_url,
            "file_path": str(file_path),
            "session_id": session_id,
            "file_id": file_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "repeats": repeats,
            "warmup": warmup,
            "bm25_weight": bm25_weight,
            "rerank_enabled": rerank_enabled,
            "rerank_threshold": rerank_threshold,
            "question_count": len(questions),
        },
        "latency": {
            "vector": _timing_stats(vector_lat_s).__dict__,
            "hybrid": _timing_stats(hybrid_lat_s).__dict__,
            "standard": _timing_stats(std_lat_s).__dict__,
            "graph": _timing_stats(graph_lat_s).__dict__,
        },
        "quality": {
            "vector_hit_avg": float(statistics.fmean(vector_hit)) if vector_hit else 0.0,
            "hybrid_hit_avg": float(statistics.fmean(hybrid_hit)) if hybrid_hit else 0.0,
            "standard_hit_avg": float(statistics.fmean(std_hit)) if std_hit else 0.0,
            "graph_hit_avg": float(statistics.fmean(graph_hit)) if graph_hit else 0.0,
        },
        "per_question": per_question,
    }
    return result


def write_markdown(out: dict[str, Any], path: Path) -> None:
    lat = out["latency"]
    q = out["quality"]
    meta = out["meta"]

    def fmt(s: float) -> str:
        return f"{s:.3f}"

    lines: list[str] = []
    lines.append("# E2E Benchmark Results (includes LLM)\n")
    lines.append("## Meta\n")
    lines.append(f"- base_url: `{meta['base_url']}`")
    lines.append(f"- file_path: `{meta['file_path']}`")
    lines.append(f"- chunk_size/overlap: `{meta['chunk_size']}/{meta['chunk_overlap']}`")
    lines.append(f"- questions: `{meta['question_count']}`; repeats: `{meta['repeats']}`; warmup: `{meta['warmup']}`\n")

    lines.append("## Latency (round-trip seconds)\n")
    lines.append("| Mode | n | avg (s) | p50 (s) | p95 (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label in [
        ("vector", "Vector (/compare)"),
        ("hybrid", "Hybrid (/compare)"),
        ("standard", "Standard (/compare-rag)"),
        ("graph", "Graph (/compare-rag)"),
    ]:
        st = lat[key]
        lines.append(
            f"| {label} | {st['n']} | {fmt(st['avg_s'])} | {fmt(st['p50_s'])} | {fmt(st['p95_s'])} |"
        )

    lines.append("\n## Quality (simple keyword-hit on answer vs expected)\n")
    lines.append(f"- vector_hit_avg: **{q['vector_hit_avg']:.3f}**")
    lines.append(f"- hybrid_hit_avg: **{q['hybrid_hit_avg']:.3f}**")
    lines.append(f"- standard_hit_avg: **{q['standard_hit_avg']:.3f}**")
    lines.append(f"- graph_hit_avg: **{q['graph_hit_avg']:.3f}**\n")

    lines.append("## Samples\n")
    for item in out["per_question"][:2]:
        lines.append(f"### Q{item['index']}: {item['question']}")
        lines.append(f"- expected: {item.get('expected_answer','')}")
        comp = item.get("compare_sample") or {}
        cr = item.get("compare_rag_sample") or {}
        v_ans = ((comp.get("vector_result") or {}).get("answer") or "").strip().replace("\n", " ")
        h_ans = ((comp.get("hybrid_result") or {}).get("answer") or "").strip().replace("\n", " ")
        s_ans = ((cr.get("standard_result") or {}).get("answer") or "").strip().replace("\n", " ")
        g_ans = ((cr.get("graph_result") or {}).get("answer") or "").strip().replace("\n", " ")
        lines.append(f"- vector: {v_ans[:180]}{'...' if len(v_ans)>180 else ''}")
        lines.append(f"- hybrid: {h_ans[:180]}{'...' if len(h_ans)>180 else ''}")
        lines.append(f"- standard: {s_ans[:180]}{'...' if len(s_ans)>180 else ''}")
        lines.append(f"- graph: {g_ans[:180]}{'...' if len(g_ans)>180 else ''}")
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_autogen_tex(out: dict[str, Any], path: Path) -> None:
    lat = out["latency"]
    q = out["quality"]

    # /compare trả 1 round-trip chung cho vector+hybrid; vẫn ghi riêng để table/plot dễ dùng.
    vec_avg = lat["vector"]["avg_s"]
    hyb_avg = lat["hybrid"]["avg_s"]
    std_avg = lat["standard"]["avg_s"]
    grp_avg = lat["graph"]["avg_s"]

    # Set ymax a bit above max to avoid clipping label
    ymax = max(vec_avg, hyb_avg, std_avg, grp_avg, 0.0) * 1.25
    ymax = max(ymax, 1.0)

    def f(x: float) -> str:
        return f"{x:.3f}"

    tex = f"""% Auto-generated by e2e_benchmark.py
% Do NOT edit manually. Re-run script to refresh numbers.

\\newcommand{{\\ETwoEVecAvg}}{{{f(vec_avg)}}}
\\newcommand{{\\ETwoEHybAvg}}{{{f(hyb_avg)}}}
\\newcommand{{\\ETwoEStdAvg}}{{{f(std_avg)}}}
\\newcommand{{\\ETwoEGrpAvg}}{{{f(grp_avg)}}}
\\newcommand{{\\ETwoEYMax}}{{{f(ymax)}}}

\\newcommand{{\\ETwoEVecHit}}{{{q['vector_hit_avg']:.3f}}}
\\newcommand{{\\ETwoEHybHit}}{{{q['hybrid_hit_avg']:.3f}}}
\\newcommand{{\\ETwoEStdHit}}{{{q['standard_hit_avg']:.3f}}}
\\newcommand{{\\ETwoEGrpHit}}{{{q['graph_hit_avg']:.3f}}}

\\newcommand{{\\ETwoEPlotHybrid}}{{(Vector,\\ETwoEVecAvg) (Hybrid,\\ETwoEHybAvg)}}
\\newcommand{{\\ETwoEPlotGraph}}{{(Standard,\\ETwoEStdAvg) (Graph,\\ETwoEGrpAvg)}}
"""
    path.write_text(tex.strip() + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="E2E benchmark for /compare and /compare-rag")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL (default: http://localhost:8000)")
    parser.add_argument("--file", default="", help="Path to PDF/DOCX to upload (optional; auto-pick from ./data if empty)")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH), help="Questions JSON path")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per question (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per question (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rerank-enabled", action="store_true", default=True)
    parser.add_argument("--rerank-threshold", type=float, default=0.0)

    args = parser.parse_args()

    file_arg = str(args.file or "").strip()
    file_path = Path(file_arg) if file_arg else None
    if file_path is None:
        # Auto-pick: prefer newest PDF in ./data, fallback to any pdf in repo (excluding bare_conf.pdf)
        candidates: list[Path] = []
        if DEFAULT_DATA_DIR.exists():
            candidates.extend(sorted(DEFAULT_DATA_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True))
        if not candidates:
            root = Path(".")
            for p in sorted(root.glob("**/*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
                if p.name.lower() == "bare_conf.pdf":
                    continue
                candidates.append(p)
        if not candidates:
            raise SystemExit("No PDF found. Provide --file <path-to-pdf>.")
        file_path = candidates[0]
        print(f"[INFO] Auto-picked file: {file_path}")
    else:
        if not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")

    try:
        out = run_e2e(
            base_url=args.base_url,
            file_path=file_path,
            questions_path=Path(args.questions),
            repeats=max(1, args.repeats),
            warmup=max(0, args.warmup),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bm25_weight=args.bm25_weight,
            rerank_enabled=bool(args.rerank_enabled),
            rerank_threshold=float(args.rerank_threshold),
        )
    except (URLError, HTTPError) as exc:
        print(f"[ERROR] Cannot reach API at {args.base_url}: {exc}")
        print("Start FastAPI (uvicorn app:app --reload) and ensure Postgres+Ollama are running, then re-run.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Benchmark failed: {exc}")
        return 1

    Path("e2e_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out, Path("e2e_results.md"))
    write_autogen_tex(out, Path("e2e_autogen.tex"))
    print("[OK] Wrote e2e_results.json, e2e_results.md, e2e_autogen.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

