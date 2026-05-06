"""
Analyze messages exported from DB (messages.csv) and compute retrieval quality
based on citations chunk metadata vs gold labels in tests/data.json.

This is useful when you already have DB logs and don't want to re-run the API benchmark.

Inputs:
  - CSV exported from messages table: columns at least include
      question, search_mode, citations
  - Questions JSON (default tests/data.json) with optional `gold_citations`
      Each gold item can include: page, chunk_index, paragraph, source_contains

Outputs:
  - Printed markdown table to stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_gold(questions_path: Path) -> dict[str, list[dict[str, Any]]]:
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        q = str((it or {}).get("question") or "").strip()
        if not q:
            continue
        raw_gold = (it or {}).get("gold_citations") or []
        gold: list[dict[str, Any]] = []
        if isinstance(raw_gold, list):
            for g in raw_gold:
                if isinstance(g, dict):
                    gold.append(dict(g))
        out[q] = gold
    return out


def _meta_matches_gold(meta: dict[str, Any], gold: dict[str, Any]) -> bool:
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


def recall_at_k(citations: list[dict[str, Any]], gold_list: list[dict[str, Any]], *, k: int) -> float | None:
    if not gold_list:
        return None
    top = citations[: max(0, int(k))]
    metas = [(c or {}).get("metadata") or {} for c in top]
    hit = 0
    for gold in gold_list:
        if any(_meta_matches_gold(m, gold) for m in metas):
            hit += 1
    return hit / len(gold_list)


def top1_chunk_hit(citations: list[dict[str, Any]], gold_list: list[dict[str, Any]]) -> float | None:
    if not gold_list or not citations:
        return None
    meta = (citations[0] or {}).get("metadata") or {}
    return 1.0 if any(_meta_matches_gold(meta, g) for g in gold_list) else 0.0


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class Agg:
    n: int = 0
    recall_samples: list[float] = None  # type: ignore[assignment]
    top1_samples: list[float] = None  # type: ignore[assignment]
    citation_counts: list[int] = None  # type: ignore[assignment]
    score_present: int = 0
    score_avgs: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.recall_samples = []
        self.top1_samples = []
        self.citation_counts = []
        self.score_avgs = []


def mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze messages.csv to compute Recall@k based on citations metadata.")
    p.add_argument("--csv", required=True, help="Path to messages.csv export")
    p.add_argument("--questions", default="tests/data.json", help="Questions JSON with gold_citations")
    p.add_argument("--k", type=int, default=4, help="Top-k citations for Recall@k")
    args = p.parse_args()

    csv_path = Path(args.csv)
    questions_path = Path(args.questions)
    k = max(1, int(args.k))

    gold_by_q = load_gold(questions_path)
    if not gold_by_q:
        raise SystemExit(f"No questions loaded from {questions_path}")

    aggs: dict[str, Agg] = {}
    unknown_questions: set[str] = set()
    rows_total = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            q = str(row.get("question") or "").strip()
            mode = str(row.get("search_mode") or "").strip()
            if not q or not mode:
                continue
            gold = gold_by_q.get(q)
            if gold is None:
                unknown_questions.add(q)
                continue

            cits_raw = str(row.get("citations") or "")
            cits = safe_json_loads(cits_raw)
            if not isinstance(cits, list):
                cits = []

            r = recall_at_k(cits, gold, k=k)
            t1 = top1_chunk_hit(cits, gold)

            a = aggs.setdefault(mode, Agg())
            a.n += 1
            a.citation_counts.append(len(cits))
            if r is not None:
                a.recall_samples.append(float(r))
            if t1 is not None:
                a.top1_samples.append(float(t1))

            # score stats (only meaningful when rerank enabled)
            scores: list[float] = []
            for c in cits[:k]:
                sc = (c or {}).get("score")
                if sc is None:
                    continue
                try:
                    scores.append(float(sc))
                except Exception:
                    continue
            if scores:
                a.score_present += 1
                a.score_avgs.append(mean(scores))

    # Print markdown summary
    modes_order = ["vector", "hybrid", "vector_no_rerank", "hybrid_no_rerank"]
    # Keep any other modes at the end
    tail = [m for m in aggs.keys() if m not in modes_order]
    order = [m for m in modes_order if m in aggs] + sorted(tail)

    lines: list[str] = []
    lines.append("# Message-table Benchmark Summary\n")
    lines.append(f"- csv: `{csv_path}`")
    lines.append(f"- questions: `{questions_path}`")
    lines.append(f"- k: `{k}`")
    lines.append(f"- total_rows_seen: `{rows_total}`")
    lines.append(f"- matched_rows_used: `{sum(aggs[m].n for m in aggs)}`")
    lines.append(f"- unknown_questions_skipped: `{len(unknown_questions)}`\n")

    lines.append("## Retrieval quality (chunk-level)\n")
    lines.append("| search_mode | n | avg_recall@k | avg_top1_chunk_hit | avg_citation_count | score_present_rows | avg_score(top-k) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in order:
        a = aggs[m]
        lines.append(
            f"| `{m}` | {a.n} | {mean(a.recall_samples):.3f} | {mean(a.top1_samples):.3f} | "
            f"{mean([float(x) for x in a.citation_counts]):.2f} | {a.score_present} | {mean(a.score_avgs):.3f} |"
        )

    def get(m: str, key: str) -> float:
        a = aggs.get(m)
        if not a:
            return 0.0
        if key == "rec":
            return mean(a.recall_samples)
        if key == "t1":
            return mean(a.top1_samples)
        return 0.0

    if "vector" in aggs and "vector_no_rerank" in aggs:
        lines.append("\n## Delta (rerank - no_rerank)\n")
        for base in ["vector", "hybrid"]:
            if base not in aggs or f"{base}_no_rerank" not in aggs:
                continue
            d_rec = get(base, "rec") - get(f"{base}_no_rerank", "rec")
            d_t1 = get(base, "t1") - get(f"{base}_no_rerank", "t1")
            lines.append(f"- `{base}`: avg_recall@k delta = **{d_rec:+.3f}**; avg_top1_chunk_hit delta = **{d_t1:+.3f}**")

    print("\n".join(lines).strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

