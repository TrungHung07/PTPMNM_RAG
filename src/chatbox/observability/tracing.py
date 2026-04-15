from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4


@dataclass(slots=True)
class SpanResult:
    span_name: str
    elapsed_ms: float


class TraceFactory:
    def new_trace_id(self) -> str:
        return str(uuid4())

    @contextmanager
    def span(self, span_name: str):
        start = perf_counter()
        result = SpanResult(span_name=span_name, elapsed_ms=0.0)
        try:
            yield result
        finally:
            result.elapsed_ms = (perf_counter() - start) * 1000.0
