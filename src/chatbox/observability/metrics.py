from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class InMemoryMetrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def observe_ms(self, name: str, value_ms: float) -> None:
        self.timings[name].append(value_ms)
