from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedText:
    file_type: str
    text: str
    metadata: dict[str, int | str | float] = field(default_factory=dict)
