from __future__ import annotations


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for token in text.replace(".", " ").split():
        if not token:
            continue
        candidate = token.strip().strip(",;:")
        if not candidate:
            continue
        if candidate[0].isupper() and candidate.lower() not in seen:
            seen.add(candidate.lower())
            entities.append(candidate)
    return entities
