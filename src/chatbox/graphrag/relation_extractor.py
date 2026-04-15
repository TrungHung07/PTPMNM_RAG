from __future__ import annotations


def extract_relations(entities: list[str], default_relation: str = "related_to") -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    for left, right in zip(entities, entities[1:], strict=False):
        relations.append((left, right, default_relation))
    return relations
