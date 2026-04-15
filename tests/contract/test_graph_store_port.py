from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore


def test_graph_store_upsert_and_search_paths() -> None:
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)

    builder.build_from_chunks([("c1", "Alice knows Bob"), ("c2", "Bob works at Acme")])
    hits = store.search_paths(query_entities=["alice", "bob"], top_k=2, max_hops=2)

    assert hits
    assert hits[0].score > 0
    assert hits[0].evidence_chunk_ids
