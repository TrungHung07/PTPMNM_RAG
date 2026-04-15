from chatbox.graphrag.builder import GraphBuilder, InMemoryGraphStore


def test_graph_builder_extracts_nodes_and_edges() -> None:
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)

    nodes, edges = builder.build_from_chunks([("c1", "Alice manages Bob")])

    assert nodes
    assert edges
    assert any(node.label.lower() == "alice" for node in nodes)
