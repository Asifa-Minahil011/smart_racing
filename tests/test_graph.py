import pytest
from src.graph import *


def small_graph():
    g = Graph()
    for i, (x, y) in enumerate([(0, 0), (10, 0), (10, 10)]):
        g.add_vertex(Vertex(i, x, y))
    g.add_edge(Edge(0, 0, 1, 10, 10, 5))                  # undirected
    g.add_edge(Edge(1, 1, 2, 10, 10, 5, directed=True))   # one-way 1 -> 2
    return g


def test_undirected_edge_both_ways():
    g = small_graph()
    assert [n for _, n in g.neighbors(0)] == [1]
    assert 0 in [n for _, n in g.neighbors(1)]


def test_directed_edge_one_way():
    g = small_graph()
    assert 2 in [n for _, n in g.neighbors(1)]
    assert list(g.neighbors(2)) == []


def test_block_and_unblock():
    g = small_graph()
    g.block_edge(0)
    assert list(g.neighbors(0)) == []
    assert list(g.neighbors(1))[0][1] == 2      # only the 1->2 edge remains
    g.unblock_edge(0)
    assert [n for _, n in g.neighbors(0)] == [1]


def test_restricted_stays_traversable():
    g = small_graph()
    g.set_edge_status(0, RESTRICTED)
    assert [n for _, n in g.neighbors(0)] == [1]


def test_invalid_input_rejected():
    g = small_graph()
    with pytest.raises(ValueError):
        g.add_edge(Edge(9, 0, 99, 1, 1, 1))              # missing endpoint
    with pytest.raises(ValueError):
        g.add_edge(Edge(9, 0, 1, -5, 1, 1))              # negative distance
    with pytest.raises(ValueError):
        g.add_edge(Edge(9, 0, 1, 1, 1, 1, difficulty=7))  # bad difficulty
    with pytest.raises(ValueError):
        g.add_edge(Edge(0, 0, 1, 1, 1, 1))               # duplicate id


def test_service_vertex_toggle():
    g = Graph()
    g.add_vertex(Vertex(0, 0, 0, FUEL_STATION))
    assert g.is_service_available(0)
    g.set_vertex_active(0, False)
    assert not g.is_service_available(0)