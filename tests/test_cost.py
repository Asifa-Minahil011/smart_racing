import glob
import pytest
from src.graph import Graph, Vertex, Edge, RESTRICTED
from src.loader import load_circuit
from src.cost import CostModel, PRESETS

CIRCUIT_FILES = sorted(glob.glob("circuits/*.json"))


def two_edge_graph():
    g = Graph()
    for i, (x, y) in enumerate([(0, 0), (10, 0), (10, 10)]):
        g.add_vertex(Vertex(i, x, y))
    g.add_edge(Edge(0, 0, 1, 10, 1.0, 0.5, difficulty=1))
    g.add_edge(Edge(1, 1, 2, 10, 1.0, 0.5, difficulty=5))
    return g


def test_cost_non_negative_all_presets():
    g = two_edge_graph()
    for p in PRESETS:
        m = CostModel(g, p)
        assert all(m.edge_cost(e) >= 0 for e in g.edges.values())


def test_harder_road_costs_more_in_safety():
    g = two_edge_graph()
    m = CostModel(g, "Safety")
    assert m.edge_cost(g.edges[1]) > m.edge_cost(g.edges[0])


def test_restricted_costs_more():
    g = two_edge_graph()
    m = CostModel(g, "Speed")
    before = m.edge_cost(g.edges[0])
    g.set_edge_status(0, RESTRICTED)
    assert m.edge_cost(g.edges[0]) > before


def test_penalty_and_speed_cap_cost_more():
    g = two_edge_graph()
    m = CostModel(g, "Balanced")
    before = m.edge_cost(g.edges[0])
    g.set_edge_penalty(0, difficulty_penalty=2, speed_cap=5, obstacle_penalty=3)
    assert m.edge_cost(g.edges[0]) > before


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        CostModel(two_edge_graph(), "Nope")


@pytest.mark.parametrize("path", CIRCUIT_FILES)
@pytest.mark.parametrize("preset", list(PRESETS))
def test_heuristic_is_lower_bound_per_edge(path, preset):
    c = load_circuit(path)
    m = CostModel(c.graph, preset)
    for e in c.graph.edges.values():
        # admissible and consistent along each edge, in either direction
        assert m.heuristic(e.from_v, e.to_v) <= m.edge_cost(e) + 1e-9
        assert m.heuristic(e.to_v, e.from_v) <= m.edge_cost(e) + 1e-9