import math
import random
import pytest
from src.graph import Graph, Vertex, Edge, RESTRICTED
from src.loader import load_circuit
from src.cost import CostModel, PRESETS
from src.routing import (dijkstra, astar, plan_route, compare_presets,
                         evaluate_route)
from src.counters import Counters

INF = float("inf")


def diamond(diff_top=1, diff_bottom=5):
    """0 -> 3 via top (0-1-3) or bottom (0-2-3); same lengths."""
    g = Graph()
    for i, (x, y) in enumerate([(0, 0), (10, 10), (10, -10), (20, 0)]):
        g.add_vertex(Vertex(i, x, y))
    g.add_edge(Edge(0, 0, 1, 15, 1.5, 0.75, difficulty=diff_top))
    g.add_edge(Edge(1, 1, 3, 15, 1.5, 0.75, difficulty=diff_top))
    g.add_edge(Edge(2, 0, 2, 15, 1.5, 0.75, difficulty=diff_bottom))
    g.add_edge(Edge(3, 2, 3, 15, 1.5, 0.75, difficulty=diff_bottom))
    return g


def check_valid_path(g, route, s, t):
    assert route.vertices[0] == s and route.vertices[-1] == t
    assert len(route.vertices) == len(route.edges) + 1
    for i, eid in enumerate(route.edges):
        e = g.edges[eid]
        assert e.status != "Blocked"
        a, b = route.vertices[i], route.vertices[i + 1]
        assert {a, b} == {e.from_v, e.to_v}
        if e.directed:
            assert e.from_v == a


# ---------- basic behaviour ----------
def test_dijkstra_prefers_cheaper_road():
    g = diamond()
    r = dijkstra(g, CostModel(g, "Balanced"), 0, 3)
    assert r.vertices == [0, 1, 3]


def test_block_forces_other_route_then_unreachable():
    g = diamond()
    m = CostModel(g, "Balanced")
    g.block_edge(1)
    assert dijkstra(g, m, 0, 3).vertices == [0, 2, 3]
    g.block_edge(3)
    assert dijkstra(g, m, 0, 3) is None
    assert astar(g, m, 0, 3) is None


def test_equal_cost_paths_deterministic_and_optimal():
    g = diamond(1, 1)
    m = CostModel(g, "Balanced")
    r1, r2 = dijkstra(g, m, 0, 3), dijkstra(g, m, 0, 3)
    assert r1.vertices == r2.vertices
    assert r1.cost == pytest.approx(2 * m.edge_cost(g.edges[0]))
    assert astar(g, m, 0, 3).cost == pytest.approx(r1.cost)


def test_source_equals_target():
    g = diamond()
    m = CostModel(g, "Speed")
    for fn in (dijkstra, astar):
        r = fn(g, m, 2, 2)
        assert r.vertices == [2] and r.edges == [] and r.cost == 0.0


def test_unreachable_target():
    g = Graph()
    for i in range(3):
        g.add_vertex(Vertex(i, i * 10, 0))
    g.add_edge(Edge(0, 0, 1, 10, 1, 0.5))
    m = CostModel(g, "Balanced")
    assert dijkstra(g, m, 0, 2) is None
    assert astar(g, m, 0, 2) is None


def test_unknown_vertex_raises():
    g = diamond()
    with pytest.raises(KeyError):
        dijkstra(g, CostModel(g), 0, 99)


def test_one_way_road_respected():
    g = Graph()
    g.add_vertex(Vertex(0, 0, 0))
    g.add_vertex(Vertex(1, 10, 0))
    g.add_edge(Edge(0, 0, 1, 10, 1, 0.5, directed=True))
    m = CostModel(g)
    assert dijkstra(g, m, 0, 1) is not None
    assert dijkstra(g, m, 1, 0) is None


def test_zero_weight_edge():
    g = Graph()
    g.add_vertex(Vertex(0, 0, 0))
    g.add_vertex(Vertex(1, 0, 0))      # same spot as vertex 0
    g.add_vertex(Vertex(2, 10, 0))
    g.add_edge(Edge(0, 0, 1, 0, 0, 0))              # zero-cost edge
    g.add_edge(Edge(1, 1, 2, 10, 1, 0.5))
    m = CostModel(g, "Balanced")
    assert m.edge_cost(g.edges[0]) == 0
    for fn in (dijkstra, astar):
        r = fn(g, m, 0, 2)
        assert r.vertices == [0, 1, 2]
        assert r.cost == pytest.approx(m.edge_cost(g.edges[1]))


def test_counters_filled():
    g = diamond()
    c = Counters()
    dijkstra(g, CostModel(g), 0, 3, counters=c)
    assert c.visited >= 2 and c.expanded >= 1 and c.relaxed >= 1
    assert 1 <= c.heap_pop <= c.heap_push


def test_observer_events():
    g = diamond()
    seen = []
    astar(g, CostModel(g), 0, 3, observer=lambda n, d: seen.append(n))
    assert "expand" in seen and "relax" in seen and seen[-1] == "path_found"


# ---------- random graphs vs an independent Bellman-Ford ----------
def random_graph(seed, n=14, m=40):
    rnd = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_vertex(Vertex(i, rnd.uniform(0, 100), rnd.uniform(0, 100)))
    for k in range(m):
        a, b = rnd.sample(range(n), 2)
        va, vb = g.vertices[a], g.vertices[b]
        d = math.hypot(va.x - vb.x, va.y - vb.y) * rnd.uniform(1.0, 1.6)
        g.add_edge(Edge(k, a, b, d, d * rnd.uniform(0.05, 0.2), d * 0.05,
                        difficulty=rnd.randint(1, 5),
                        directed=rnd.random() < 0.3))
    for k in rnd.sample(range(m), 4):
        g.set_edge_status(k, RESTRICTED)
    for k in rnd.sample(range(m), 4):
        g.set_edge_penalty(k, rnd.uniform(0, 2), rnd.choice([None, 3.0]),
                           rnd.uniform(0, 5))
    for k in rnd.sample(range(m), 4):
        g.block_edge(k)
    return g


def bellman_ford(g, model, s):
    dist = {v: INF for v in g.vertices}
    dist[s] = 0.0
    for _ in range(len(g.vertices)):
        changed = False
        for u in g.vertices:
            if dist[u] == INF:
                continue
            for e, v in g.neighbors(u):
                nd = dist[u] + model.edge_cost(e)
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    changed = True
        if not changed:
            break
    return dist


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("preset", list(PRESETS))
def test_dijkstra_and_astar_match_bellman_ford(seed, preset):
    g = random_graph(seed)
    m = CostModel(g, preset)
    truth = bellman_ford(g, m, 0)
    for t in g.vertices:
        d, a = dijkstra(g, m, 0, t), astar(g, m, 0, t)
        if truth[t] == INF:
            assert d is None and a is None
        else:
            assert d.cost == pytest.approx(truth[t], abs=1e-9)
            assert a.cost == pytest.approx(truth[t], abs=1e-9)
            check_valid_path(g, d, 0, t)
            check_valid_path(g, a, 0, t)


# ---------- heuristic never overestimates on the real circuit ----------
@pytest.mark.parametrize("preset", list(PRESETS))
def test_heuristic_admissible_on_circuit1(preset):
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph, preset)
    for s in c.graph.vertices:
        for t in c.graph.vertices:
            r = dijkstra(c.graph, m, s, t)
            if r is not None:
                assert m.heuristic(s, t) <= r.cost + 1e-9


# ---------- checkpoint-aware planning ----------
def visits_in_order(vertices, cps):
    i = 0
    for v in vertices:
        if i < len(cps) and v == cps[i]:
            i += 1
    return i == len(cps)


def test_full_lap_plan_keeps_checkpoint_order():
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph, "Balanced")
    plan = plan_route(c.graph, m, c.start, 0, c.finish)
    assert plan is not None
    assert plan.vertices[0] == c.start and plan.vertices[-1] == c.finish
    assert visits_in_order(plan.vertices, c.checkpoints)
    assert len(plan.legs) == 11
    assert plan.cost == pytest.approx(sum(r.cost for _, _, r in plan.legs))


def test_mid_race_plan():
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph, "Speed")
    plan = plan_route(c.graph, m, 5, 5, c.finish)     # next is checkpoint 6
    assert visits_in_order(plan.vertices, c.checkpoints[5:])


def test_only_finish_left():
    c = load_circuit("circuits/circuit1.json")
    plan = plan_route(c.graph, CostModel(c.graph), 10, 10, c.finish)
    assert len(plan.legs) == 1 and plan.vertices[-1] == c.finish


def test_reroute_after_road_closure():
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph, "Balanced")
    plan = plan_route(c.graph, m, c.start, 0, c.finish)
    closed = plan.edges[0]
    c.graph.block_edge(closed)                 # same graph, same model
    plan2 = plan_route(c.graph, m, c.start, 0, c.finish)
    assert plan2 is not None and closed not in plan2.edges
    assert visits_in_order(plan2.vertices, c.checkpoints)
    assert plan2.cost >= plan.cost - 1e-9


def test_unreachable_checkpoint_returns_none():
    c = load_circuit("circuits/circuit1.json")
    for eid in (13, 14, 18):                   # isolate checkpoint 10
        c.graph.block_edge(eid)
    assert plan_route(c.graph, CostModel(c.graph), c.start, 0, c.finish) is None


@pytest.mark.parametrize("preset", list(PRESETS))
def test_plan_same_cost_with_dijkstra_and_astar(preset):
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph, preset)
    a = plan_route(c.graph, m, c.start, 0, c.finish, "astar")
    d = plan_route(c.graph, m, c.start, 0, c.finish, "dijkstra")
    assert a.cost == pytest.approx(d.cost, abs=1e-9)


def test_plan_argument_validation():
    c = load_circuit("circuits/circuit1.json")
    m = CostModel(c.graph)
    with pytest.raises(ValueError):
        plan_route(c.graph, m, 0, 11, c.finish)
    with pytest.raises(ValueError):
        plan_route(c.graph, m, 0, -1, c.finish)
    with pytest.raises(ValueError):
        plan_route(c.graph, m, 0, 0, c.finish, algo="bfs")


def test_compare_presets_and_evaluate_route():
    c = load_circuit("circuits/circuit1.json")
    res = compare_presets(c.graph, 3, 4)
    assert set(res) == set(PRESETS)
    for name, r in res.items():
        m = CostModel(c.graph, name)
        again = evaluate_route(c.graph, m, r.vertices, r.edges)
        assert again.cost == pytest.approx(r.cost)