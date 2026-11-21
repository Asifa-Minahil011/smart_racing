import random
import pytest
from src.graph import Graph, Vertex, Edge
from src.search import (bfs_path, bfs_tree, reachable_checkpoints,
                        unreachable_vertices)
from src.counters import Counters


def make(n, edges, directed=False):
    g = Graph()
    for i in range(n):
        g.add_vertex(Vertex(i, i * 10, 0))
    for k, (a, b) in enumerate(edges):
        g.add_edge(Edge(k, a, b, 10, 1, 1, directed=directed))
    return g


def test_spec_example_bfs_minimizes_edges_not_distance():
    # S=0, T=1, A=2, B=3. Direct S-T has length 10; S-A-B-T has 3 edges of 2.
    g = Graph()
    for i in range(4):
        g.add_vertex(Vertex(i, 0, 0))
    g.add_edge(Edge(0, 0, 1, 10, 1, 1))
    g.add_edge(Edge(1, 0, 2, 2, 1, 1))
    g.add_edge(Edge(2, 2, 3, 2, 1, 1))
    g.add_edge(Edge(3, 3, 1, 2, 1, 1))
    vertices, edges = bfs_path(g, 0, 1)
    assert vertices == [0, 1] and edges == [0]


def test_source_equals_target():
    g = make(3, [(0, 1), (1, 2)])
    assert bfs_path(g, 1, 1) == ([1], [])


def test_unreachable_returns_none():
    g = make(4, [(0, 1), (2, 3)])          # two components
    assert bfs_path(g, 0, 3) is None
    assert unreachable_vertices(g, 0) == [2, 3]


def test_single_vertex_graph():
    g = make(1, [])
    assert bfs_path(g, 0, 0) == ([0], [])
    assert bfs_tree(g, 0) == {0: None}


def test_cycle_does_not_loop_forever():
    g = make(3, [(0, 1), (1, 2), (2, 0)])
    assert len(bfs_path(g, 0, 2)[1]) == 1


def test_blocked_edge_is_avoided():
    g = make(3, [(0, 1), (1, 2), (0, 2)])
    g.block_edge(2)                         # the direct 0-2 road
    assert bfs_path(g, 0, 2)[0] == [0, 1, 2]
    g.block_edge(1)
    assert bfs_path(g, 0, 2) is None


def test_one_way_edge_respected():
    g = make(2, [(0, 1)], directed=True)
    assert bfs_path(g, 0, 1) is not None
    assert bfs_path(g, 1, 0) is None


def test_counters():
    g = make(3, [(0, 1), (1, 2)])
    c = Counters()
    bfs_tree(g, 0, counters=c)
    assert c.visited == 3 and c.expanded == 3 and c.edges_examined == 4


def test_observer_receives_events():
    g = make(3, [(0, 1), (1, 2)])
    seen = []
    bfs_path(g, 0, 2, observer=lambda name, data: seen.append(name))
    assert "visit" in seen and "expand" in seen and seen[-1] == "path_found"


def test_reachable_checkpoints_in_order():
    g = Graph()
    g.add_vertex(Vertex(0, 0, 0, "Start"))
    g.add_vertex(Vertex(1, 1, 0, "Checkpoint", 1))
    g.add_vertex(Vertex(2, 2, 0, "Checkpoint", 2))
    g.add_vertex(Vertex(3, 3, 0, "Checkpoint", 3))
    g.add_edge(Edge(0, 0, 1, 10, 1, 1))
    g.add_edge(Edge(1, 1, 2, 10, 1, 1))
    assert reachable_checkpoints(g, 0) == [1, 2]    # 3 is disconnected


def brute_dist(n, edges, directed, s):
    """Bellman-Ford with unit weights: an independent check of BFS."""
    INF = 10 ** 9
    d = [INF] * n
    d[s] = 0
    for _ in range(n):
        for a, b in edges:
            if d[a] + 1 < d[b]:
                d[b] = d[a] + 1
            if not directed and d[b] + 1 < d[a]:
                d[a] = d[b] + 1
    return d


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("directed", [False, True])
def test_bfs_matches_bruteforce_min_edges(seed, directed):
    rnd = random.Random(seed)
    n = 12
    edges = []
    for _ in range(rnd.randint(5, 25)):
        a, b = rnd.sample(range(n), 2)
        edges.append((a, b))
    g = make(n, edges, directed)
    d = brute_dist(n, edges, directed, 0)
    for t in range(n):
        path = bfs_path(g, 0, t)
        if d[t] >= 10 ** 9:
            assert path is None
        else:
            assert len(path[1]) == d[t]