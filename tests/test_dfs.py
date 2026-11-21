import random
import pytest
from src.graph import Graph, Vertex, Edge
from src.loader import load_circuit
from src.search import (bfs_path, dfs_traversal, find_cycle,
                        connected_components, dead_ends, alternative_routes)
from src.counters import Counters


def make(n, edges, directed=False, types=None):
    types = types or {}
    g = Graph()
    for i in range(n):
        g.add_vertex(Vertex(i, i * 10, 0, types.get(i, "Intersection")))
    for k, (a, b) in enumerate(edges):
        g.add_edge(Edge(k, a, b, 10, 1, 1, directed=directed))
    return g


def spec_graph():
    """S=0, T=1: direct edge 0 (length 10) vs S-A-B-T (edges 1, 2, 3)."""
    g = Graph()
    for i in range(4):
        g.add_vertex(Vertex(i, 0, 0))
    g.add_edge(Edge(0, 0, 1, 10, 1, 1))
    g.add_edge(Edge(1, 0, 2, 2, 1, 1))
    g.add_edge(Edge(2, 2, 3, 2, 1, 1))
    g.add_edge(Edge(3, 3, 1, 2, 1, 1))
    return g


# ---------- traversal ----------
def test_traversal_order_and_times():
    g = make(3, [(0, 1), (1, 2)])
    order, disc, fin = dfs_traversal(g, 0)
    assert order == [0, 1, 2]
    assert disc[0] < disc[1] < disc[2] < fin[2] < fin[1] < fin[0]


def test_traversal_only_reachable_vertices():
    g = make(4, [(0, 1), (2, 3)])
    assert dfs_traversal(g, 0)[0] == [0, 1]


def test_traversal_terminates_on_cycle():
    g = make(3, [(0, 1), (1, 2), (2, 0)])
    assert sorted(dfs_traversal(g, 0)[0]) == [0, 1, 2]


def test_traversal_counters():
    g = make(3, [(0, 1), (1, 2)])
    c = Counters()
    dfs_traversal(g, 0, counters=c)
    assert c.visited == 3 and c.expanded == 3 and c.edges_examined == 4


# ---------- cycle detection ----------
def test_cycle_found_in_triangle():
    cyc = find_cycle(make(3, [(0, 1), (1, 2), (2, 0)]))
    assert cyc is not None and cyc[0] == cyc[-1] and len(cyc) == 4


def test_no_cycle_in_undirected_tree():
    assert find_cycle(make(4, [(0, 1), (1, 2), (1, 3)])) is None


def test_single_undirected_edge_is_not_a_cycle():
    assert find_cycle(make(2, [(0, 1)])) is None


def test_directed_chain_has_no_cycle():
    assert find_cycle(make(3, [(0, 1), (1, 2)], directed=True)) is None


def test_directed_ring_has_cycle():
    assert find_cycle(make(3, [(0, 1), (1, 2), (2, 0)], directed=True))


def test_parallel_roads_form_a_cycle():
    assert find_cycle(make(2, [(0, 1), (0, 1)])) is not None


def test_single_vertex_no_cycle():
    assert find_cycle(make(1, [])) is None


def test_blocking_breaks_the_cycle():
    g = make(3, [(0, 1), (1, 2), (2, 0)])
    g.block_edge(2)
    assert find_cycle(g) is None


# ---------- components ----------
def test_components_basic():
    g = make(5, [(0, 1), (2, 3)])
    assert connected_components(g) == [[0, 1], [2, 3], [4]]


def test_components_single_vertex():
    assert connected_components(make(1, [])) == [[0]]


def test_components_treat_one_way_as_connected():
    assert connected_components(make(2, [(0, 1)], directed=True)) == [[0, 1]]


# ---------- dead ends ----------
def test_dead_end_leaf_detected():
    # triangle 0-1-2 (finish = 0) plus a leaf 3 hanging off vertex 1
    g = make(4, [(0, 1), (1, 2), (2, 0), (1, 3)])
    assert dead_ends(g, finish=0) == [3]


def test_checkpoint_leaf_is_not_a_dead_end():
    g = make(4, [(0, 1), (1, 2), (2, 0), (1, 3)], types={3: "Checkpoint"})
    assert dead_ends(g, finish=0) == []


def test_circuit1_dead_end_is_vertex_20():
    c = load_circuit("circuits/circuit1.json")
    assert dead_ends(c.graph, c.finish) == [20]


# ---------- alternative routes ----------
def test_alternative_to_direct_road():
    g = spec_graph()
    alts = alternative_routes(g, 0, 1, main_edges=[0])
    assert alts == [([0, 2, 3, 1], [1, 2, 3])]


def test_no_alternative_when_only_one_path():
    g = make(2, [(0, 1)])
    assert alternative_routes(g, 0, 1, main_edges=[0]) == []


def test_max_paths_limit():
    g = make(5, [(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)])
    main = bfs_path(g, 0, 4)[1]
    assert len(alternative_routes(g, 0, 4, main, max_paths=1)) == 1
    assert len(alternative_routes(g, 0, 4, main, max_paths=10)) == 2


def test_max_edges_prunes_long_paths():
    g = spec_graph()
    assert alternative_routes(g, 0, 1, [0], max_edges=1) == []
    assert len(alternative_routes(g, 0, 1, [0], max_edges=3)) == 1


def test_blocked_edge_never_used_in_alternatives():
    g = spec_graph()
    g.block_edge(2)
    assert alternative_routes(g, 0, 1, [0]) == []


def test_start_equals_target_gives_no_alternatives():
    assert alternative_routes(spec_graph(), 1, 1) == []


def test_alternatives_on_circuit1_are_simple_paths():
    c = load_circuit("circuits/circuit1.json")
    main = bfs_path(c.graph, 0, 10)[1]
    alts = alternative_routes(c.graph, 0, 10, main, max_paths=5)
    assert len(alts) == 5
    for vertices, edges in alts:
        assert vertices[0] == 0 and vertices[-1] == 10
        assert len(set(vertices)) == len(vertices)
        assert set(edges) != set(main)


def all_simple_paths(n, edges, directed, s, t):
    adj = {i: [] for i in range(n)}
    for k, (a, b) in enumerate(edges):
        adj[a].append((k, b))
        if not directed:
            adj[b].append((k, a))
    out = set()

    def rec(u, seen, path):
        if u == t:
            out.add(tuple(path))
            return
        for k, v in adj[u]:
            if v not in seen:
                seen.add(v)
                path.append(k)
                rec(v, seen, path)
                path.pop()
                seen.remove(v)

    rec(s, {s}, [])
    return out


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("directed", [False, True])
def test_alternatives_match_bruteforce(seed, directed):
    rnd = random.Random(seed)
    n = 7
    edges = [tuple(rnd.sample(range(n), 2)) for _ in range(rnd.randint(4, 12))]
    g = make(n, edges, directed)
    everything = all_simple_paths(n, edges, directed, 0, n - 1)
    main = bfs_path(g, 0, n - 1)
    alts = alternative_routes(g, 0, n - 1, main[1] if main else None,
                              max_paths=10 ** 9, max_edges=n)
    got = {tuple(e) for _, e in alts}
    expected = everything - ({tuple(main[1])} if main else set())
    assert got == expected