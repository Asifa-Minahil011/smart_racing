"""BFS and DFS on the track graph. Owner: Asifa Minahil."""

from collections import deque
from src.counters import notify


def bfs_tree(graph, start, target=None, counters=None, observer=None):
    """Breadth-first search from start. Returns parent dict:
    vertex -> (parent_vertex, edge_id), start -> None.
    Stops early when target is expanded. Respects one-way roads and
    skips Blocked edges (through graph.neighbors).
    Time O(V+E), auxiliary space O(V)."""
    if start not in graph.vertices:
        raise KeyError(f"Vertex {start} does not exist")
    parent = {start: None}
    queue = deque([start])
    if counters:
        counters.visited += 1
    notify(observer, "visit", vertex=start)
    while queue:
        u = queue.popleft()
        if counters:
            counters.expanded += 1
        notify(observer, "expand", vertex=u)
        if u == target:
            break
        for edge, v in graph.neighbors(u):
            if counters:
                counters.edges_examined += 1
            if v not in parent:
                parent[v] = (u, edge.id)
                if counters:
                    counters.visited += 1
                notify(observer, "visit", vertex=v, via=edge.id)
                queue.append(v)
    return parent


def rebuild_path(parent, target):
    """Turn a parent dict into (vertex_list, edge_id_list)."""
    vertices, edges = [target], []
    while parent[vertices[-1]] is not None:
        p, eid = parent[vertices[-1]]
        vertices.append(p)
        edges.append(eid)
    vertices.reverse()
    edges.reverse()
    return vertices, edges


def bfs_path(graph, start, target, counters=None, observer=None):
    """Path with the fewest EDGES, as (vertices, edge_ids), or None if
    unreachable. It does NOT minimize distance, time or weighted cost."""
    parent = bfs_tree(graph, start, target, counters, observer)
    if target not in parent:
        return None
    result = rebuild_path(parent, target)
    notify(observer, "path_found", vertices=result[0])
    return result


def reachable_checkpoints(graph, start, counters=None, observer=None):
    """Checkpoint vertex ids reachable from start, in checkpoint order."""
    parent = bfs_tree(graph, start, None, counters, observer)
    return [v.id for v in graph.checkpoints_in_order() if v.id in parent]


def unreachable_vertices(graph, start):
    parent = bfs_tree(graph, start)
    return [vid for vid in graph.vertices if vid not in parent]