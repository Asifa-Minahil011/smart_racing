"""BFS and DFS on the track graph. Owner: Asifa Minahil."""
from collections import deque
from src.counters import notify
from src.graph import BLOCKED
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
# ======================= DFS =======================

def dfs_traversal(graph, start, counters=None, observer=None):
    """Iterative DFS from start (follows one-way roads, skips Blocked edges).
    Returns (order, discovery, finish): visit order and discovery/finish
    times. Time O(V+E), auxiliary space O(V)."""
    if start not in graph.vertices:
        raise KeyError(f"Vertex {start} does not exist")
    disc, fin, order = {}, {}, []
    clock = 0

    def discover(v, via=None):
        nonlocal clock
        disc[v] = clock
        clock += 1
        order.append(v)
        if counters:
            counters.visited += 1
        notify(observer, "visit", vertex=v, via=via)

    discover(start)
    stack = [(start, graph.neighbors(start))]
    while stack:
        u, it = stack[-1]
        advanced = False
        for edge, v in it:
            if counters:
                counters.edges_examined += 1
            if v not in disc:
                discover(v, edge.id)
                stack.append((v, graph.neighbors(v)))
                advanced = True
                break
        if not advanced:
            stack.pop()
            fin[u] = clock
            clock += 1
            if counters:
                counters.expanded += 1
            notify(observer, "expand", vertex=u)
    return order, disc, fin


def find_cycle(graph, counters=None, observer=None):
    """Return one cycle as a vertex list [w, ..., u, w], or None.
    A cycle is a back edge to a vertex that is still on the DFS stack
    (gray). The edge used to arrive is ignored, so a two-way road is not
    counted as a cycle by itself, but parallel roads are. Respects one-way
    roads and skips Blocked edges. Time O(V+E)."""
    color = {}          # absent = unseen, 1 = on stack, 2 = finished
    for root in sorted(graph.vertices):
        if root in color:
            continue
        color[root] = 1
        if counters:
            counters.visited += 1
        path = [root]
        stack = [(root, None, graph.neighbors(root))]
        while stack:
            u, parent_edge, it = stack[-1]
            advanced = False
            for edge, v in it:
                if edge.id == parent_edge:
                    continue
                if counters:
                    counters.edges_examined += 1
                c = color.get(v)
                if c is None:
                    color[v] = 1
                    if counters:
                        counters.visited += 1
                    notify(observer, "visit", vertex=v, via=edge.id)
                    path.append(v)
                    stack.append((v, edge.id, graph.neighbors(v)))
                    advanced = True
                    break
                if c == 1:                       # back edge: cycle found
                    cycle = path[path.index(v):] + [v]
                    notify(observer, "path_found", vertices=cycle)
                    return cycle
            if not advanced:
                color[u] = 2
                path.pop()
                stack.pop()
                if counters:
                    counters.expanded += 1
                notify(observer, "expand", vertex=u)
    return None


def _undirected_adjacency(graph):
    adj = {v: [] for v in graph.vertices}
    for e in graph.edges.values():
        if e.status != BLOCKED:
            adj[e.from_v].append(e.to_v)
            adj[e.to_v].append(e.from_v)
    return adj


def connected_components(graph, counters=None):
    """Components, treating every road as two-way (weak connectivity).
    Returns a list of sorted vertex lists, ordered by smallest vertex id.
    Time O(V+E)."""
    adj = _undirected_adjacency(graph)
    seen, components = set(), []
    for root in sorted(graph.vertices):
        if root in seen:
            continue
        seen.add(root)
        comp, stack = [], [root]
        while stack:
            u = stack.pop()
            comp.append(u)
            if counters:
                counters.expanded += 1
            for v in adj[u]:
                if counters:
                    counters.edges_examined += 1
                if v not in seen:
                    seen.add(v)
                    if counters:
                        counters.visited += 1
                    stack.append(v)
        components.append(sorted(comp))
    return components


def _can_reach(graph, start, target, banned_edge):
    """Is target reachable from start without ever using banned_edge?"""
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        if u == target:
            return True
        for edge, v in graph.neighbors(u):
            if edge.id != banned_edge and v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def dead_ends(graph, finish):
    """Dead end (our documented rule): a vertex that is not Start, Finish
    or a Checkpoint, and from which the finish cannot be reached without
    reversing, however the vertex was entered. For each way of arriving
    (each traversable incoming road) we search for the finish while
    forbidding that same road. If every arrival fails, it is a dead end.
    Time O(V * deg * (V+E)): fine for track-sized graphs."""
    result = []
    for vid in sorted(graph.vertices):
        v = graph.vertices[vid]
        if vid == finish or v.type in ("Start", "Finish", "Checkpoint"):
            continue
        incoming = [e.id for e in graph.edges.values()
                    if e.status != BLOCKED
                    and (e.to_v == vid or (not e.directed and e.from_v == vid))]
        if not incoming:
            continue
        if all(not _can_reach(graph, vid, finish, eid) for eid in incoming):
            result.append(vid)
    return result


def alternative_routes(graph, start, target, main_edges=None, max_paths=10,
                       max_edges=30, max_steps=200000, counters=None,
                       observer=None):
    """Simple start-to-target paths that differ from the main route by at
    least one edge. Returns a list of (vertices, edge_ids).
    Enumerating all simple paths is exponential in the worst case, so the
    search is bounded by max_paths (routes returned), max_edges (path
    length) and max_steps (expansions). Recursion depth <= max_edges."""
    if start == target:
        return []
    main_set = set(main_edges) if main_edges is not None else None
    found, steps = [], [0]
    on_path, vpath, epath = {start}, [start], []

    def rec(u):
        if len(found) >= max_paths or steps[0] >= max_steps:
            return
        if u == target:
            if main_set is None or set(epath) != main_set:
                found.append((list(vpath), list(epath)))
                notify(observer, "path_found", vertices=list(vpath))
            return
        if len(epath) >= max_edges:
            return
        steps[0] += 1
        if counters:
            counters.expanded += 1
        for edge, v in graph.neighbors(u):
            if counters:
                counters.edges_examined += 1
            if v in on_path:
                continue
            on_path.add(v)
            vpath.append(v)
            epath.append(edge.id)
            rec(v)
            on_path.discard(v)
            vpath.pop()
            epath.pop()

    rec(start)
    return found