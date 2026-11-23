import heapq
from dataclasses import dataclass

from src.counters import notify
from src.cost import CostModel, PRESETS
from src.search import rebuild_path


@dataclass
class Route:
    vertices: list
    edges: list
    cost: float        # exact scalar cost that was optimized
    distance: float
    time: float        # estimated travel time (effective, with restrictions)
    fuel: float        # estimated fuel (with difficulty multiplier)


@dataclass
class Plan(Route):
    legs: list         # [(from_vertex, to_vertex, Route), ...]


def evaluate_route(graph, model, vertices, edges):
    """Cost a route under any CostModel (also used to compare presets)."""
    cost = dist = time = fuel = 0.0
    for eid in edges:
        e = graph.edges[eid]
        cost += model.edge_cost(e)
        dist += e.distance
        time += model.effective_time(e)
        fuel += model.effective_fuel(e)
    return Route(list(vertices), list(edges), cost, dist, time, fuel)


def _search(graph, model, source, target, use_heuristic, counters, observer):
    for v in (source, target):
        if v not in graph.vertices:
            raise KeyError(f"Vertex {v} does not exist")
    if use_heuristic:
        def h(v):
            return model.heuristic(v, target)
    else:
        def h(v):
            return 0.0

    dist = {source: 0.0}
    parent = {source: None}
    heap = [(h(source), source, 0.0)]        # (f, vertex, g): ties go to lower id
    if counters:
        counters.visited += 1
        counters.heap_push += 1
    notify(observer, "visit", vertex=source)
    notify(observer, "push", vertex=source)

    while heap:
        f, u, g = heapq.heappop(heap)
        if counters:
            counters.heap_pop += 1
        notify(observer, "pop", vertex=u, f=f)
        if g > dist[u]:                      # stale entry (lazy deletion)
            continue
        if counters:
            counters.expanded += 1
        notify(observer, "expand", vertex=u, g=g)
        if u == target:
            break
        for edge, v in graph.neighbors(u):
            if counters:
                counters.edges_examined += 1
            ng = g + model.edge_cost(edge)
            if v not in dist or ng < dist[v]:
                if v not in dist:
                    if counters:
                        counters.visited += 1
                    notify(observer, "visit", vertex=v, via=edge.id)
                dist[v] = ng
                parent[v] = (u, edge.id)
                if counters:
                    counters.relaxed += 1
                    counters.heap_push += 1
                notify(observer, "relax", vertex=v, via=edge.id, g=ng)
                heapq.heappush(heap, (ng + h(v), v, ng))
                notify(observer, "push", vertex=v)

    if target not in dist:
        return None
    vertices, edges = rebuild_path(parent, target)
    route = evaluate_route(graph, model, vertices, edges)
    notify(observer, "path_found", vertices=vertices, cost=route.cost)
    return route


def dijkstra(graph, model, source, target, counters=None, observer=None):
    """Optimal route for non-negative edge costs, or None if unreachable.
    Binary heap + lazy deletion: O((V+E) log V)."""
    return _search(graph, model, source, target, False, counters, observer)


def astar(graph, model, source, target, counters=None, observer=None):
    """A* with h = ratio * straight-line distance (see CostModel.heuristic).
    The heuristic is admissible and consistent. There is no closed set:
    stale heap entries are skipped by comparing g, so a vertex would simply
    be reopened if a heuristic were admissible but inconsistent. With our
    consistent heuristic that never happens."""
    return _search(graph, model, source, target, True, counters, observer)


def plan_route(graph, model, current, next_cp_index, finish, algo="astar",
               counters=None, observer=None):
    """Checkpoint-ordered plan: current -> next checkpoint -> ... -> finish.
    next_cp_index is a 0-based index into graph.checkpoints_in_order()
    (0 = none visited yet, len = all visited, only the finish remains).
    Returns a Plan, or None when some leg is unreachable (the engine then
    applies the documented recovery rule). Call it again after any road
    closure or cost change; the model reads live edge state, so it does not
    need rebuilding."""
    if algo not in ("astar", "dijkstra"):
        raise ValueError(f"Unknown algo {algo}")
    search = astar if algo == "astar" else dijkstra
    cps = [v.id for v in graph.checkpoints_in_order()]
    if not 0 <= next_cp_index <= len(cps):
        raise ValueError(f"next_cp_index must be 0..{len(cps)}")
    targets = cps[next_cp_index:] + [finish]

    legs, vertices, edges, here = [], [current], [], current
    for t in targets:
        leg = search(graph, model, here, t, counters, observer)
        if leg is None:
            return None
        legs.append((here, t, leg))
        vertices.extend(leg.vertices[1:])
        edges.extend(leg.edges)
        here = t
    total = evaluate_route(graph, model, vertices, edges)
    return Plan(total.vertices, total.edges, total.cost, total.distance,
                total.time, total.fuel, legs)


def compare_presets(graph, source, target, algo="astar"):
    """Best route under each preset: {preset_name: Route or None}."""
    search = astar if algo == "astar" else dijkstra
    return {name: search(graph, CostModel(graph, name), source, target)
            for name in PRESETS}