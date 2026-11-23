import os
import random

from experiments.common import RESULTS_DIR, ensure_dir, write_csv
from src.cost import PRESETS, CostModel
from src.graph import Edge, Graph, Vertex
from src.loader import load_circuit
from src.routing import dijkstra, evaluate_route
from src.search import alternative_routes

KEYS = ("d", "t", "f", "r", "o")


def model_with(graph, weights):
    """CostModel with custom weights. Only use it with dijkstra: the A*
    heuristic ratio inside the model was computed for the Balanced preset."""
    m = CostModel(graph, "Balanced")
    m.w = dict(weights)
    return m


def fmt(vertices):
    return ">".join(map(str, vertices))


# ---------- part 1: presets, and routes re-costed under every preset ----------
def part1(g, legs, lines, rows):
    models = {p: CostModel(g, p) for p in PRESETS}
    for a, b in legs:
        lines.append(f"\nLeg {a} -> {b}")
        lines.append(f"{'chosen by':10} {'route':22} {'dist':>5} {'time':>6} {'fuel':>6}   "
                     + "  ".join(f"cost[{p[:3]}]" for p in PRESETS))
        for name, m in models.items():
            r = dijkstra(g, m, a, b)
            cross = [evaluate_route(g, models[p], r.vertices, r.edges).cost for p in PRESETS]
            lines.append(f"{name:10} {fmt(r.vertices):22} {r.distance:5.0f} {r.time:6.1f} "
                         f"{r.fuel:6.1f}   " + "  ".join(f"{c:9.3f}" for c in cross))
            rows.append({"leg": f"{a}-{b}", "chosen_by": name, "route": fmt(r.vertices),
                         "distance": f"{r.distance:.1f}", "time": f"{r.time:.2f}",
                         "fuel": f"{r.fuel:.2f}",
                         **{f"cost_{p}": f"{c:.4f}" for p, c in zip(PRESETS, cross)}})
    lines.append("\nRead down a column: the route chosen by a preset is the cheapest one "
                 "under that preset's own cost,\nand it costs more under the other presets.")


# ---------- part 2: weight sweep ----------
def sweep(g, a, b, wt_values, others="even"):
    """Raise the time weight from 0 to 1; the other four weights share the rest."""
    out = []
    for wt in wt_values:
        rest = (1 - wt) / 4
        w = {"d": rest, "t": wt, "f": rest, "r": rest, "o": rest}
        r = dijkstra(g, model_with(g, w), a, b)
        out.append((wt, tuple(r.vertices)))
    return out


def part2(g, legs, lines, rows):
    steps = [i / 20 for i in range(21)]
    for a, b in legs:
        lines.append(f"\nWeight sweep, leg {a} -> {b}: time weight 0 -> 1, "
                     "other four weights share the remainder equally")
        prev = None
        for wt, route in sweep(g, a, b, steps):
            if route != prev:
                lines.append(f"  from w_time = {wt:.2f}: route {fmt(route)}")
                rows.append({"leg": f"{a}-{b}", "chosen_by": f"sweep w_t>={wt:.2f}",
                             "route": fmt(route)})
                prev = route
        if len({r for _, r in sweep(g, a, b, steps)}) == 1:
            lines.append("  the route never changes on this leg")


# ---------- part 3a: Pareto analysis on a real leg ----------
def objectives(g, model, route):
    """(time, fuel, risk): three genuinely different physical-ish criteria."""
    t = sum(model.effective_time(g.edges[e]) for e in route.edges)
    f = sum(model.effective_fuel(g.edges[e]) for e in route.edges)
    r = sum((g.edges[e].difficulty - 1) / 4 for e in route.edges)
    return (t, f, r)


def dominates(p, q):
    return all(x <= y + 1e-12 for x, y in zip(p, q)) and any(x < y - 1e-12 for x, y in zip(p, q))


def part3a(g, a, b, lines, samples=3000, seed=5):
    base = CostModel(g, "Balanced")
    found = alternative_routes(g, a, b, None, max_paths=3000, max_edges=8)
    routes = [evaluate_route(g, base, v, e) for v, e in found]
    objs = [objectives(g, base, r) for r in routes]
    front = [i for i, p in enumerate(objs)
             if not any(dominates(q, p) for j, q in enumerate(objs) if j != i)]
    rnd = random.Random(seed)
    chosen = set()
    for _ in range(samples):
        w = [rnd.random() for _ in range(3)]
        chosen.add(min(range(len(objs)),
                       key=lambda i: sum(x * y for x, y in zip(w, objs[i]))))
    unsupported = [i for i in front if i not in chosen]
    lines.append(f"\nPareto analysis, leg {a} -> {b}: {len(routes)} candidate simple paths "
                 f"(<= 8 roads), objectives (time, fuel, risk)")
    lines.append(f"  Pareto-optimal routes: {len(front)}")
    lines.append(f"  chosen by at least one of {samples} random weight vectors: "
                 f"{len([i for i in front if i in chosen])}")
    if unsupported:
        lines.append(f"  Pareto-optimal but NEVER chosen: {len(unsupported)}")
        for i in unsupported[:3]:
            lines.append(f"    {fmt(routes[i].vertices)}  "
                         "(time, fuel, risk) = (%.1f, %.1f, %.1f)" % objs[i])
    else:
        lines.append("  Every Pareto-optimal route on this leg is chosen by some weights, "
                     "so this circuit shows no such gap; the constructed example below does.")


# ---------- part 3b: constructed example of an unsupported compromise ----------
def three_roads():
    g = Graph()
    g.add_vertex(Vertex(0, 0, 0, "Start"))
    g.add_vertex(Vertex(1, 100, 0, "Intersection"))
    g.add_edge(Edge(0, 0, 1, 100, 10, 5, difficulty=5))   # A: fast but dangerous
    g.add_edge(Edge(1, 0, 1, 100, 36, 5, difficulty=3))   # B: compromise
    g.add_edge(Edge(2, 0, 1, 100, 50, 5, difficulty=1))   # C: slow but safe
    return g


def part3b(lines, rows):
    g = three_roads()
    names = {0: "A fast+dangerous", 1: "B compromise", 2: "C slow+safe"}
    lines.append("\nConstructed example: three parallel roads, weights only on time and risk")
    m0 = CostModel(g, "Balanced")
    for e in g.edges.values():
        c = m0.components(e)
        lines.append(f"  {names[e.id]:18} normalised time {c['t']:.2f}  risk {c['r']:.2f}")
    picks = {}
    for i in range(101):
        wt = i / 100
        w = {"d": 0, "t": wt, "f": 0, "r": 1 - wt, "o": 0}
        r = dijkstra(g, model_with(g, w), 0, 1)
        picks.setdefault(names[r.edges[0]], []).append(wt)
    for name in names.values():
        ws = picks.get(name)
        lines.append(f"  {name:18} "
                     + (f"chosen for w_time in [{min(ws):.2f}, {max(ws):.2f}]" if ws
                        else "chosen for NO weight at all (101 values tried)"))
        rows.append({"leg": "constructed", "chosen_by": name,
                     "route": "chosen for " + (f"w_time {min(ws):.2f}-{max(ws):.2f}" if ws else "no weight")})
    lines.append("  B is not dominated (A is faster, C is safer) yet no weighted sum ever picks it:")
    lines.append("  it lies above the straight line joining A and C in (time, risk) space.")


def run(circuit="circuits/circuit1.json", outdir=RESULTS_DIR, log=print):
    ensure_dir(outdir)
    g = load_circuit(circuit).graph
    legs = [(3, 4), (8, 9)]
    lines, rows = ["CHALLENGE 2: multi-objective route planning", "=" * 44], []
    lines.append("\nPART 1. Frozen preset weights (d, t, f, r, o):")
    for p, w in PRESETS.items():
        lines.append(f"  {p:9} " + "  ".join(f"{k}={w[k]:.2f}" for k in KEYS))
    part1(g, legs, lines, rows)
    lines.append("\nPART 2. Changing weights changes the chosen route")
    part2(g, legs, lines, rows)
    lines.append("\nPART 3. Limitation of one weighted scalar cost")
    part3a(g, 3, 4, lines)
    part3b(lines, rows)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "challenge2_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    fields = sorted({k for r in rows for k in r},
                    key=lambda k: ["leg", "chosen_by", "route", "distance", "time", "fuel"].index(k)
                    if k in ["leg", "chosen_by", "route", "distance", "time", "fuel"] else 99)
    write_csv(os.path.join(outdir, "challenge2_presets.csv"), rows, fields)
    log(text)
    return text


if __name__ == "__main__":
    run()
    print("Saved challenge2_report.txt and challenge2_presets.csv in", RESULTS_DIR)