import argparse
import glob
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import (RESULTS_DIR, grid_graph, random_connected_graph,
                                stats, time_runs, write_csv, write_environment)
from src.cost import PRESETS, CostModel
from src.counters import Counters
from src.loader import load_circuit
from src.routing import astar, dijkstra

CIRCUIT_PAIRS = [(0, 9), (1, 7), (2, 8), (3, 4), (8, 9), (0, 20), (5, 10), (11, 6)]


def build_instances(quick=False, seed=7):
    """List of (name, graph, [(source, target), ...])."""
    out = []
    for path in sorted(glob.glob("circuits/*.json")):
        c = load_circuit(path)
        ids = set(c.graph.vertices)
        pairs = [p for p in CIRCUIT_PAIRS if p[0] in ids and p[1] in ids]
        out.append((os.path.basename(path).replace(".json", ""), c.graph, pairs))
    grid_sizes = [10, 20] if quick else [10, 20, 40, 80]
    for k in grid_sizes:
        g = grid_graph(k, seed + k)
        n = k * k
        rnd = random.Random(seed + n)
        pairs = [(0, n - 1)] + [tuple(rnd.sample(range(n), 2)) for _ in range(4)]
        out.append((f"grid{k}x{k}", g, pairs))
    k = 20
    g = grid_graph(k, seed + k, free_highway=True)       # weak-heuristic case
    rnd = random.Random(seed + 99)
    out.append((f"grid{k}x{k}-weakh", g,
                [(0, k * k - 1)] + [tuple(rnd.sample(range(k * k), 2)) for _ in range(4)]))
    n = 300 if quick else 3000
    g = random_connected_graph(n, 3 * n, seed)
    rnd = random.Random(seed + 1)
    out.append((f"random{n}", g, [tuple(rnd.sample(range(n), 2)) for _ in range(5)]))
    return out


def verify_equal_costs(instances):
    """Untimed correctness pass: A* and Dijkstra must agree on every query."""
    checked = 0
    for name, g, pairs in instances:
        for preset in PRESETS:
            model = CostModel(g, preset)
            for s, t in pairs:
                d, a = dijkstra(g, model, s, t), astar(g, model, s, t)
                assert (d is None) == (a is None), f"{name}: reachability differs"
                if d is not None:
                    assert abs(d.cost - a.cost) <= 1e-9 * max(1.0, d.cost), (
                        f"{name} {preset} {s}->{t}: {d.cost} != {a.cost}")
                checked += 1
    return checked


def classify(exp_d, exp_a):
    return "fewer" if exp_a < exp_d else "more" if exp_a > exp_d else "equal"


def run(quick=False, runs=7, warmup=1, outdir=RESULTS_DIR, log=print):
    env = write_environment(outdir, runs, warmup)
    log(env)
    instances = build_instances(quick)
    checked = verify_equal_costs(instances)
    log(f"Correctness: A* cost == Dijkstra cost on all {checked} queries\n")

    raw, summary, compare = [], [], []
    for name, g, pairs in instances:
        V, E = len(g.vertices), len(g.edges)
        for preset in PRESETS:
            model = CostModel(g, preset)         # built once, outside the timing
            for s, t in pairs:
                res = {}
                for algo, fn in (("Dijkstra", dijkstra), ("A*", astar)):
                    c = Counters()               # untimed counter run
                    r = fn(g, model, s, t, counters=c)
                    times = time_runs(lambda: fn(g, model, s, t), runs, warmup)
                    for i, tm in enumerate(times, 1):
                        raw.append({"instance": name, "preset": preset,
                                    "source": s, "target": t, "algorithm": algo,
                                    "run": i, "time_s": f"{tm:.9f}"})
                    st = stats(times)
                    cost = r.cost if r else ""
                    summary.append({
                        "instance": name, "V": V, "E": E, "preset": preset,
                        "source": s, "target": t, "algorithm": algo,
                        "mean_ms": f"{st['mean'] * 1e3:.4f}",
                        "median_ms": f"{st['median'] * 1e3:.4f}",
                        "std_ms": f"{st['std'] * 1e3:.4f}",
                        "expanded": c.expanded, "relaxed": c.relaxed,
                        "heap_push": c.heap_push, "heap_pop": c.heap_pop,
                        "heap_ops": c.heap_push + c.heap_pop,
                        "final_cost": f"{cost:.9f}" if r else "unreachable"})
                    res[algo] = (c, st)
                cd, sd = res["Dijkstra"]
                ca, sa = res["A*"]
                compare.append({
                    "instance": name, "V": V, "preset": preset,
                    "source": s, "target": t,
                    "expanded_dijkstra": cd.expanded, "expanded_astar": ca.expanded,
                    "expanded_ratio": f"{ca.expanded / max(cd.expanded, 1):.4f}",
                    "astar_vs_dijkstra": classify(cd.expanded, ca.expanded),
                    "ms_dijkstra": f"{sd['mean'] * 1e3:.4f}",
                    "ms_astar": f"{sa['mean'] * 1e3:.4f}"})
        log(f"done {name:12} V={V:5d} E={E:6d}")

    sfields = ["instance", "V", "E", "preset", "source", "target", "algorithm",
               "mean_ms", "median_ms", "std_ms", "expanded", "relaxed", "heap_push",
               "heap_pop", "heap_ops", "final_cost"]
    cfields = ["instance", "V", "preset", "source", "target", "expanded_dijkstra",
               "expanded_astar", "expanded_ratio", "astar_vs_dijkstra",
               "ms_dijkstra", "ms_astar"]
    write_csv(os.path.join(outdir, "exp2_raw.csv"), raw,
              ["instance", "preset", "source", "target", "algorithm", "run", "time_s"])
    write_csv(os.path.join(outdir, "exp2_summary.csv"), summary, sfields)
    write_csv(os.path.join(outdir, "exp2_comparison.csv"), compare, cfields)
    chart(compare, os.path.join(outdir, "exp2_chart.png"))
    report(compare, log)
    return compare


def report(compare, log=print):
    counts = {"fewer": 0, "equal": 0, "more": 0}
    for r in compare:
        counts[r["astar_vs_dijkstra"]] += 1
    log(f"\nA* expanded fewer / equal / more vertices than Dijkstra in "
        f"{counts['fewer']} / {counts['equal']} / {counts['more']} of {len(compare)} queries\n")
    log(f"{'instance':12} {'preset':9} {'mean A*/Dij expanded':>21} {'mean ms Dij':>12} {'mean ms A*':>11}")
    keys = sorted({(r["instance"], r["preset"]) for r in compare},
                  key=lambda k: ([r["instance"] for r in compare].index(k[0]), k[1]))
    for inst, preset in keys:
        rows = [r for r in compare if r["instance"] == inst and r["preset"] == preset]
        log(f"{inst:12} {preset:9} "
            f"{statistics.mean(float(r['expanded_ratio']) for r in rows):21.3f} "
            f"{statistics.mean(float(r['ms_dijkstra']) for r in rows):12.3f} "
            f"{statistics.mean(float(r['ms_astar']) for r in rows):11.3f}")


def chart(compare, path):
    names = list(dict.fromkeys(r["instance"] for r in compare))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    w = 0.38
    for j, (key, label) in enumerate([("expanded", "mean vertices expanded"),
                                      ("ms", "mean time (ms)")]):
        dij, ast = [], []
        for n in names:
            rows = [r for r in compare if r["instance"] == n]
            if key == "expanded":
                dij.append(statistics.mean(r["expanded_dijkstra"] for r in rows))
                ast.append(statistics.mean(r["expanded_astar"] for r in rows))
            else:
                dij.append(statistics.mean(float(r["ms_dijkstra"]) for r in rows))
                ast.append(statistics.mean(float(r["ms_astar"]) for r in rows))
        xs = range(len(names))
        ax[j].bar([x - w / 2 for x in xs], dij, w, label="Dijkstra")
        ax[j].bar([x + w / 2 for x in xs], ast, w, label="A*")
        ax[j].set_xticks(list(xs))
        ax[j].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[j].set_yscale("log")
        ax[j].set_ylabel(label + " (log scale)")
        ax[j].legend()
        ax[j].grid(alpha=0.3, axis="y")
    ax[0].set_title("Search effort per instance (all presets and pairs)")
    ax[1].set_title("Runtime per instance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small instances, 3 runs")
    a = ap.parse_args()
    run(quick=a.quick, runs=3 if a.quick else 7)
    print("\nSaved CSV files and exp2_chart.png in", RESULTS_DIR)