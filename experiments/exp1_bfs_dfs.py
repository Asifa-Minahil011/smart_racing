"""Experiment 1: BFS and DFS vs input size and density. Owner: Person A.

Run from the project root:   python -m experiments.exp1_bfs_dfs
Quick test run:              python -m experiments.exp1_bfs_dfs --quick
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import (RESULTS_DIR, random_connected_graph, stats,
                                time_runs, write_csv, write_environment)
from src.counters import Counters
from src.graph import Edge, Graph, Vertex
from src.search import (bfs_tree, connected_components, dfs_traversal,
                        find_cycle)

SIZES = [100, 500, 1000, 2000, 5000, 10000]
DENSITIES = [("sparse", 2), ("medium", 5), ("dense", 10)]   # E = factor * V


# ---------------- correctness checks (never timed) ----------------
def verify_bfs_levels(g, start):
    """Certificate that BFS distances are minimum edge counts: every tree
    path has length = level, and no road joins vertices whose levels differ
    by more than 1 (so no shorter path can exist)."""
    parent = bfs_tree(g, start)
    level = {}
    for v, p in parent.items():                 # dict keeps discovery order
        level[v] = 0 if p is None else level[p[0]] + 1
    for e in g.edges.values():
        if e.from_v in level or e.to_v in level:
            assert e.from_v in level and e.to_v in level, "BFS missed a vertex"
            assert abs(level[e.from_v] - level[e.to_v]) <= 1, "levels differ by > 1"
    return len(level)


def verify_dfs(g, start):
    n, m = len(g.vertices), len(g.edges)
    order, _, _ = dfs_traversal(g, start)
    assert len(order) == len(set(order)) == n, "DFS must visit every vertex once"
    assert len(connected_components(g)) == 1, "generator must give 1 component"
    cycle = find_cycle(g)
    assert (cycle is not None) == (m >= n), "cycle iff E >= V (connected graph)"


def verify_small_cases():
    tree = random_connected_graph(50, 49, seed=3)           # no cycle
    assert find_cycle(tree) is None
    g = Graph()                                             # two components
    for i in range(6):
        g.add_vertex(Vertex(i, i, 0))
    for k, (a, b) in enumerate([(0, 1), (1, 2), (3, 4), (4, 5)]):
        g.add_edge(Edge(k, a, b, 1, 1, 1))
    assert len(connected_components(g)) == 2
    assert len(bfs_tree(g, 0)) == 3


# ---------------- the experiment ----------------
def run(sizes=SIZES, runs=7, warmup=1, outdir=RESULTS_DIR, seed=1, log=print):
    verify_small_cases()
    env = write_environment(outdir, runs, warmup)
    log(env)
    raw, summary = [], []
    for n in sizes:
        for dname, factor in DENSITIES:
            m = factor * n
            g = random_connected_graph(n, m, seed + n)
            verify_bfs_levels(g, 0)
            verify_dfs(g, 0)
            for algo, fn in (("BFS", lambda: bfs_tree(g, 0)),
                             ("DFS", lambda: dfs_traversal(g, 0))):
                c = Counters()                                   # untimed counter run
                (bfs_tree if algo == "BFS" else dfs_traversal)(g, 0, counters=c)
                times = time_runs(fn, runs, warmup)
                for i, t in enumerate(times, 1):
                    raw.append({"V": n, "E": m, "density": dname, "algorithm": algo,
                                "run": i, "time_s": f"{t:.9f}"})
                s = stats(times)
                summary.append({
                    "V": n, "E": m, "V_plus_E": n + m, "density": dname,
                    "algorithm": algo,
                    "mean_ms": f"{s['mean'] * 1e3:.4f}",
                    "median_ms": f"{s['median'] * 1e3:.4f}",
                    "std_ms": f"{s['std'] * 1e3:.4f}",
                    "visited": c.visited, "edges_examined": c.edges_examined,
                    "us_per_V_plus_E": f"{s['mean'] * 1e6 / (n + m):.4f}"})
                log(f"V={n:6d} E={m:7d} {dname:6} {algo}  "
                    f"mean {s['mean'] * 1e3:9.3f} ms  visited {c.visited}  "
                    f"examined {c.edges_examined}")
    write_csv(os.path.join(outdir, "exp1_raw.csv"), raw,
              ["V", "E", "density", "algorithm", "run", "time_s"])
    fields = ["V", "E", "V_plus_E", "density", "algorithm", "mean_ms", "median_ms",
              "std_ms", "visited", "edges_examined", "us_per_V_plus_E"]
    write_csv(os.path.join(outdir, "exp1_summary.csv"), summary, fields)
    chart(summary, os.path.join(outdir, "exp1_chart.png"))
    return summary


def chart(summary, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {"sparse": "o", "medium": "s", "dense": "^"}
    colors = {"BFS": "tab:blue", "DFS": "tab:orange"}
    for algo in ("BFS", "DFS"):
        for dname in styles:
            pts = sorted((r["V_plus_E"], float(r["mean_ms"]), float(r["us_per_V_plus_E"]))
                         for r in summary if r["algorithm"] == algo and r["density"] == dname)
            xs = [p[0] for p in pts]
            ax[0].plot(xs, [p[1] for p in pts], marker=styles[dname], color=colors[algo],
                       alpha=0.8, label=f"{algo} {dname}")
            ax[1].plot(xs, [p[2] for p in pts], marker=styles[dname], color=colors[algo],
                       alpha=0.8, label=f"{algo} {dname}")
    ax[0].set(xlabel="V + E", ylabel="mean time (ms)", title="Runtime vs V + E")
    ax[1].set(xlabel="V + E", ylabel="microseconds per (V + E)",
              title="Time per unit of V + E (flat = linear growth)")
    ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, 3 runs")
    a = ap.parse_args()
    if a.quick:
        run(sizes=[100, 500], runs=3)
    else:
        run()
    print("\nSaved CSV files and exp1_chart.png in", RESULTS_DIR)