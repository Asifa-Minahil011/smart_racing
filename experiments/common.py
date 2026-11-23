import csv
import datetime
import gc
import math
import os
import platform
import random
import statistics
import sys
import time

from src.graph import Edge, Graph, Vertex

RESULTS_DIR = os.path.join("experiments", "results")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def time_runs(fn, runs=7, warmup=1):
    """Return a list of `runs` wall-clock times in seconds."""
    for _ in range(warmup):
        fn()
    gc.collect()
    gc.disable()                 # like timeit: keep the garbage collector out of the timings
    try:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return times


def stats(times):
    return {"mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0}


def write_csv(path, rows, fields):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_environment(outdir, runs, warmup):
    ensure_dir(outdir)
    text = "\n".join([
        f"Date:       {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"OS:         {platform.platform()}",
        f"CPU:        {platform.processor() or 'unknown'}  ({os.cpu_count()} logical cores)",
        f"Python:     {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Timer:      time.perf_counter()",
        f"Procedure:  {warmup} warm-up run(s), then {runs} timed runs per instance",
        "Excluded:   graph construction, CostModel construction, rendering, file I/O, correctness checks",
        "Included:   the algorithm's own queue/heap/dictionary initialisation",
        "GC:         Python garbage collector disabled during timed runs (as timeit does)",
        "Counters:   collected in a separate untimed run",
    ]) + "\n"
    with open(os.path.join(outdir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _add_edge(g, eid, a, b, rnd):
    va, vb = g.vertices[a], g.vertices[b]
    straight = math.hypot(va.x - vb.x, va.y - vb.y)
    d = max(straight * rnd.uniform(1.0, 1.4), 1e-6)   # never shorter than the straight line
    g.add_edge(Edge(eid, a, b, d, d * 0.1, d * 0.05,
                    difficulty=rnd.randint(1, 5)))


def random_connected_graph(n, m, seed):
    """Connected undirected graph with n vertices and exactly m >= n-1 edges:
    a random spanning tree plus (m-n+1) random extra roads. Random
    coordinates in a 1000 x 1000 square."""
    rnd = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_vertex(Vertex(i, rnd.uniform(0, 1000), rnd.uniform(0, 1000)))
    eid = 0
    for i in range(1, n):
        _add_edge(g, eid, i, rnd.randrange(i), rnd)
        eid += 1
    while eid < m:
        a, b = rnd.sample(range(n), 2)
        _add_edge(g, eid, a, b, rnd)
        eid += 1
    return g


def grid_graph(k, seed, free_highway=False):
    """k x k road grid with jittered coordinates (a city-like, geometric graph).
    free_highway=True adds one corner-to-corner road with zero travel time and
    zero fuel. It is a legal, unusually cheap road: it makes the minimum
    cost-per-length ratio tiny, so the A* heuristic becomes weak."""
    rnd = random.Random(seed)
    g = Graph()
    for r in range(k):
        for c in range(k):
            g.add_vertex(Vertex(r * k + c, c * 10 + rnd.uniform(-3, 3),
                                r * 10 + rnd.uniform(-3, 3)))
    eid = 0
    for r in range(k):
        for c in range(k):
            v = r * k + c
            if c + 1 < k:
                _add_edge(g, eid, v, v + 1, rnd)
                eid += 1
            if r + 1 < k:
                _add_edge(g, eid, v, v + k, rnd)
                eid += 1
    if free_highway:
        a, b = 0, k * k - 1
        va, vb = g.vertices[a], g.vertices[b]
        straight = math.hypot(va.x - vb.x, va.y - vb.y)
        g.add_edge(Edge(eid, a, b, straight, 0.0, 0.0, difficulty=1))
    return g