import os
os.environ.setdefault("MPLBACKEND", "Agg")

from experiments import exp1_bfs_dfs, exp2_dijkstra_astar


def quiet(*a, **k):
    pass


def test_exp1_runs_and_counters_are_linear(tmp_path):
    rows = exp1_bfs_dfs.run(sizes=[50, 120], runs=2, outdir=str(tmp_path), log=quiet)
    for name in ("exp1_raw.csv", "exp1_summary.csv", "exp1_chart.png", "environment.txt"):
        assert (tmp_path / name).exists()
    assert len(rows) == 2 * 3 * 2            # sizes x densities x algorithms
    for r in rows:
        assert r["visited"] == r["V"]                     # every vertex once
        assert r["edges_examined"] == 2 * r["E"]          # each road seen from both ends


def test_exp1_verifiers_detect_a_wrong_answer():
    from experiments.common import random_connected_graph
    g = random_connected_graph(30, 60, seed=1)
    assert exp1_bfs_dfs.verify_bfs_levels(g, 0) == 30
    exp1_bfs_dfs.verify_dfs(g, 0)


def test_exp2_runs_and_costs_match(tmp_path):
    rows = exp2_dijkstra_astar.run(quick=True, runs=2, outdir=str(tmp_path), log=quiet)
    for name in ("exp2_raw.csv", "exp2_summary.csv", "exp2_comparison.csv",
                 "exp2_chart.png", "environment.txt"):
        assert (tmp_path / name).exists()
    assert rows
    assert {r["astar_vs_dijkstra"] for r in rows} <= {"fewer", "equal", "more"}
    assert any(r["instance"].endswith("weakh") for r in rows)