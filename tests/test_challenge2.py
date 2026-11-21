import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pytest
from experiments import challenge2_presets as c2
from src.cost import PRESETS, CostModel
from src.loader import load_circuit
from src.routing import dijkstra, evaluate_route


def test_script_runs_and_writes_files(tmp_path):
    text = c2.run(outdir=str(tmp_path), log=lambda *a: None)
    assert (tmp_path / "challenge2_report.txt").exists()
    assert (tmp_path / "challenge2_presets.csv").exists()
    assert "chosen for NO weight at all" in text


def test_each_presets_route_is_cheapest_under_its_own_cost():
    g = load_circuit("circuits/circuit1.json").graph
    models = {p: CostModel(g, p) for p in PRESETS}
    for a, b in [(3, 4), (8, 9)]:
        routes = {p: dijkstra(g, m, a, b) for p, m in models.items()}
        for p, m in models.items():
            own = routes[p].cost
            for q, r in routes.items():
                assert evaluate_route(g, m, r.vertices, r.edges).cost >= own - 1e-9


def test_weight_sweep_changes_the_route():
    g = load_circuit("circuits/circuit1.json").graph
    routes = {r for _, r in c2.sweep(g, 3, 4, [i / 20 for i in range(21)])}
    assert len(routes) >= 2


def test_compromise_road_is_never_chosen_by_any_weighting():
    g = c2.three_roads()
    picked = set()
    for i in range(101):
        wt = i / 100
        m = c2.model_with(g, {"d": 0, "t": wt, "f": 0, "r": 1 - wt, "o": 0})
        picked.add(dijkstra(g, m, 0, 1).edges[0])
    assert picked == {0, 2}          # roads A and C only; compromise B (id 1) never