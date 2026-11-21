import glob
import pytest
from src.loader import load_circuit, build_circuit, CircuitError
from src.graph import FUEL_STATION, NITRO_PICKUP, DEAD_END

CIRCUIT_FILES = sorted(glob.glob("circuits/*.json"))


def tiny(extra_edge=None, cp_orders=(1, 2)):
    data = {
        "start": 0, "finish": 0,
        "vertices": [
            {"id": 0, "x": 0, "y": 0, "type": "Start"},
            {"id": 1, "x": 10, "y": 0, "type": "Checkpoint", "checkpoint_order": cp_orders[0]},
            {"id": 2, "x": 10, "y": 10, "type": "Checkpoint", "checkpoint_order": cp_orders[1]},
        ],
        "edges": [{"id": 0, "from": 0, "to": 1}, {"id": 1, "from": 1, "to": 2}],
    }
    if extra_edge:
        data["edges"].append(extra_edge)
    return data


def test_valid_tiny_circuit():
    c = build_circuit(tiny(), min_checkpoints=2)
    assert c.checkpoints == [1, 2]


def test_distance_shorter_than_straight_line_rejected():
    with pytest.raises(CircuitError):
        build_circuit(tiny({"id": 2, "from": 0, "to": 2, "distance": 5}), min_checkpoints=2)


def test_missing_endpoint_rejected():
    with pytest.raises(CircuitError):
        build_circuit(tiny({"id": 2, "from": 0, "to": 99}), min_checkpoints=2)


def test_checkpoint_gap_rejected():
    with pytest.raises(CircuitError):
        build_circuit(tiny(cp_orders=(1, 3)), min_checkpoints=2)


def test_too_few_checkpoints_rejected():
    with pytest.raises(CircuitError):
        build_circuit(tiny(), min_checkpoints=10)


@pytest.mark.parametrize("path", CIRCUIT_FILES)
def test_circuit_file_meets_spec(path):
    c = load_circuit(path)
    g = c.graph
    assert len(c.checkpoints) >= 10
    assert sum(e.is_shortcut for e in g.edges.values()) >= 2
    assert g.vertices_of_type(FUEL_STATION)
    assert g.vertices_of_type(NITRO_PICKUP)
    assert g.vertices_of_type(DEAD_END)