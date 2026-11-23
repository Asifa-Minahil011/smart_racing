import json
import math
from dataclasses import dataclass

from src.graph import Graph, Vertex, Edge, CHECKPOINT, START

TIME_PER_UNIT = 0.1    # default base_time = distance * this
FUEL_PER_UNIT = 0.05   # default fuel_cost = distance * this


class CircuitError(Exception):
    """Raised when a circuit file is invalid."""


@dataclass
class Circuit:
    name: str
    graph: Graph
    start: int
    finish: int
    laps: int
    checkpoints: list   # vertex ids in required order


def load_circuit(path, min_checkpoints=10):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return build_circuit(data, min_checkpoints)


def build_circuit(data, min_checkpoints=10):
    g = Graph()
    try:
        for v in data["vertices"]:
            g.add_vertex(Vertex(
                id=v["id"], x=v["x"], y=v["y"],
                type=v.get("type", "Intersection"),
                checkpoint_order=v.get("checkpoint_order"),
                active=v.get("active", True)))

        for e in data["edges"]:
            a, b = e["from"], e["to"]
            if a not in g.vertices or b not in g.vertices:
                raise CircuitError(f"Edge {e['id']}: endpoint does not exist")
            va, vb = g.vertices[a], g.vertices[b]
            straight = math.hypot(va.x - vb.x, va.y - vb.y)
            distance = e.get("distance", straight)
            if distance < straight - 1e-6:
                raise CircuitError(
                    f"Edge {e['id']}: distance {distance} is shorter than the "
                    f"straight line {straight:.2f} (breaks A* admissibility)")
            g.add_edge(Edge(
                id=e["id"], from_v=a, to_v=b, distance=distance,
                base_time=e.get("base_time", distance * TIME_PER_UNIT),
                fuel_cost=e.get("fuel_cost", distance * FUEL_PER_UNIT),
                difficulty=e.get("difficulty", 1),
                is_shortcut=e.get("is_shortcut", False),
                directed=e.get("directed", False)))

        start, finish = data["start"], data["finish"]
        laps = data.get("laps", 3)
        name = data.get("name", "Unnamed circuit")
    except (KeyError, ValueError) as err:
        raise CircuitError(f"Invalid circuit data: {err}") from err

    if start not in g.vertices or finish not in g.vertices:
        raise CircuitError("start or finish vertex does not exist")
    if g.vertices[start].type != START:
        raise CircuitError("start vertex must have type Start")
    if not isinstance(laps, int) or laps < 1:
        raise CircuitError("laps must be an integer >= 1")

    cps = g.checkpoints_in_order()
    if any(v.type != CHECKPOINT for v in cps):
        raise CircuitError("checkpoint_order is set on a non-Checkpoint vertex")
    if any(v.type == CHECKPOINT and v.checkpoint_order is None
           for v in g.vertices.values()):
        raise CircuitError("a Checkpoint vertex has no checkpoint_order")
    orders = [v.checkpoint_order for v in cps]
    if orders != list(range(1, len(orders) + 1)):
        raise CircuitError(f"checkpoint orders must be 1..N with no gaps: {orders}")
    if len(cps) < min_checkpoints:
        raise CircuitError(f"need at least {min_checkpoints} checkpoints, got {len(cps)}")

    return Circuit(name, g, start, finish, laps, [v.id for v in cps])