"""Edge cost, normalization and presets. Owner: Person A.
One CostModel is used by the algorithms AND the UI, so the number shown
to the player is exactly the number that was optimized."""

import math
from src.graph import RESTRICTED

# Weights (d, t, f, r, o) = distance, time, fuel, difficulty, obstacle.
# All non-negative. DRAFT values: tune, then freeze and document.
PRESETS = {
    "Balanced": {"d": 0.2, "t": 0.2, "f": 0.2, "r": 0.2,  "o": 0.2},
    "Speed":    {"d": 0.1, "t": 0.6, "f": 0.1, "r": 0.1,  "o": 0.1},
    "Economy":  {"d": 0.1, "t": 0.1, "f": 0.6, "r": 0.1,  "o": 0.1},
    "Safety":   {"d": 0.1, "t": 0.1, "f": 0.1, "r": 0.35, "o": 0.35},
}

RESTRICTED_TIME_FACTOR = 1.5   # restricted road: travel time x1.5
OBSTACLE_REF = 10.0            # obstacle penalty that maps to O = 1


def fuel_multiplier(difficulty):
    """Game rule: 1.0 + 0.1 * (difficulty - 1)."""
    return 1.0 + 0.1 * (difficulty - 1)


class CostModel:
    def __init__(self, graph, preset="Balanced"):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        w = PRESETS[preset]
        if any(x < 0 for x in w.values()):
            raise ValueError("Weights must be non-negative")
        self.graph = graph
        self.preset = preset
        self.w = w
        edges = list(graph.edges.values())
        # Normalization references: maximum over the circuit's BASE values
        # (computed once, so costs do not shift when events happen).
        self.d_ref = max((e.distance for e in edges), default=1.0) or 1.0
        self.t_ref = max((e.base_time for e in edges), default=1.0) or 1.0
        self.f_ref = max((e.fuel_cost * fuel_multiplier(e.difficulty)
                          for e in edges), default=1.0) or 1.0
        self.ratio = self._heuristic_ratio(edges)

    # ---------- effective edge attributes ----------
    def effective_time(self, e):
        t = e.base_time
        if e.status == RESTRICTED:
            t *= RESTRICTED_TIME_FACTOR
        if e.speed_cap is not None and e.base_time > 0 and e.distance > 0:
            normal_speed = e.distance / e.base_time
            if e.speed_cap < normal_speed:
                t *= normal_speed / max(e.speed_cap, 1e-9)
        return t

    def effective_fuel(self, e):
        return e.fuel_cost * fuel_multiplier(e.difficulty)

    def components(self, e):
        """Normalized factors D, T, F, R, O (for display and for the cost)."""
        return {
            "d": e.distance / self.d_ref,
            "t": self.effective_time(e) / self.t_ref,
            "f": self.effective_fuel(e) / self.f_ref,
            "r": (e.difficulty + e.difficulty_penalty - 1) / 4.0,
            "o": e.obstacle_penalty / OBSTACLE_REF,
        }

    # ---------- the one cost function ----------
    def edge_cost(self, e):
        c = self.components(e)
        return sum(self.w[k] * c[k] for k in ("d", "t", "f", "r", "o"))

    # ---------- A* support ----------
    def _static_cost(self, e):
        """Cost with NO dynamic effects. Lower bound of edge_cost, because
        restrictions, penalties and speed caps can only increase cost."""
        w = self.w
        return (w["d"] * e.distance / self.d_ref
                + w["t"] * e.base_time / self.t_ref
                + w["f"] * self.effective_fuel(e) / self.f_ref
                + w["r"] * (e.difficulty - 1) / 4.0)

    def _heuristic_ratio(self, edges):
        """min over edges of static_cost / straight-line length.
        Edges with zero straight length are skipped."""
        best = None
        for e in edges:
            a = self.graph.vertices[e.from_v]
            b = self.graph.vertices[e.to_v]
            straight = math.hypot(a.x - b.x, a.y - b.y)
            if straight <= 0:
                continue
            r = self._static_cost(e) / straight
            best = r if best is None else min(best, r)
        return best or 0.0   # 0 means h = 0 (A* behaves like Dijkstra)

    def heuristic(self, vid, target):
        a = self.graph.vertices[vid]
        b = self.graph.vertices[target]
        return self.ratio * math.hypot(a.x - b.x, a.y - b.y)