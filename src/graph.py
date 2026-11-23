from dataclasses import dataclass

# Edge status values
OPEN = "Open"
BLOCKED = "Blocked"
RESTRICTED = "Restricted"

# Vertex types
START = "Start"
FINISH = "Finish"
CHECKPOINT = "Checkpoint"
INTERSECTION = "Intersection"
FUEL_STATION = "FuelStation"
NITRO_PICKUP = "NitroPickup"
DEAD_END = "DeadEnd"

VERTEX_TYPES = {START, FINISH, CHECKPOINT, INTERSECTION,
                FUEL_STATION, NITRO_PICKUP, DEAD_END}
SERVICE_TYPES = {FUEL_STATION, NITRO_PICKUP}


@dataclass
class Vertex:
    id: int
    x: float
    y: float
    type: str = INTERSECTION
    checkpoint_order: int = None   # 1..10 for mandatory checkpoints, else None
    active: bool = True            # only meaningful for service vertices


@dataclass
class Edge:
    id: int
    from_v: int
    to_v: int
    distance: float
    base_time: float
    fuel_cost: float
    difficulty: int = 1            # 1 (easy) .. 5 (hard)
    status: str = OPEN
    is_shortcut: bool = False
    directed: bool = False         # False = traversable both ways
    # Dynamic effects, set by the event system through set_edge_penalty()
    difficulty_penalty: float = 0.0   # extra difficulty from oil spill etc.
    speed_cap: float = None           # None = no cap
    obstacle_penalty: float = 0.0


class Graph:
    def __init__(self):
        self.vertices = {}   # id -> Vertex
        self.edges = {}      # id -> Edge
        self.adj = {}        # vertex id -> list of (edge_id, neighbor_id)

    # ---------- building ----------
    def add_vertex(self, vertex):
        if vertex.id in self.vertices:
            raise ValueError(f"Duplicate vertex id {vertex.id}")
        if vertex.type not in VERTEX_TYPES:
            raise ValueError(f"Unknown vertex type {vertex.type}")
        self.vertices[vertex.id] = vertex
        self.adj[vertex.id] = []

    def add_edge(self, edge):
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id {edge.id}")
        if edge.from_v not in self.vertices or edge.to_v not in self.vertices:
            raise ValueError(f"Edge {edge.id}: endpoint does not exist")
        if edge.distance < 0 or edge.base_time < 0 or edge.fuel_cost < 0:
            raise ValueError(f"Edge {edge.id}: negative attribute")
        if not 1 <= edge.difficulty <= 5:
            raise ValueError(f"Edge {edge.id}: difficulty must be 1..5")
        self.edges[edge.id] = edge
        self.adj[edge.from_v].append((edge.id, edge.to_v))
        if not edge.directed:
            # same edge id in both lists, so blocking it affects both directions
            self.adj[edge.to_v].append((edge.id, edge.from_v))

    # ---------- queries ----------
    def get_vertex(self, vid):
        return self.vertices[vid]

    def get_edge(self, eid):
        return self.edges[eid]

    def neighbors(self, vid):
        """Yield (edge, neighbor_id) for every traversable road out of vid.
        Blocked edges are skipped. Restricted edges are still returned.
        Inactive service vertices stay passable (they just give no service)."""
        for eid, nid in self.adj[vid]:
            edge = self.edges[eid]
            if edge.status != BLOCKED:
                yield edge, nid

    def all_neighbors(self, vid):
        """Like neighbors() but includes blocked edges (for display)."""
        for eid, nid in self.adj[vid]:
            yield self.edges[eid], nid

    def is_service_available(self, vid):
        v = self.vertices[vid]
        return v.type in SERVICE_TYPES and v.active

    def vertices_of_type(self, vtype):
        return [v for v in self.vertices.values() if v.type == vtype]

    def checkpoints_in_order(self):
        cps = [v for v in self.vertices.values() if v.checkpoint_order is not None]
        return sorted(cps, key=lambda v: v.checkpoint_order)

    # ---------- dynamic updates (never rebuild the graph) ----------
    def set_edge_status(self, eid, status):
        if status not in (OPEN, BLOCKED, RESTRICTED):
            raise ValueError(f"Unknown status {status}")
        if eid not in self.edges:
            raise KeyError(f"Edge {eid} does not exist")
        self.edges[eid].status = status

    def block_edge(self, eid):
        self.set_edge_status(eid, BLOCKED)

    def unblock_edge(self, eid):
        self.set_edge_status(eid, OPEN)

    def set_edge_penalty(self, eid, difficulty_penalty=0.0, speed_cap=None,
                         obstacle_penalty=0.0):
        """Set the COMBINED effect of all active temporary effects on an edge.
        The event system computes the combination (max penalty, strictest cap)."""
        if eid not in self.edges:
            raise KeyError(f"Edge {eid} does not exist")
        if difficulty_penalty < 0 or obstacle_penalty < 0:
            raise ValueError("Penalties must be non-negative")
        e = self.edges[eid]
        e.difficulty_penalty = difficulty_penalty
        e.speed_cap = speed_cap
        e.obstacle_penalty = obstacle_penalty

    def set_vertex_active(self, vid, active):
        if vid not in self.vertices:
            raise KeyError(f"Vertex {vid} does not exist")
        self.vertices[vid].active = active