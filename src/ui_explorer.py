"""Track Explorer / Route Planner screen (Pygame). Owner: Asifa Minahil.
Controls:
  left click    pick the DESTINATION vertex
  shift + click pick the START vertex (default: the circuit start)
  right click   reset start and clear the destination
  buttons       algorithm, preset, Block, Plan lap, Reset, Next map
  Block mode    click a road to block / unblock it; the route is recalculated
  keys          1-4 algorithm, B block mode, L plan lap, R reset, N next circuit, Esc quit
"""

import glob
import math
import sys

import pygame

from src.cost import PRESETS, CostModel
from src.counters import Counters
from src.graph import BLOCKED, RESTRICTED
from src.loader import load_circuit
from src.routing import astar, dijkstra, evaluate_route, plan_route
from src.search import (alternative_routes, bfs_path, connected_components,
                        dead_ends, dfs_traversal, find_cycle,
                        reachable_checkpoints, unreachable_vertices)

W, H = 1280, 720
MAP_W = 950
PANEL_X = 960
ALGOS = ["BFS", "DFS", "Dijkstra", "A*"]

BG = (24, 26, 32)
PANEL_BG = (34, 37, 46)
TEXT = (225, 228, 235)
DIM = (140, 146, 160)
ROAD = (120, 126, 140)
ROUTE = (60, 220, 130)
ALT = (170, 120, 255)
BLOCK = (230, 60, 60)
RESTR = (240, 170, 40)
VCOL = {"Start": (60, 200, 90), "Finish": (170, 90, 220),
        "Checkpoint": (70, 140, 255), "Intersection": (150, 154, 165),
        "FuelStation": (235, 70, 70), "NitroPickup": (255, 160, 40),
        "DeadEnd": (20, 20, 20)}


def fit(font, text, max_w):
    """Shorten text with '...' so it fits in max_w pixels."""
    if font.size(text)[0] <= max_w:
        return text
    while text and font.size(text + "...")[0] > max_w:
        text = text[:-1]
    return text + "..."


def dist_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def draw_dashed(surf, color, a, b, width=2, dash=9):
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return
    n = int(length // dash)
    for i in range(0, n, 2):
        p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        q = (a[0] + (b[0] - a[0]) * min(i + 1, n) / n,
             a[1] + (b[1] - a[1]) * min(i + 1, n) / n)
        pygame.draw.line(surf, color, p, q, width)


class Explorer:
    def __init__(self, files=None):
        self.files = files or sorted(glob.glob("circuits/*.json"))
        if not self.files:
            raise FileNotFoundError("no circuits/*.json files found")
        self.font = pygame.font.SysFont("consolas", 15)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 17, bold=True)
        self.algo = "Dijkstra"
        self.preset = "Balanced"
        self.block_mode = False
        self.lap_mode = False
        self.index = 0
        self.hover_edge = None
        self.load(0)

    # ---------------- state ----------------
    def load(self, i):
        self.index = i % len(self.files)
        self.circuit = load_circuit(self.files[self.index])
        self.start = self.circuit.start
        self.dest = None
        self.msg = ""
        self._layout()
        self.compute()

    def _layout(self):
        vs = list(self.circuit.graph.vertices.values())
        xs, ys = [v.x for v in vs], [v.y for v in vs]
        pad = 35
        sx = (MAP_W - 2 * pad) / max(max(xs) - min(xs), 1)
        sy = (H - 2 * pad - 20) / max(max(ys) - min(ys), 1)
        self.scale = min(sx, sy)
        self.ox = pad - min(xs) * self.scale
        self.oy = pad - min(ys) * self.scale
        self.pos = {v.id: (int(v.x * self.scale + self.ox),
                           int(v.y * self.scale + self.oy)) for v in vs}

    # ---------------- the algorithms (all real calls) ----------------
    def compute(self):
        g = self.circuit.graph
        self.model = CostModel(g, self.preset)
        self.route = self.plan = self.cycle = None
        self.alts, self.visited = [], set()
        self.counters = self.optimal = self.cmp = None
        self.dfs_order, self.deadends, self.components = [], [], []
        self.reach = reachable_checkpoints(g, self.start)
        self.unreach = unreachable_vertices(g, self.start)
        fin = self.circuit.finish

        if self.lap_mode:
            algo = "astar" if self.algo == "A*" else "dijkstra"
            self.counters = Counters()
            self.plan = plan_route(g, self.model, self.start, 0, fin, algo,
                                   counters=self.counters)
            return
        if self.dest is None:
            return
        s, d = self.start, self.dest

        if self.algo == "DFS":
            self.counters = Counters()
            order, _, _ = dfs_traversal(g, s, counters=self.counters)
            self.dfs_order, self.visited = order, set(order)
            self.cycle = find_cycle(g)
            self.deadends = dead_ends(g, fin)
            self.components = connected_components(g)
            return

        self.counters = Counters()
        if self.algo == "BFS":
            r = bfs_path(g, s, d, counters=self.counters)
            self.route = evaluate_route(g, self.model, *r) if r else None
            self.optimal = dijkstra(g, self.model, s, d)
        else:
            fn = astar if self.algo == "A*" else dijkstra
            self.route = fn(g, self.model, s, d, counters=self.counters)
            cd, ca = Counters(), Counters()
            dijkstra(g, self.model, s, d, counters=cd)
            astar(g, self.model, s, d, counters=ca)
            self.cmp = (cd.expanded, ca.expanded)

        if self.route and self.route.edges:
            # Bounded search (enumerating all simple paths is exponential):
            # simple paths at most 4 roads longer than the main route,
            # first 300 found, then ranked by cost and the best 3 shown.
            found = alternative_routes(g, s, d, self.route.edges,
                                       max_paths=300,
                                       max_edges=len(self.route.edges) + 4)
            alts = [evaluate_route(g, self.model, v, e) for v, e in found]
            self.alts = sorted(alts, key=lambda r: r.cost)[:3]

    # ---------------- input ----------------
    def buttons(self):
        out, x = [], PANEL_X
        for a in ALGOS:
            out.append((pygame.Rect(x, 40, 72, 26), a, ("algo", a), self.algo == a))
            x += 76
        x = PANEL_X
        for p in PRESETS:
            out.append((pygame.Rect(x, 92, 72, 26), p, ("preset", p), self.preset == p))
            x += 76
        row = [("Block", ("block",), self.block_mode),
               ("Plan lap", ("lap",), self.lap_mode),
               ("Reset", ("reset",), False),
               ("Next map", ("next",), False)]
        x = PANEL_X
        for label, act, on in row:
            out.append((pygame.Rect(x, 130, 72, 26), label, act, on))
            x += 76
        return out

    def act(self, a):
        if a[0] == "algo":
            self.algo = a[1]
        elif a[0] == "preset":
            self.preset = a[1]
        elif a[0] == "block":
            self.block_mode = not self.block_mode
            self.msg = "Block mode ON: click a road" if self.block_mode else ""
        elif a[0] == "lap":
            self.lap_mode = not self.lap_mode
        elif a[0] == "reset":
            self.load(self.index)
            return
        elif a[0] == "next":
            self.load(self.index + 1)
            return
        self.compute()

    def vertex_at(self, p):
        for vid, q in self.pos.items():
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= 13:
                return vid
        return None

    def edge_at(self, p, tol=7):
        best, best_d = None, tol
        for e in self.circuit.graph.edges.values():
            d = dist_to_segment(p, self.pos[e.from_v], self.pos[e.to_v])
            if d < best_d:
                best, best_d = e, d
        return best

    def handle(self, ev):
        """Returns False when the window should close."""
        g = self.circuit.graph
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return False
            keymap = {pygame.K_1: ("algo", "BFS"), pygame.K_2: ("algo", "DFS"),
                      pygame.K_3: ("algo", "Dijkstra"), pygame.K_4: ("algo", "A*"),
                      pygame.K_b: ("block",), pygame.K_l: ("lap",),
                      pygame.K_r: ("reset",), pygame.K_n: ("next",)}
            if ev.key in keymap:
                self.act(keymap[ev.key])
        elif ev.type == pygame.MOUSEMOTION:
            self.hover_edge = self.edge_at(ev.pos) if ev.pos[0] < MAP_W else None
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 3:
                self.start, self.dest = self.circuit.start, None
                self.compute()
            elif ev.button == 1:
                for rect, _, a, _ in self.buttons():
                    if rect.collidepoint(ev.pos):
                        self.act(a)
                        return True
                if ev.pos[0] >= MAP_W:
                    return True
                if self.block_mode:
                    e = self.edge_at(ev.pos)
                    if e:
                        if e.status == BLOCKED:
                            g.unblock_edge(e.id)
                            self.msg = f"Road {e.id} reopened: route recalculated"
                        else:
                            g.block_edge(e.id)
                            self.msg = f"Road {e.id} blocked: route recalculated"
                        self.compute()
                    return True
                v = self.vertex_at(ev.pos)
                if v is not None:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.start = v
                    else:
                        self.dest = v
                    self.compute()
        return True

    # ---------------- drawing ----------------
    def draw(self, screen):
        g = self.circuit.graph
        screen.fill(BG)
        route_edges = set(self.route.edges if self.route else
                          self.plan.edges if self.plan else [])
        alt_edges = {eid for r in self.alts for eid in r.edges}

        for e in g.edges.values():
            a, b = self.pos[e.from_v], self.pos[e.to_v]
            if e.status == BLOCKED:
                pygame.draw.line(screen, BLOCK, a, b, 3)
                m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                pygame.draw.line(screen, BLOCK, (m[0] - 6, m[1] - 6), (m[0] + 6, m[1] + 6), 3)
                pygame.draw.line(screen, BLOCK, (m[0] - 6, m[1] + 6), (m[0] + 6, m[1] - 6), 3)
                continue
            color = RESTR if e.status == RESTRICTED else ROAD
            if e.is_shortcut:
                draw_dashed(screen, (255, 120, 120), a, b, 2)
            else:
                pygame.draw.line(screen, color, a, b, 2)
            if e.directed:
                self._arrow(screen, a, b)
        for e in g.edges.values():
            if e.id in alt_edges and e.id not in route_edges:
                pygame.draw.line(screen, ALT, self.pos[e.from_v], self.pos[e.to_v], 4)
        for eid in route_edges:
            e = g.edges[eid]
            pygame.draw.line(screen, ROUTE, self.pos[e.from_v], self.pos[e.to_v], 6)

        for vid, v in g.vertices.items():
            p = self.pos[vid]
            col = VCOL[v.type]
            if vid in self.visited:
                pygame.draw.circle(screen, (255, 230, 90), p, 15, 2)
            pygame.draw.circle(screen, col, p, 10)
            if v.type == "DeadEnd":
                pygame.draw.circle(screen, (200, 200, 200), p, 10, 1)
            label = f"C{v.checkpoint_order}" if v.checkpoint_order else str(vid)
            t = self.small.render(label, True, TEXT)
            screen.blit(t, (p[0] + 11, p[1] - 20))
        if self.start is not None:
            pygame.draw.circle(screen, (60, 255, 120), self.pos[self.start], 16, 3)
        if self.dest is not None:
            pygame.draw.circle(screen, (255, 90, 90), self.pos[self.dest], 16, 3)

        self._legend(screen)
        self._panel(screen)

    def _arrow(self, screen, a, b):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        pts = [(mx + 9 * math.cos(ang), my + 9 * math.sin(ang)),
               (mx + 9 * math.cos(ang + 2.5), my + 9 * math.sin(ang + 2.5)),
               (mx + 9 * math.cos(ang - 2.5), my + 9 * math.sin(ang - 2.5))]
        pygame.draw.polygon(screen, ROAD, pts)

    def _legend(self, screen):
        x = 12
        for name, col in [("Start", VCOL["Start"]), ("Checkpoint", VCOL["Checkpoint"]),
                          ("Fuel", VCOL["FuelStation"]), ("Nitro", VCOL["NitroPickup"]),
                          ("Dead end", (200, 200, 200))]:
            pygame.draw.circle(screen, col, (x + 5, H - 14), 5)
            t = self.small.render(name, True, DIM)
            screen.blit(t, (x + 14, H - 21))
            x += t.get_width() + 30
        t = self.small.render("dashed = shortcut   arrow = one-way   X = blocked",
                              True, DIM)
        screen.blit(t, (x + 10, H - 21))
        if self.hover_edge:
            e = self.hover_edge
            c = self.model.edge_cost(e)
            s = (f"Road {e.id}: {e.from_v}-{e.to_v}  dist {e.distance:.0f}  "
                 f"diff {e.difficulty}  {e.status}  cost[{self.preset}] {c:.3f}")
            screen.blit(self.font.render(s, True, TEXT), (12, 8))
        elif self.msg:
            screen.blit(self.font.render(self.msg, True, (255, 220, 120)), (12, 8))

    def _panel(self, screen):
        pw = W - PANEL_X - 10
        pygame.draw.rect(screen, PANEL_BG, (PANEL_X - 10, 0, W - PANEL_X + 10, H))
        screen.blit(self.big.render(self.circuit.name, True, TEXT), (PANEL_X, 4))
        screen.blit(self.small.render("Algorithm", True, DIM), (PANEL_X, 25))
        screen.blit(self.small.render("Preset (cost weights)", True, DIM), (PANEL_X, 77))
        for rect, label, _, on in self.buttons():
            pygame.draw.rect(screen, (70, 110, 200) if on else (55, 59, 72), rect, border_radius=4)
            t = self.small.render(label, True, TEXT)
            screen.blit(t, t.get_rect(center=rect.center))

        y = 170
        lines = []

        def add(text, color=TEXT):
            lines.append((text, color))

        sname = f"v{self.start}" if self.start is not None else "-"
        dname = f"v{self.dest}" if self.dest is not None else "-"
        add(f"Start {sname}   Destination {dname}", DIM)

        if self.lap_mode:
            if self.plan is None:
                add("Lap plan: a checkpoint is unreachable", BLOCK)
            else:
                p = self.plan
                add(f"LAP PLAN through {len(p.legs) - 1} checkpoints + finish", ROUTE)
                add(f"Distance {p.distance:.0f}  Time {p.time:.1f}  Fuel {p.fuel:.1f}")
                add(f"Exact cost [{self.preset}]: {p.cost:.3f}", ROUTE)
                for a, b, r in p.legs[:6]:
                    add(f" leg {a}->{b}: {r.cost:.3f}", DIM)
                if len(p.legs) > 6:
                    add(f" ... {len(p.legs) - 6} more legs", DIM)
        elif self.dest is None:
            add("Click a destination vertex.", DIM)
        elif self.algo == "DFS":
            add(f"DFS from v{self.start}: {len(self.visited)} vertices visited")
            add("Reaches destination: " + ("yes" if self.dest in self.visited else "NO"),
                ROUTE if self.dest in self.visited else BLOCK)
            add("Order: " + ", ".join(map(str, self.dfs_order[:14])) +
                (" ..." if len(self.dfs_order) > 14 else ""), DIM)
            add("Cycle: " + (" > ".join(map(str, self.cycle)) if self.cycle else "none"))
            add("Dead ends: " + (", ".join(map(str, self.deadends)) or "none"))
            add(f"Components: {len(self.components)}")
        elif self.route is None:
            add("NO ROUTE: destination unreachable", BLOCK)
        else:
            r = self.route
            add(f"{self.algo} route, {len(r.edges)} roads", ROUTE)
            add("Path: " + " > ".join(map(str, r.vertices)))
            add(f"Distance {r.distance:.0f}  Time {r.time:.1f}  Fuel {r.fuel:.1f}")
            add(f"Exact cost [{self.preset}]: {r.cost:.3f}", ROUTE)
            if self.algo == "BFS" and self.optimal:
                o = self.optimal
                add(f"BFS = fewest roads, NOT cheapest.", RESTR)
                add(f"Cheapest (Dijkstra): {o.cost:.3f}  ({len(o.edges)} roads)", RESTR)
            if self.cmp:
                add(f"Expanded: Dijkstra {self.cmp[0]}  A* {self.cmp[1]}", DIM)
            if self.alts:
                add("Alternatives (short simple paths, by cost):", TEXT)
                for i, a in enumerate(self.alts, 1):
                    add(f" {i}. {a.cost:.3f} d{a.distance:.0f} "
                        + ">".join(map(str, a.vertices)), ALT)
            elif self.route.edges:
                add("No alternative route exists.", DIM)

        if self.counters:
            c = self.counters
            add(f"Counters: visited {c.visited} expanded {c.expanded}", DIM)
            add(f" examined {c.edges_examined} relaxed {c.relaxed} "
                f"heap {c.heap_push}/{c.heap_pop}", DIM)
        add("")
        add("Reachable checkpoints: " + (
            f"{len(self.reach)}/{len(self.circuit.checkpoints)}"), TEXT)
        if self.unreach:
            add("Unreachable: " + ",".join(map(str, self.unreach[:12])), BLOCK)

        for text, color in lines:
            screen.blit(self.small.render(fit(self.small, text, pw), True, color), (PANEL_X, y))
            y += 17

        box = pygame.Rect(PANEL_X, H - 100, pw, 88)
        pygame.draw.rect(screen, (45, 49, 60), box, border_radius=4)
        pygame.draw.rect(screen, (90, 96, 112), box, 1, border_radius=4)
        screen.blit(self.small.render("Resource recommendations", True, TEXT),
                    (box.x + 8, box.y + 8))
        screen.blit(self.small.render("(DP / greedy panel: Person B)", True, DIM),
                    (box.x + 8, box.y + 26))


def run(files=None):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Smart Racing Championship - Route Planner")
    app = Explorer(files)
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            running = app.handle(ev) and running
        app.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    run(sys.argv[1:] or None)