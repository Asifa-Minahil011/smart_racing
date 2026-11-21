import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest
from src.ui_explorer import Explorer, W, H, ALGOS
from src.cost import PRESETS


@pytest.fixture
def app():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    a = Explorer(["circuits/circuit1.json"])
    a.screen = screen
    yield a
    pygame.quit()


def click(app, pos, button=1):
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))


def click_vertex(app, vid, button=1):
    click(app, app.pos[vid], button)


def test_draws_in_every_state(app):
    click_vertex(app, 9)
    for algo in ALGOS:
        for preset in PRESETS:
            app.algo, app.preset = algo, preset
            app.compute()
            app.draw(app.screen)
    app.lap_mode = True
    app.compute()
    app.draw(app.screen)
    assert app.plan is not None


def test_click_sets_destination_and_route(app):
    click_vertex(app, 4)
    assert app.dest == 4 and app.route is not None
    assert app.route.vertices[0] == app.start and app.route.vertices[-1] == 4


def test_ui_cost_equals_optimized_cost(app):
    from src.routing import dijkstra
    click_vertex(app, 8)
    again = dijkstra(app.circuit.graph, app.model, app.start, 8)
    assert app.route.cost == pytest.approx(again.cost)


def test_blocking_a_road_recalculates(app):
    click_vertex(app, 4)
    first = list(app.route.edges)
    app.act(("block",))
    mid = app.route.edges[0]
    e = app.circuit.graph.edges[mid]
    a, b = app.pos[e.from_v], app.pos[e.to_v]
    click(app, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
    assert app.circuit.graph.edges[mid].status == "Blocked"
    assert mid not in app.route.edges
    app.act(("reset",))
    assert app.circuit.graph.edges[mid].status == "Open"


def test_unreachable_destination_does_not_crash(app):
    g = app.circuit.graph
    for e in list(g.edges.values()):
        if 20 in (e.from_v, e.to_v):
            g.block_edge(e.id)
    click_vertex(app, 20)
    assert app.route is None
    app.draw(app.screen)


def test_dfs_mode_reports_cycle_and_dead_end(app):
    app.algo = "DFS"
    click_vertex(app, 5)
    assert app.cycle is not None
    assert app.deadends == [20]


def test_buttons_and_keys_work(app):
    for rect, label, a, on in app.buttons():
        if label == "Safety":
            click(app, rect.center)
    assert app.preset == "Safety"
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4))
    assert app.algo == "A*"


def test_right_click_resets_selection(app):
    click_vertex(app, 6)
    click(app, (500, 400), button=3)
    assert app.dest is None


def test_bfs_shows_optimal_for_comparison(app):
    app.algo = "BFS"
    click_vertex(app, 9)
    assert app.optimal is not None
    assert app.optimal.cost <= app.route.cost + 1e-9