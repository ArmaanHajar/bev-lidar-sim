"""Named driving scenarios: distinct maps with fixed A-to-B ego missions.

Each scenario is a map builder plus mission parameters. The default `city`
keeps the classic grid with random ego trips; the others give the ego a
fixed origin -> destination mission that restarts on arrival:

  * `suburbs`    — an irregular 4x3 district with missing links, so the grid
                   degenerates into T-junctions and forced detours.
  * `riverside`  — a winding east-west drive along a river, with stop-sign
                   T-junctions and signalized 4-ways strung along the curve.
  * `roadworks`  — the classic grid with one block closed for construction
                   (paved, coned off, barricaded); traffic and the ego must
                   route around it.

Builders are deterministic (seeds only affect world dressing and traffic).
`make_sim(name, seed, driving_mode)` is the one-stop constructor the CLI
uses; `SCENARIOS` lists what is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..maps.roadgraph import RoadGraph, RoadNode
from .city.graph import (EDGE_REACH, add_connectors, add_two_way_road,
                         build_grid_roadgraph)
from .city.simulator import CitySimulator

# --- riverside geometry ------------------------------------------------------
RIVER_AMP = 18.0
RIVER_WAVE = 55.0
RIVER_X = (-105.0, -35.0, 35.0, 105.0)   # junctions along the curve
RIVER_END = 160.0


def _river_y(x) -> float:
    return float(RIVER_AMP * np.sin(np.asarray(x, dtype=float) / RIVER_WAVE))


def _river_pts(x0: float, x1: float, step: float = 6.0) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    xs = np.append(xs, x1)
    return np.stack([xs, RIVER_AMP * np.sin(xs / RIVER_WAVE)], axis=1)


def build_riverside_roadgraph() -> RoadGraph:
    """A winding main road with side streets; the river runs to the south."""
    graph = RoadGraph()
    graph.nodes["RW"] = RoadNode("RW", -RIVER_END, _river_y(-RIVER_END),
                                 "boundary")
    graph.nodes["RE"] = RoadNode("RE", RIVER_END, _river_y(RIVER_END),
                                 "boundary")
    # Alternate stop-controlled T-junctions and signalized 4-ways.
    side = {0: ("N",), 1: ("N", "S"), 2: ("N",), 3: ("N", "S")}
    for i, x in enumerate(RIVER_X):
        kind = "signal" if len(side[i]) == 2 else "stop"
        offset = float((i * 7) % 13) if kind == "signal" else 0.0
        graph.nodes[f"M{i}"] = RoadNode(f"M{i}", x, _river_y(x), kind, offset)
        for s in side[i]:
            sy = _river_y(x) + (55.0 if s == "N" else -55.0)
            nid = f"B{i}{s}"
            graph.nodes[nid] = RoadNode(nid, x, sy, "boundary")

    main_x = (-RIVER_END,) + RIVER_X + (RIVER_END,)
    main_ids = ("RW",) + tuple(f"M{i}" for i in range(len(RIVER_X))) + ("RE",)
    for (xa, a), (xb, b) in zip(zip(main_x[:-1], main_ids[:-1]),
                                zip(main_x[1:], main_ids[1:])):
        add_two_way_road(graph, a, b, center=_river_pts(xa, xb), speed=12.5)
    for i in range(len(RIVER_X)):
        for s in side[i]:
            add_two_way_road(graph, f"M{i}", f"B{i}{s}", speed=10.0)

    add_connectors(graph)
    graph.validate()
    return graph


def _river_water() -> list:
    """A water band hugging the south bank, clear of the side streets."""
    xs = np.arange(-15.0, 86.0, 6.0)
    ys = RIVER_AMP * np.sin(xs / RIVER_WAVE)
    near = np.stack([xs, ys - 26.0], axis=1)
    far = np.stack([xs, ys - 70.0], axis=1)
    return [np.vstack([near, far[::-1]])]


# --- registry ----------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    build: callable
    district_name: str
    ego_od: tuple | None = None
    seed_vehicles: int = 19
    max_vehicles: int = 29
    world_params: dict = field(default_factory=dict)


SUBURB_SKIPS = frozenset({((1, 1), (2, 1)), ((2, 0), (2, 1))})
WORKS_EDGE = ((1, 2), (2, 2))
WORKS_CENTER = np.array([[-35.0, 35.0], [35.0, 35.0]])

SCENARIOS = {
    "city": Scenario(
        "city", "the classic 4x4 grid district; random ego destinations",
        build_grid_roadgraph, "GRID DISTRICT"),
    "suburbs": Scenario(
        "suburbs", "irregular blocks, T-junctions, and a cross-town mission",
        lambda: build_grid_roadgraph(
            xs=(-110.0, -40.0, 15.0, 95.0), ys=(-85.0, -15.0, 70.0),
            skip=SUBURB_SKIPS, with_signals=False),
        "SUBURB LOOPS", ego_od=("W-1", "E-1"),
        seed_vehicles=13, max_vehicles=20,
        world_params=dict(building_attempts=170, tree_step=13.0)),
    "riverside": Scenario(
        "riverside", "a winding drive along the river, end to end",
        build_riverside_roadgraph, "RIVERSIDE DRIVE", ego_od=("RW", "RE"),
        seed_vehicles=11, max_vehicles=17,
        world_params=dict(building_attempts=150, tree_step=11.0,
                          water=_river_water())),
    "roadworks": Scenario(
        "roadworks", "one block closed for construction; detour around it",
        lambda: build_grid_roadgraph(skip=frozenset({WORKS_EDGE})),
        "GRID DISTRICT — ROADWORKS", ego_od=("W-2", "E-2"),
        seed_vehicles=17, max_vehicles=26,
        world_params=dict(closed_roads=[WORKS_CENTER])),
}


def make_sim(name: str, seed: int = 1,
             driving_mode: str = "normal") -> CitySimulator:
    """Build the named scenario's simulator (the CLI entry point)."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; "
                         f"choose from {', '.join(SCENARIOS)}")
    sc = SCENARIOS[name]
    return CitySimulator(seed=seed, graph=sc.build(),
                         driving_mode=driving_mode, ego_od=sc.ego_od,
                         district_name=sc.district_name,
                         seed_vehicles=sc.seed_vehicles,
                         max_vehicles=sc.max_vehicles,
                         world_params=dict(sc.world_params))
