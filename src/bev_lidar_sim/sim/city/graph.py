"""RoadGraph construction utilities and the default grid district.

Two things live here: generic map-building helpers (polyline offsetting for
curved two-way roads, arc-length trimming to stop lines, tangent-based
connector generation) and the concrete builders that use them. The default
city is `build_city_roadgraph()`, a 4x4 signal/stop checkerboard; scenario
maps in `sim/scenarios.py` reuse the same helpers for curved and irregular
layouts.

Conventions every builder must follow:
  * A lane at a controlled node ends LINE_OFF short of the node center — at
    its stop line. `add_two_way_road` applies this automatically.
  * Turn classification uses lane *tangents* at the junction, so curved
    approaches classify correctly; the connector's quadratic control point
    is the node center.
  * No U-turns, and no turn connectors at signals: the two-phase signal only
    protects opposing straight movements. Turns belong at all-way stops.

Builders are pure functions of their arguments (no RNG), so two processes
always produce identical graphs.
"""

from __future__ import annotations

import math

import numpy as np

from ...maps.roadgraph import ConnectorDef, LaneDef, RoadGraph, RoadNode

# --- City dimensions -------------------------------------------------------
GRID_N = 4
BLOCK = 70.0
GRID_COORDS = tuple((i - (GRID_N - 1) / 2.0) * BLOCK for i in range(GRID_N))
EDGE_REACH = 45.0
CITY_MIN = GRID_COORDS[0] - EDGE_REACH
CITY_MAX = GRID_COORDS[-1] + EDGE_REACH

ROAD_HALF = 5.5
SIDEWALK = 3.0
LANE_OFF = 1.9
LINE_OFF = 6.2
CITY_SPEED = 11.5             # 41 km/h
TURN_SPEED = 4.8


# --- geometry helpers -------------------------------------------------------
def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(math.hypot(dx, dy), 1e-9)
    return dx / n, dy / n


def _right(d):
    return d[1], -d[0]


def _quadratic(start, control, end, n=14):
    t = np.linspace(0.0, 1.0, n)[:, None]
    a = np.asarray(start, dtype=float)
    c = np.asarray(control, dtype=float)
    b = np.asarray(end, dtype=float)
    return (1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * c + t ** 2 * b


def offset_polyline(pts, d: float) -> np.ndarray:
    """Offset a polyline `d` meters to the right of its travel direction."""
    pts = np.asarray(pts, dtype=float)
    seg = np.diff(pts, axis=0)
    seg /= np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)[:, None]
    nrm = np.stack([seg[:, 1], -seg[:, 0]], axis=1)   # right normals
    vn = np.vstack([nrm[:1], nrm[:-1] + nrm[1:], nrm[-1:]])
    vn /= np.maximum(np.hypot(vn[:, 0], vn[:, 1]), 1e-9)[:, None]
    return pts + d * vn


def trim_polyline(pts, t0: float, t1: float) -> np.ndarray:
    """Cut `t0` meters off the start and `t1` off the end (arc length)."""
    pts = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    lo, hi = t0, cum[-1] - t1
    if hi - lo < 0.5:
        raise ValueError("polyline too short for requested trims")

    def point_at(u):
        i = min(int(np.searchsorted(cum, u, side="right")) - 1, len(seg) - 1)
        i = max(i, 0)
        f = (u - cum[i]) / max(seg[i], 1e-9)
        return pts[i] + f * (pts[i + 1] - pts[i])

    keep = pts[(cum > lo + 1e-6) & (cum < hi - 1e-6)]
    return np.vstack([point_at(lo), keep, point_at(hi)])


def _end_tangent(pts) -> tuple:
    return _unit(pts[-2], pts[-1])


def _start_tangent(pts) -> tuple:
    return _unit(pts[0], pts[1])


# --- generic builders -------------------------------------------------------
def add_two_way_road(graph: RoadGraph, a_id: str, b_id: str,
                     center=None, speed: float = CITY_SPEED) -> None:
    """Add both directed lanes of a road between two existing nodes.

    `center` is the road centerline polyline from node a to node b (defaults
    to a straight line). Lanes are trimmed to the stop line at controlled
    ends and offset LANE_OFF to the right of travel.
    """
    na, nb = graph.nodes[a_id], graph.nodes[b_id]
    if center is None:
        center = np.array([[na.x, na.y], [nb.x, nb.y]])
    center = np.asarray(center, dtype=float)
    for start, end, pts in ((a_id, b_id, center), (b_id, a_id, center[::-1])):
        ns, ne = graph.nodes[start], graph.nodes[end]
        trimmed = trim_polyline(pts,
                                LINE_OFF if ns.kind != "boundary" else 0.0,
                                LINE_OFF if ne.kind != "boundary" else 0.0)
        lane_pts = offset_polyline(trimmed, LANE_OFF)
        approach = None
        if ne.kind != "boundary":
            d = _unit((ns.x, ns.y), (ne.x, ne.y))
            approach = "main" if abs(d[0]) > abs(d[1]) else "cross"
        graph.lanes[f"{start}>{end}"] = LaneDef(
            f"{start}>{end}", lane_pts, speed, start, end, approach)


def add_connectors(graph: RoadGraph, straight_speed: float = CITY_SPEED,
                   turn_speed: float = TURN_SPEED) -> None:
    """Emit lane-to-lane movements at every controlled node.

    Turn classification uses lane end/start tangents, so it works for curved
    roads. Straight movements only at signals; no U-turns anywhere.
    """
    outgoing: dict[str, list[LaneDef]] = {}
    for lane in graph.lanes.values():
        outgoing.setdefault(lane.start, []).append(lane)

    for lane in graph.lanes.values():
        node = graph.nodes[lane.end]
        if node.kind == "boundary":
            continue
        center = (node.x, node.y)
        din = _end_tangent(lane.pts)
        for out_lane in outgoing.get(lane.end, []):
            if out_lane.end == lane.start:
                continue                    # no U-turns
            dout = _start_tangent(out_lane.pts)
            dot = din[0] * dout[0] + din[1] * dout[1]
            cross = din[0] * dout[1] - din[1] * dout[0]
            if dot > 0.5:
                turn = "straight"
            elif cross < -0.5:
                turn = "right"
            else:
                turn = "left"
            # At signals, opposing straight traffic may run together.
            # Turns remain at serialized all-way stops until the city has
            # movement-specific protected turn phases/reservations.
            if turn != "straight" and node.kind == "signal":
                continue
            start = lane.pts[-1]
            end = out_lane.pts[0]
            if turn == "straight":
                pts = np.stack([start, end])
                speed = straight_speed
            else:
                pts = _quadratic(start, center, end)
                speed = turn_speed
            graph.connectors.append(ConnectorDef(
                f"{turn}:{lane.start}>{lane.end}>{out_lane.end}",
                pts, speed, lane.id, out_lane.id, turn))


# --- grid district ----------------------------------------------------------
def _node_id(node) -> str:
    return f"{node[0]}-{node[1]}"


def build_grid_roadgraph(xs=GRID_COORDS, ys=GRID_COORDS,
                         skip: frozenset = frozenset()) -> RoadGraph:
    """A signal/stop checkerboard grid with boundary stubs on every road.

    `skip` holds internal edges to omit, as normalized node-tuple pairs
    (e.g. `((1, 2), (2, 2))`). Any intersection missing an edge becomes an
    all-way stop — a two-phase signal cannot serve an approach that must
    turn, so T-junctions have to be stop-controlled.
    """
    edge_x = xs[0] - EDGE_REACH, xs[-1] + EDGE_REACH
    edge_y = ys[0] - EDGE_REACH, ys[-1] + EDGE_REACH
    skip = {tuple(sorted(e, key=str)) for e in skip}

    def has(a, b):
        return tuple(sorted((a, b), key=str)) not in skip

    graph = RoadGraph()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            neighbors = [(("W", j) if i == 0 else (i - 1, j)),
                         (("E", j) if i == len(xs) - 1 else (i + 1, j)),
                         (("S", i) if j == 0 else (i, j - 1)),
                         (("N", i) if j == len(ys) - 1 else (i, j + 1))]
            degree = sum(has((i, j), n) for n in neighbors)
            kind = "signal" if (i + j) % 2 == 0 and degree == 4 else "stop"
            offset = float((i * 5 + j * 8) % 17) if kind == "signal" else 0.0
            nid = _node_id((i, j))
            graph.nodes[nid] = RoadNode(nid, x, y, kind, offset)
    for j, y in enumerate(ys):
        for side, x in (("W", edge_x[0]), ("E", edge_x[1])):
            nid = _node_id((side, j))
            graph.nodes[nid] = RoadNode(nid, x, y, "boundary")
    for i, x in enumerate(xs):
        for side, y in (("S", edge_y[0]), ("N", edge_y[1])):
            nid = _node_id((side, i))
            graph.nodes[nid] = RoadNode(nid, x, y, "boundary")

    sequences = []
    for j in range(len(ys)):
        sequences.append([("W", j)] + [(i, j) for i in range(len(xs))]
                         + [("E", j)])
    for i in range(len(xs)):
        sequences.append([("S", i)] + [(i, j) for j in range(len(ys))]
                         + [("N", i)])
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if has(a, b):
                add_two_way_road(graph, _node_id(a), _node_id(b))

    add_connectors(graph)
    graph.validate()
    return graph


def build_city_roadgraph() -> RoadGraph:
    """The default 4x4 grid district as a validated RoadGraph."""
    return build_grid_roadgraph()
