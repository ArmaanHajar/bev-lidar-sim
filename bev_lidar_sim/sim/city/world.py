"""Static city geometry derived from a RoadGraph, for renderer and LiDAR.

Everything is generated from the map itself, so curved roads, T-junctions,
and irregular layouts decorate correctly with no grid assumptions:

  * pavement/sidewalk strips buffer each road centerline (`RoadDef`),
  * curbs offset the centerline and break at junction openings,
  * stop lines, crosswalks, and poles come from lane end tangents at
    controlled nodes,
  * buildings are rejection-sampled against road clearance,
  * trees and parked cars sample positions along road tangents.

Physical obstacles (`static_boxes` plus curbs) are sensed by the LiDAR;
paint (dashes, crosswalks, stop lines) and water are renderer-only.
Scenario dressing — closed roads with barriers, water bands — comes in via
constructor arguments so the map schema stays pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...sensors.scene import Box
from ..traffic import REFL
from .graph import (LANE_OFF, ROAD_HALF, SIDEWALK, _right, _unit,
                    offset_polyline, trim_polyline)

BARRIER_REFL = 0.90            # striped construction barriers read hot


@dataclass(frozen=True)
class CityIntersection:
    node: str                 # RoadGraph node id
    x: float
    y: float
    kind: str                 # "signal" | "stop"


def _walk(pts, spacing, lo=0.0, hi=None):
    """Yield (point, tangent) every `spacing` meters of arc length."""
    pts = np.asarray(pts, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    hi = total if hi is None else min(hi, total)
    u = lo
    while u <= hi:
        i = min(int(np.searchsorted(cum, u, side="right")) - 1,
                len(seg_len) - 1)
        i = max(i, 0)
        f = (u - cum[i]) / max(seg_len[i], 1e-9)
        p = pts[i] + f * (pts[i + 1] - pts[i])
        d = seg[i] / max(seg_len[i], 1e-9)
        yield (float(p[0]), float(p[1])), (float(d[0]), float(d[1]))
        u += spacing


def _polyline_length(pts) -> float:
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _quad(cx, cy, d, along, across):
    """4 corners of a rectangle centered at (cx,cy), long axis along `d`."""
    r = _right(d)
    ax, ay = d[0] * along / 2.0, d[1] * along / 2.0
    bx, by = r[0] * across / 2.0, r[1] * across / 2.0
    return np.array([[cx + ax + bx, cy + ay + by],
                     [cx + ax - bx, cy + ay - by],
                     [cx - ax - bx, cy - ay - by],
                     [cx - ax + bx, cy - ay + by]])


class CityWorld:
    """Static render and LiDAR geometry for any RoadGraph-shaped district."""

    is_city = True

    def __init__(self, seed: int, graph, intersections,
                 building_attempts: int = 260, tree_step: float = 17.0,
                 parked_step: float = 21.0, closed_roads=(), water=()):
        self.rng = np.random.default_rng(seed + 77)
        self.graph = graph
        self.intersections = intersections
        xs = [n.x for n in graph.nodes.values()]
        ys = [n.y for n in graph.nodes.values()]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))

        self.road_polys = []       # (N,2) asphalt polygons
        self.sidewalk_polys = []   # (N,2) sidewalk polygons
        self.lane_dashes = []      # [((x,y),(x,y)), ...] center paint
        self.crosswalk_quads = []  # (4,2) oriented white bars
        self.stop_lines = []       # [((x,y),(x,y)), ...]
        self.curbs = []            # [((x,y),(x,y), refl), ...]
        self.buildings = []
        self.trees = []
        self.poles = []
        self.parked = []
        self.barriers = []         # construction barriers (sensed + drawn)
        self.water_polys = [np.asarray(w, dtype=float) for w in water]

        self._controlled = {nid: n for nid, n in graph.nodes.items()
                            if n.kind != "boundary"}
        self._road_samples = []    # dense points on all pavement, for setback
        self._build_roads(closed_roads)
        self._build_junction_paint()
        self._build_buildings(building_attempts)
        self._build_roadside(tree_step, parked_step)
        self._build_street_furniture()
        for center in closed_roads:
            self._close_road(center)

    # --- pavement, curbs, dashes -----------------------------------------
    def _trims(self, pts) -> tuple:
        """Arc-length curb/dash setback at each end of a road centerline."""
        def setback(p):
            for node in self._controlled.values():
                if math.hypot(p[0] - node.x, p[1] - node.y) < 1.0:
                    return ROAD_HALF
            return 0.0
        pts = np.asarray(pts, dtype=float)
        return setback(pts[0]), setback(pts[-1])

    def _pave(self, center) -> None:
        left = offset_polyline(center, -ROAD_HALF - SIDEWALK)
        right = offset_polyline(center, ROAD_HALF + SIDEWALK)
        self.sidewalk_polys.append(np.vstack([left, right[::-1]]))
        left = offset_polyline(center, -ROAD_HALF)
        right = offset_polyline(center, ROAD_HALF)
        self.road_polys.append(np.vstack([left, right[::-1]]))
        for p, _ in _walk(center, 3.0):
            self._road_samples.append(p)

    def _build_roads(self, closed_roads) -> None:
        for road in self.graph.roads:
            center = np.asarray(road.pts, dtype=float)
            self._pave(center)

            t0, t1 = self._trims(center)
            try:
                open_center = trim_polyline(center, t0, t1)
            except ValueError:
                continue                    # stub shorter than the junction
            for side in (-1.0, 1.0):
                curb = offset_polyline(open_center, side * ROAD_HALF)
                for a, b in zip(curb[:-1], curb[1:]):
                    self.curbs.append((tuple(a), tuple(b), REFL["curb"]))
            self._dashes(open_center)

        # Square pavement pads close the junction boxes (roads stop at the
        # node center, so crossing strips already overlap; pads cover skews).
        for node in self._controlled.values():
            for r, dest in ((ROAD_HALF + SIDEWALK, self.sidewalk_polys),
                            (ROAD_HALF, self.road_polys)):
                dest.append(np.array([[node.x - r, node.y - r],
                                      [node.x + r, node.y - r],
                                      [node.x + r, node.y + r],
                                      [node.x - r, node.y + r]]))

    def _dashes(self, center) -> None:
        length = _polyline_length(center)
        marks = list(_walk(center, 8.0, lo=1.5, hi=max(length - 2.5, 1.5)))
        for (p, d) in marks:
            self.lane_dashes.append(
                ((p[0], p[1]), (p[0] + d[0] * 3.2, p[1] + d[1] * 3.2)))

    # --- controlled-approach paint ----------------------------------------
    def _build_junction_paint(self) -> None:
        for lane in self.graph.lanes.values():
            node = self.graph.nodes[lane.end]
            if node.kind == "boundary":
                continue
            end = lane.pts[-1]
            d = _unit(lane.pts[-2], lane.pts[-1])
            r = _right(d)
            # Stop line spans the approach half of the road: from the road
            # centerline out to the right curb.
            a = (end[0] - r[0] * LANE_OFF, end[1] - r[1] * LANE_OFF)
            b = (end[0] + r[0] * (ROAD_HALF - LANE_OFF),
                 end[1] + r[1] * (ROAD_HALF - LANE_OFF))
            self.stop_lines.append((a, b))

            # Zebra crossing just outside the conflict box, full road width.
            cx = node.x - d[0] * (ROAD_HALF + 0.9)
            cy = node.y - d[1] * (ROAD_HALF + 0.9)
            for off in np.arange(-ROAD_HALF + 0.8, ROAD_HALF - 0.4, 1.35):
                self.crosswalk_quads.append(
                    _quad(cx + r[0] * off, cy + r[1] * off, d, 1.6, 0.58))

    # --- buildings ---------------------------------------------------------
    def _build_buildings(self, attempts: int) -> None:
        if not self._road_samples:
            return
        samples = np.asarray(self._road_samples)
        x0, y0, x1, y1 = self.bounds
        setback = ROAD_HALF + SIDEWALK + 1.0
        placed = []
        for _ in range(attempts):
            w = float(self.rng.uniform(9.0, 24.0))
            h = float(self.rng.uniform(9.0, 24.0))
            cx = float(self.rng.uniform(x0 - 20.0, x1 + 20.0))
            cy = float(self.rng.uniform(y0 - 20.0, y1 + 20.0))
            half_diag = math.hypot(w, h) / 2.0
            d2 = (samples[:, 0] - cx) ** 2 + (samples[:, 1] - cy) ** 2
            if d2.min() < (setback + half_diag * 0.78) ** 2:
                continue                    # too close to a road
            if any(abs(cx - px) < (w + pw) / 2.0 + 2.0
                   and abs(cy - py) < (h + ph) / 2.0 + 2.0
                   for px, py, pw, ph in placed):
                continue                    # overlaps a placed building
            placed.append((cx, cy, w, h))
            self.buildings.append(
                Box(cx, cy, 0.0, w, h, "building", REFL["building"]))

    # --- roadside dressing ---------------------------------------------------
    def _near_junction(self, p, clearance: float) -> bool:
        return any(math.hypot(p[0] - n.x, p[1] - n.y) < clearance
                   for n in self._controlled.values())

    def _build_roadside(self, tree_step: float, parked_step: float) -> None:
        for road in self.graph.roads:
            center = np.asarray(road.pts, dtype=float)
            for p, d in _walk(center, tree_step, lo=8.0):
                if self._near_junction(p, 12.0) or self.rng.random() > 0.72:
                    continue
                side = 1.0 if self.rng.random() < 0.5 else -1.0
                r = _right(d)
                off = side * (ROAD_HALF + 1.55)
                jit = float(self.rng.uniform(-3.0, 3.0))
                self.trees.append(Box(p[0] + r[0] * off + d[0] * jit,
                                      p[1] + r[1] * off + d[1] * jit,
                                      0.0, 0.75, 0.75, "tree", REFL["tree"]))
            for p, d in _walk(center, parked_step, lo=11.0):
                if self._near_junction(p, 14.0) or self.rng.random() > 0.58:
                    continue
                side = 1.0 if self.rng.random() < 0.5 else -1.0
                r = _right(d)
                off = side * (ROAD_HALF - 1.15)
                jit = float(self.rng.uniform(-4.0, 4.0))
                heading = math.atan2(d[1], d[0])
                self.parked.append(Box(p[0] + r[0] * off + d[0] * jit,
                                       p[1] + r[1] * off + d[1] * jit,
                                       heading, 4.5, 1.85, "car",
                                       REFL["car"]))

    def _build_street_furniture(self) -> None:
        for item in self.intersections:
            label = "pole" if item.kind == "signal" else "sign"
            refl = REFL[label]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                self.poles.append(
                    Box(item.x + sx * (ROAD_HALF + 0.7),
                        item.y + sy * (ROAD_HALF + 0.7), 0.0,
                        0.45, 0.45, label, refl))

    # --- scenario dressing ----------------------------------------------------
    def _close_road(self, center) -> None:
        """Pave a lane-less road and barricade both ends (roadworks)."""
        center = np.asarray(center, dtype=float)
        self._pave(center)
        for side in (-1.0, 1.0):
            curb = offset_polyline(center, side * ROAD_HALF)
            for a, b in zip(curb[:-1], curb[1:]):
                self.curbs.append((tuple(a), tuple(b), REFL["curb"]))
        length = _polyline_length(center)
        for u in (ROAD_HALF + 1.5, length - ROAD_HALF - 1.5):
            for (p, d) in _walk(center, 1.0, lo=u, hi=u):
                heading = math.atan2(d[1], d[0]) + math.pi / 2.0
                self.barriers.append(
                    Box(p[0], p[1], heading, 2.0 * ROAD_HALF - 1.0, 0.6,
                        "barrier", BARRIER_REFL))
        # A few cones/works boxes along the closure sell the construction.
        for (p, d) in _walk(center, length / 3.0, lo=length / 4.0):
            self.barriers.append(
                Box(p[0], p[1], math.atan2(d[1], d[0]), 3.2, 1.6,
                    "barrier", BARRIER_REFL))

    def static_boxes(self):
        return (self.buildings + self.trees + self.poles + self.parked
                + self.barriers)
