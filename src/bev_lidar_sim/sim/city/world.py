"""Static city geometry shared by the street renderer and the LiDAR.

Buildings, trees, poles, parked cars, curbs, crossings, and lane paint are
all ordinary 2D geometry, so the richer environment remains inexpensive to
run. Physical obstacles (everything in `static_boxes` plus curbs) are sensed
by the LiDAR; paint (dashes, crosswalks, stop lines) is renderer-only.

The block/furniture layout is seeded and currently assumes the grid layout
from `city.graph`; generalizing it to imported road networks (deriving road
rectangles and infill lots from RoadGraph lanes) is the planned next step for
map import support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...sensors.scene import Box
from ..traffic import REFL
from .graph import (CITY_MAX, CITY_MIN, GRID_COORDS, LINE_OFF, ROAD_HALF,
                    SIDEWALK, _unit)


@dataclass(frozen=True)
class CityIntersection:
    node: str                 # RoadGraph node id
    x: float
    y: float
    kind: str                 # "signal" | "stop"


class CityWorld:
    """Static render and LiDAR geometry for a compact downtown grid."""

    is_city = True

    def __init__(self, seed: int, intersections: list[CityIntersection]):
        self.rng = np.random.default_rng(seed + 77)
        self.intersections = intersections
        self.bounds = (CITY_MIN, CITY_MIN, CITY_MAX, CITY_MAX)

        self.road_rects = []
        self.sidewalk_rects = []
        self.lane_dashes = []
        self.crosswalk_bars = []
        self.stop_lines = []
        self.curbs = []
        self.buildings = []
        self.trees = []
        self.poles = []
        self.parked = []

        self._build_roads()
        self._build_blocks()
        self._build_street_furniture()

    def _build_roads(self) -> None:
        span = CITY_MAX - CITY_MIN
        border = ROAD_HALF + SIDEWALK
        for y in GRID_COORDS:
            self.sidewalk_rects.append((CITY_MIN, y - border, span, 2 * border))
            self.road_rects.append((CITY_MIN, y - ROAD_HALF, span,
                                    2 * ROAD_HALF))
        for x in GRID_COORDS:
            self.sidewalk_rects.append((x - border, CITY_MIN, 2 * border, span))
            self.road_rects.append((x - ROAD_HALF, CITY_MIN,
                                    2 * ROAD_HALF, span))

        # Curbs and dashed center lines stop at every intersection opening.
        for y in GRID_COORDS:
            for lo, hi in self._open_intervals(GRID_COORDS):
                for sy in (-1.0, 1.0):
                    self.curbs.append(((lo, y + sy * ROAD_HALF),
                                       (hi, y + sy * ROAD_HALF), REFL["curb"]))
                self._dashes((lo, y), (hi, y))
        for x in GRID_COORDS:
            for lo, hi in self._open_intervals(GRID_COORDS):
                for sx in (-1.0, 1.0):
                    self.curbs.append(((x + sx * ROAD_HALF, lo),
                                       (x + sx * ROAD_HALF, hi), REFL["curb"]))
                self._dashes((x, lo), (x, hi))

        for item in self.intersections:
            self._intersection_paint(item.x, item.y)

    @staticmethod
    def _open_intervals(crossings):
        intervals = []
        start = CITY_MIN
        for p in crossings:
            intervals.append((start, p - ROAD_HALF))
            start = p + ROAD_HALF
        intervals.append((start, CITY_MAX))
        return [(a, b) for a, b in intervals if b - a > 1.0]

    def _dashes(self, a, b) -> None:
        d = _unit(a, b)
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        for s in np.arange(1.5, max(length - 1.5, 1.5), 8.0):
            e = min(s + 3.2, length - 1.0)
            if e > s:
                self.lane_dashes.append(
                    ((a[0] + d[0] * s, a[1] + d[1] * s),
                     (a[0] + d[0] * e, a[1] + d[1] * e)))

    def _intersection_paint(self, x, y) -> None:
        # Stop lines cover the approaching half of each two-way road.
        self.stop_lines.extend([
            ((x - LINE_OFF, y - ROAD_HALF), (x - LINE_OFF, y)),
            ((x + LINE_OFF, y), (x + LINE_OFF, y + ROAD_HALF)),
            ((x, y - LINE_OFF), (x + ROAD_HALF, y - LINE_OFF)),
            ((x - ROAD_HALF, y + LINE_OFF), (x, y + LINE_OFF)),
        ])

        # Zebra crossings: short white bars just outside the conflict box.
        for sx in (-1.0, 1.0):
            bx = x + sx * (ROAD_HALF + 0.9) - 0.8
            for yy in np.arange(y - ROAD_HALF + 0.5,
                                y + ROAD_HALF - 0.2, 1.35):
                self.crosswalk_bars.append((bx, yy, 1.6, 0.58))
        for sy in (-1.0, 1.0):
            by = y + sy * (ROAD_HALF + 0.9) - 0.8
            for xx in np.arange(x - ROAD_HALF + 0.5,
                                x + ROAD_HALF - 0.2, 1.35):
                self.crosswalk_bars.append((xx, by, 0.58, 1.6))

    def _build_blocks(self) -> None:
        # Four varied buildings per block create alleys and corner occlusions.
        for ix in range(len(GRID_COORDS) - 1):
            for iy in range(len(GRID_COORDS) - 1):
                x0 = GRID_COORDS[ix] + ROAD_HALF + SIDEWALK + 1.0
                x1 = GRID_COORDS[ix + 1] - ROAD_HALF - SIDEWALK - 1.0
                y0 = GRID_COORDS[iy] + ROAD_HALF + SIDEWALK + 1.0
                y1 = GRID_COORDS[iy + 1] - ROAD_HALF - SIDEWALK - 1.0
                xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                gap = 2.0
                lots = [(x0, xm - gap, y0, ym - gap),
                        (xm + gap, x1, y0, ym - gap),
                        (x0, xm - gap, ym + gap, y1),
                        (xm + gap, x1, ym + gap, y1)]
                for lx0, lx1, ly0, ly1 in lots:
                    if self.rng.random() < 0.10:
                        continue
                    inset = float(self.rng.uniform(0.8, 2.7))
                    w = max(6.0, lx1 - lx0 - 2 * inset)
                    h = max(6.0, ly1 - ly0 - 2 * inset)
                    cx = (lx0 + lx1) / 2.0 + float(self.rng.uniform(-0.7, 0.7))
                    cy = (ly0 + ly1) / 2.0 + float(self.rng.uniform(-0.7, 0.7))
                    self.buildings.append(
                        Box(cx, cy, 0.0, w, h, "building", REFL["building"]))

        # Trees and parked vehicles make otherwise similar blocks distinct.
        for a, b in zip(GRID_COORDS[:-1], GRID_COORDS[1:]):
            for street in GRID_COORDS:
                for side in (-1.0, 1.0):
                    if self.rng.random() < 0.72:
                        x = (a + b) / 2.0 + float(self.rng.uniform(-14.0, 14.0))
                        y = street + side * (ROAD_HALF + 1.55)
                        self.trees.append(
                            Box(x, y, 0.0, 0.75, 0.75, "tree", REFL["tree"]))
                    if self.rng.random() < 0.58:
                        x = (a + b) / 2.0 + float(self.rng.uniform(-12.0, 12.0))
                        y = street + side * (ROAD_HALF - 1.15)
                        self.parked.append(
                            Box(x, y, 0.0, 4.5, 1.85, "car", REFL["car"]))

                    if self.rng.random() < 0.72:
                        x = street + side * (ROAD_HALF + 1.55)
                        y = (a + b) / 2.0 + float(self.rng.uniform(-14.0, 14.0))
                        self.trees.append(
                            Box(x, y, 0.0, 0.75, 0.75, "tree", REFL["tree"]))
                    if self.rng.random() < 0.58:
                        x = street + side * (ROAD_HALF - 1.15)
                        y = (a + b) / 2.0 + float(self.rng.uniform(-12.0, 12.0))
                        self.parked.append(
                            Box(x, y, math.pi / 2.0, 4.5, 1.85,
                                "car", REFL["car"]))

    def _build_street_furniture(self) -> None:
        for item in self.intersections:
            label = "pole" if item.kind == "signal" else "sign"
            refl = REFL[label]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                self.poles.append(
                    Box(item.x + sx * (ROAD_HALF + 0.7),
                        item.y + sy * (ROAD_HALF + 0.7), 0.0,
                        0.45, 0.45, label, refl))

    def static_boxes(self):
        return self.buildings + self.trees + self.poles + self.parked
