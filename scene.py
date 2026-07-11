"""2D bird's-eye-view scene primitives.

A scene is a collection of oriented rectangles (vehicles) plus loose wall
segments (curbs / buildings). Everything the LiDAR can hit is ultimately a
set of line segments, but we keep the rectangle metadata around so we can
draw free ground-truth boxes later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    """An oriented rectangle in the world plane (a vehicle footprint)."""

    cx: float
    cy: float
    heading: float  # radians, 0 = facing +x
    length: float   # extent along heading
    width: float    # extent perpendicular to heading
    label: str = "car"

    def corners(self) -> np.ndarray:
        """Return the 4 corners as a (4, 2) array, counter-clockwise."""
        hl, hw = self.length / 2.0, self.width / 2.0
        local = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]])
        c, s = np.cos(self.heading), np.sin(self.heading)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.cx, self.cy])

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the 4 edges as (A, B) arrays, each (4, 2)."""
        c = self.corners()
        a = c
        b = np.roll(c, -1, axis=0)
        return a, b


@dataclass
class Scene:
    """A collection of boxes plus loose wall segments."""

    boxes: list[Box] = field(default_factory=list)
    wall_a: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    wall_b: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def add_wall(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        self.wall_a = np.vstack([self.wall_a, a])
        self.wall_b = np.vstack([self.wall_b, b])

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """All hittable segments in the scene as (A, B), each (M, 2)."""
        a_list, b_list = [self.wall_a], [self.wall_b]
        for box in self.boxes:
            a, b = box.segments()
            a_list.append(a)
            b_list.append(b)
        return np.vstack(a_list), np.vstack(b_list)


def build_demo_scene() -> Scene:
    """A straight road with curbs and a few parked / oncoming vehicles.

    Ego is assumed at the origin facing +x (down the road).
    """
    scene = Scene()

    # Road curbs: two long walls parallel to the x-axis.
    road_half_width = 6.0
    x0, x1 = -20.0, 40.0
    scene.add_wall((x0, +road_half_width), (x1, +road_half_width))
    scene.add_wall((x0, -road_half_width), (x1, -road_half_width))

    # A building edge set back on the right, past the curb.
    scene.add_wall((10.0, -road_half_width - 2.0), (28.0, -road_half_width - 2.0))

    # Vehicles: (cx, cy, heading, length, width)
    scene.boxes.extend([
        Box(cx=12.0, cy=2.5, heading=0.0, length=4.5, width=2.0, label="car"),
        Box(cx=22.0, cy=-2.5, heading=np.pi, length=4.5, width=2.0, label="car"),
        Box(cx=30.0, cy=2.0, heading=0.05, length=8.0, width=2.4, label="truck"),
        Box(cx=6.0, cy=-3.0, heading=np.pi, length=4.2, width=1.9, label="car"),
    ])
    return scene
