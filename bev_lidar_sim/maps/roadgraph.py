"""Neutral road-network schema shared by map sources and the simulator.

A `RoadGraph` describes a drivable map with three record types:

  * `RoadNode`   — an intersection (signalized or all-way stop) or a map-edge
                   endpoint where traffic enters and leaves ("boundary").
  * `LaneDef`    — a directed lane centerline polyline running from one node
                   to another. A lane that ends at a controlled node ends *at
                   its stop line*; the control applies at the lane's end.
  * `ConnectorDef` — a lane-to-lane movement (straight/left/right) through
                   the intersection at the source lane's end node. Its
                   polyline starts exactly at the source lane's last point
                   and ends exactly at the destination lane's first point.

Map *sources* (the procedural city in `city/graph.py`, future importers for
SUMO netconvert or Waymo Open Motion Dataset maps) emit a `RoadGraph`; the
traffic simulator consumes one without knowing where it came from. The graph
is plain data — pure numpy plus stdlib json — with no simulator, renderer, or
sensor knowledge, and it round-trips through JSON so scenarios can be saved
and shared as files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NODE_KINDS = ("signal", "stop", "boundary")
TURNS = ("straight", "left", "right")


@dataclass(frozen=True)
class RoadNode:
    """An intersection or a boundary endpoint of the map."""

    id: str
    x: float
    y: float
    kind: str                   # "signal" | "stop" | "boundary"
    signal_offset: float = 0.0  # phase offset (s) when kind == "signal"


@dataclass
class LaneDef:
    """A directed lane centerline from node `start` to node `end`."""

    id: str
    pts: np.ndarray             # (N, 2) polyline, N >= 2
    speed: float                # speed limit (m/s)
    start: str
    end: str
    approach: str | None = None  # signal phase group at the end node


@dataclass
class ConnectorDef:
    """A movement joining lane `src` to lane `dst` through an intersection."""

    id: str
    pts: np.ndarray             # (N, 2); pts[0] == src end, pts[-1] == dst start
    speed: float
    src: str
    dst: str
    turn: str                   # "straight" | "left" | "right"


@dataclass
class RoadDef:
    """The physical roadway carrying a pair of lanes: centerline + width.

    Lanes define where vehicles drive; roads define where pavement is. World
    decoration (asphalt, curbs, sidewalks) and future offroad metrics both
    need this, and an importer that only has lane centerlines can omit it —
    consumers must treat `roads` as optional.
    """

    id: str
    pts: np.ndarray             # (N, 2) road centerline polyline
    half_width: float


@dataclass
class RoadGraph:
    nodes: dict[str, RoadNode] = field(default_factory=dict)
    lanes: dict[str, LaneDef] = field(default_factory=dict)
    connectors: list[ConnectorDef] = field(default_factory=list)
    roads: list[RoadDef] = field(default_factory=list)

    # --- queries ---------------------------------------------------------
    def boundary_in(self) -> list[LaneDef]:
        """Lanes that begin at the map edge (traffic sources)."""
        return [l for l in self.lanes.values()
                if self.nodes[l.start].kind == "boundary"]

    def boundary_out(self) -> list[LaneDef]:
        """Lanes that end at the map edge (traffic sinks)."""
        return [l for l in self.lanes.values()
                if self.nodes[l.end].kind == "boundary"]

    def connectors_from(self, lane_id: str) -> list[ConnectorDef]:
        return [c for c in self.connectors if c.src == lane_id]

    # --- validation ------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError on dangling references or broken geometry."""
        for node in self.nodes.values():
            if node.kind not in NODE_KINDS:
                raise ValueError(f"node {node.id}: unknown kind {node.kind!r}")
        for lane in self.lanes.values():
            pts = np.asarray(lane.pts, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
                raise ValueError(f"lane {lane.id}: bad polyline {pts.shape}")
            if not np.all(np.isfinite(pts)):
                raise ValueError(f"lane {lane.id}: non-finite points")
            for ref in (lane.start, lane.end):
                if ref not in self.nodes:
                    raise ValueError(f"lane {lane.id}: unknown node {ref!r}")
            if (self.nodes[lane.end].kind == "signal"
                    and lane.approach is None):
                raise ValueError(
                    f"lane {lane.id}: ends at a signal but has no approach")
        for con in self.connectors:
            if con.turn not in TURNS:
                raise ValueError(f"connector {con.id}: bad turn {con.turn!r}")
            for ref in (con.src, con.dst):
                if ref not in self.lanes:
                    raise ValueError(
                        f"connector {con.id}: unknown lane {ref!r}")
            pts = np.asarray(con.pts, dtype=float)
            src, dst = self.lanes[con.src], self.lanes[con.dst]
            if not np.allclose(pts[0], src.pts[-1], atol=1e-6):
                raise ValueError(
                    f"connector {con.id}: start detached from lane {con.src}")
            if not np.allclose(pts[-1], dst.pts[0], atol=1e-6):
                raise ValueError(
                    f"connector {con.id}: end detached from lane {con.dst}")
        for road in self.roads:
            pts = np.asarray(road.pts, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
                raise ValueError(f"road {road.id}: bad polyline {pts.shape}")
            if road.half_width <= 0:
                raise ValueError(f"road {road.id}: non-positive width")

    # --- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "x": n.x, "y": n.y, "kind": n.kind,
                 "signal_offset": n.signal_offset}
                for n in self.nodes.values()],
            "lanes": [
                {"id": l.id, "pts": np.asarray(l.pts).tolist(),
                 "speed": l.speed, "start": l.start, "end": l.end,
                 "approach": l.approach}
                for l in self.lanes.values()],
            "connectors": [
                {"id": c.id, "pts": np.asarray(c.pts).tolist(),
                 "speed": c.speed, "src": c.src, "dst": c.dst, "turn": c.turn}
                for c in self.connectors],
            "roads": [
                {"id": r.id, "pts": np.asarray(r.pts).tolist(),
                 "half_width": r.half_width}
                for r in self.roads],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoadGraph:
        graph = cls()
        for n in data["nodes"]:
            graph.nodes[n["id"]] = RoadNode(
                n["id"], float(n["x"]), float(n["y"]), n["kind"],
                float(n.get("signal_offset", 0.0)))
        for l in data["lanes"]:
            graph.lanes[l["id"]] = LaneDef(
                l["id"], np.asarray(l["pts"], dtype=float), float(l["speed"]),
                l["start"], l["end"], l.get("approach"))
        for c in data["connectors"]:
            graph.connectors.append(ConnectorDef(
                c["id"], np.asarray(c["pts"], dtype=float), float(c["speed"]),
                c["src"], c["dst"], c["turn"]))
        for r in data.get("roads", []):
            graph.roads.append(RoadDef(
                r["id"], np.asarray(r["pts"], dtype=float),
                float(r["half_width"])))
        return graph

    def save_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load_json(cls, path) -> RoadGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))
