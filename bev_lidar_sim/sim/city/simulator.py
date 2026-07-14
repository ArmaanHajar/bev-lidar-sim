"""City traffic simulation driven by a RoadGraph.

`CitySimulator` instantiates the shared traffic machinery from any
`RoadGraph` — signals and all-way stops from node records, `Path`s from lane
and connector records — then routes every vehicle boundary-to-boundary with
breadth-first search over the connector graph. It never inspects the map's
shape, so a graph loaded from JSON or emitted by a future importer drops in
via the `graph` argument. The default is the procedural grid from
`city.graph` (the static `CityWorld` decoration still assumes that grid).
"""

from __future__ import annotations

import math
from collections import deque

from ...maps.roadgraph import RoadGraph
from ..traffic import AllWayStop, Path, Signal, Simulator
from .graph import build_city_roadgraph
from .world import CityIntersection, CityWorld


class CitySimulator(Simulator):
    """Traffic simulation on a generated four-by-four downtown street grid."""

    def __init__(self, seed: int = 1, graph: RoadGraph | None = None,
                 driving_mode: str = "normal", ego_od: tuple | None = None,
                 district_name: str = "GRID DISTRICT",
                 seed_vehicles: int = 19, max_vehicles: int = 29,
                 world_params: dict | None = None):
        super().__init__(seed, driving_mode)
        self.ego_status = "cruising to destination"
        self.district_name = district_name
        self.graph = graph if graph is not None else build_city_roadgraph()
        self.ego_od = ego_od          # (origin, dest) boundary node ids
        self.seed_vehicles = seed_vehicles
        self.max_vehicles = max_vehicles

        self._build_controls()
        self.world = CityWorld(seed, self.graph, self.intersections,
                               **(world_params or {}))
        self._build_paths()

        ego_paths = self._ego_route()
        ego_start = 0 if ego_od else (2 if len(ego_paths) > 3 else 0)
        self.ego = self._spawn_paths(ego_paths, is_ego=True, u=5.0, v=7.5,
                                     start_index=ego_start)
        self._seed_traffic()
        self.next_spawn_t = float(self.rng.uniform(1.0, 2.5))

    # --- Sim objects from the RoadGraph -----------------------------------
    def _build_controls(self) -> None:
        self.intersections = []
        for node in self.graph.nodes.values():
            if node.kind == "signal":
                self.signals[node.id] = Signal(node.x,
                                               offset=node.signal_offset)
            elif node.kind == "stop":
                self.stop_mgrs[node.id] = AllWayStop(node.x, node.y)
            else:
                continue
            self.intersections.append(
                CityIntersection(node.id, node.x, node.y, node.kind))

    def _build_paths(self) -> None:
        self.lanes = {}
        self.incoming_boundary = []
        self.outgoing_boundary = []
        for ld in self.graph.lanes.values():
            lane = Path(ld.pts, f"lane:{ld.id}", speed=ld.speed)
            lane.start_node = ld.start
            lane.end_node = ld.end
            lane.turn_kind = None
            end = self.graph.nodes[ld.end]
            if end.kind == "signal":
                lane.stops = [(lane.length,
                               ("light", self.signals[ld.end], ld.approach))]
            elif end.kind == "stop":
                lane.stops = [(lane.length, ("sign", self.stop_mgrs[ld.end]))]
            self.lanes[ld.id] = lane
            if self.graph.nodes[ld.start].kind == "boundary":
                self.incoming_boundary.append(lane)
            if end.kind == "boundary":
                self.outgoing_boundary.append(lane)

        self.transitions = {}
        for cd in self.graph.connectors:
            src, dst = self.lanes[cd.src], self.lanes[cd.dst]
            connector = Path(cd.pts, cd.id, is_turn=cd.turn != "straight",
                             speed=cd.speed)
            connector.turn_kind = cd.turn
            connector.start_node = src.end_node
            connector.end_node = src.end_node
            self.transitions.setdefault(id(src), []).append(
                (connector, dst, cd.turn))

    # --- Routing -----------------------------------------------------------
    def _ego_route(self):
        """The ego's next trip: the fixed A->B mission, or a random one."""
        if self.ego_od is None:
            return self._random_route(min_lanes=5, min_turns=1, skip=2)
        origin, dest = self.ego_od
        starts = [l for l in self.incoming_boundary if l.start_node == origin]
        goals = [l for l in self.outgoing_boundary if l.end_node == dest]
        if not starts or not goals:
            raise ValueError(f"ego_od {self.ego_od}: no boundary lane match")
        paths = self._route_between(starts[0], goals[0])
        if paths is None:
            raise ValueError(f"ego_od {self.ego_od}: no route exists")
        return paths

    def ego_remaining(self) -> float:
        """Meters left on the ego's current trip (for the HUD)."""
        left = self.ego.path.length - self.ego.u
        for p, u0 in self.ego.route:
            left += p.length - u0
        return max(left, 0.0)

    def _route_between(self, start: Path, goal: Path):
        queue = deque([start])
        previous = {id(start): None}
        objects = {id(start): start}
        while queue:
            lane = queue.popleft()
            if lane is goal:
                break
            for connector, nxt, _ in self.transitions.get(id(lane), []):
                if id(nxt) in previous:
                    continue
                previous[id(nxt)] = (id(lane), connector)
                objects[id(nxt)] = nxt
                queue.append(nxt)
        if id(goal) not in previous:
            return None

        hops = []
        current = id(goal)
        while previous[current] is not None:
            prior, connector = previous[current]
            hops.append((connector, objects[current]))
            current = prior
        hops.reverse()
        paths = [start]
        for connector, lane in hops:
            paths.extend([connector, lane])
        return paths

    def _random_route(self, min_lanes=3, require_clear=False, min_turns=0,
                      skip=0):
        starts = list(self.incoming_boundary)
        goals = list(self.outgoing_boundary)
        self.rng.shuffle(starts)
        self.rng.shuffle(goals)
        # Routes come from a shortest-hop search, which favors straight
        # lines; without min_turns whether a trip ever turns is left to the
        # luck of the origin/destination draw (the ego once drove straight
        # for 3+ minutes). Turns are counted past `skip` — the ego spawns at
        # paths[2], so a turn in the skipped stub must not satisfy the quota.
        # min_turns is a preference, not a guarantee: it relaxes to any
        # acceptable route before the defensive fallback.
        for need_turns in (min_turns, 0):
            for start in starts:
                if require_clear and self._path_blocked(start, 0.0, 18.0):
                    continue
                for goal in goals:
                    if goal.end_node == start.start_node:
                        continue
                    paths = self._route_between(start, goal)
                    if paths is None or (len(paths) + 1) // 2 < min_lanes:
                        continue
                    # The ego skips its boundary stub when receiving a new
                    # destination, so clearance must be checked where it will
                    # actually appear rather than only at paths[0].
                    spawn_path = paths[min(skip, len(paths) - 1)]
                    if (require_clear
                            and self._path_blocked(spawn_path, 0.0, 18.0)):
                        continue
                    if sum(p.is_turn for p in paths[skip:]) >= need_turns:
                        return paths
            if min_turns == 0:
                break
        # The right/straight/stop-left graph is strongly connected, so this is
        # only a defensive fallback if future map rules remove movements.
        for start in starts:
            for goal in goals:
                paths = self._route_between(start, goal)
                if paths is not None:
                    return paths
        raise RuntimeError("city lane graph contains no boundary route")

    # --- Traffic ---------------------------------------------------------
    def _spawn_paths(self, paths, is_ego=False, kind="car", u=0.0, v=7.0,
                     start_index=0):
        veh = self._spawn(paths[start_index], u=u, v=v, kind=kind,
                          is_ego=is_ego)
        veh.route = [(p, 0.0) for p in paths[start_index + 1:]]
        return veh

    def _path_blocked(self, path, u, radius) -> bool:
        return any(v.path is path and abs(v.u - u) < radius
                   for v in self.vehicles)

    def _seed_traffic(self) -> None:
        for _ in range(42):
            if len(self.vehicles) >= self.seed_vehicles:
                break
            paths = self._random_route(min_lanes=3)
            lane_indices = list(range(0, len(paths), 2))
            idx = int(self.rng.choice(lane_indices[:-1] or lane_indices))
            path = paths[idx]
            u = float(self.rng.uniform(0.12, 0.72) * path.length)
            if self._path_blocked(path, u, 13.0):
                continue
            kind = "truck" if self.rng.random() < 0.12 else "car"
            self._spawn_paths(paths, kind=kind, u=u,
                              v=float(self.rng.uniform(5.5, 9.5)),
                              start_index=idx)

    def _run_events(self) -> None:
        if self.t < self.next_spawn_t:
            return
        if len(self.vehicles) < self.max_vehicles:
            paths = self._random_route(min_lanes=3, require_clear=True)
            if not self._path_blocked(paths[0], 0.0, 18.0):
                kind = "truck" if self.rng.random() < 0.10 else "car"
                self._spawn_paths(paths, kind=kind, v=6.5)
        self.next_spawn_t = self.t + float(self.rng.uniform(2.2, 4.8))

    def _transitions(self) -> None:
        keep = []
        ego_restarted = False
        for veh in self.vehicles:
            while veh.u >= veh.path.length:
                carry = veh.u - veh.path.length
                if veh.route:
                    path, u0 = veh.route.pop(0)
                    veh.path, veh.u = path, u0 + carry
                    continue
                if veh.is_ego:
                    for mgr in self.stop_mgrs.values():
                        mgr.forget(veh.vid)
                    if self.ego_od is not None:
                        # Fixed A->B mission: restart the same trip from A.
                        paths = self._ego_route()
                        start = 0
                        ego_restarted = True
                        self.ego_status = "arrived — restarting trip"
                    else:
                        paths = self._random_route(min_lanes=5,
                                                   require_clear=True,
                                                   min_turns=1, skip=2)
                        start = 2 if len(paths) > 3 else 0
                        self.ego_status = "new destination"
                    veh.path = paths[start]
                    veh.route = [(p, 0.0) for p in paths[start + 1:]]
                    veh.u = min(carry, max(veh.path.length - 0.1, 0.0))
                    break
                veh.u = math.inf
                break
            if math.isinf(veh.u):
                for mgr in self.stop_mgrs.values():
                    mgr.forget(veh.vid)
            else:
                keep.append(veh)
        if ego_restarted:
            # The teleporting ego must not materialize inside a vehicle
            # already queued at the trip origin; despawn any such blocker.
            blockers = [v for v in keep
                        if not v.is_ego and v.path is self.ego.path
                        and abs(v.u - self.ego.u) < 12.0]
            for v in blockers:
                for mgr in self.stop_mgrs.values():
                    mgr.forget(v.vid)
            keep = [v for v in keep if v not in blockers]
        self.vehicles = keep

    # --- Renderer helpers ------------------------------------------------
    def turn_signal(self, veh):
        if veh.path.is_turn:
            return getattr(veh.path, "turn_kind", None)
        if veh.route and veh.path.length - veh.u < 28.0:
            nxt = veh.route[0][0]
            if nxt.is_turn:
                return getattr(nxt, "turn_kind", None)
        return None

    def blinking(self, veh) -> bool:
        return self.turn_signal(veh) in ("left", "right") \
            and int(self.t * 3.4) % 2 == 0
