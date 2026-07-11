"""A small rule-following traffic microsimulator.

The world is a straight two-lane road (eastbound + westbound) with a few
signalized intersections and a stop sign. Every vehicle drives with the
Intelligent Driver Model (IDM), the standard car-following model. Traffic
lights and stop signs are handled the way real microsimulators do it: a red
light or an un-cleared stop line acts as a *virtual stopped car* at the line,
so the same IDM code that keeps distance to the lead car also brings a vehicle
to a smooth stop at a signal.

Coordinates: each vehicle has a longitudinal position `u` (meters travelled in
its own direction of travel). `pose()` maps (lane, u) to world (x, y, heading),
so all driving logic is 1-D and only rendering cares about x/y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- World geometry -------------------------------------------------------
ROAD_LENGTH = 400.0
LANE_E_Y = -2.0          # eastbound lane centerline (moves +x)
LANE_W_Y = +2.0          # westbound lane centerline (moves -x)
LANE_HALF = 4.0          # road edge (curb) offset from centerline
INTER_HALF = 5.0         # half-width of an intersection box

INTERSECTIONS = [120.0, 200.0, 280.0]   # world-x of cross streets
LIGHT_XS = [120.0, 280.0]               # signalized intersections
STOP_SIGN_XS = [200.0]                  # eastbound stop sign


# --- Traffic controls -----------------------------------------------------
@dataclass
class TrafficLight:
    x: float
    offset: float = 0.0
    green: float = 8.0
    yellow: float = 2.5
    red: float = 8.0

    @property
    def period(self) -> float:
        return self.green + self.yellow + self.red

    def state(self, t: float) -> str:
        p = (t + self.offset) % self.period
        if p < self.green:
            return "green"
        if p < self.green + self.yellow:
            return "yellow"
        return "red"


@dataclass
class StopSign:
    x: float


# --- Vehicle --------------------------------------------------------------
@dataclass
class Vehicle:
    vid: int
    lane: str                 # "E" or "W"
    u: float
    v: float
    length: float = 4.5
    width: float = 2.0
    v0: float = 13.0          # desired speed (m/s)
    is_ego: bool = False
    color: str = "#ff8c1a"
    cleared: set = field(default_factory=set)   # stop-sign ids already obeyed
    exit_x: float | None = None                 # turn off the road here
    blink_until: float = -1.0

    @property
    def heading(self) -> float:
        return 0.0 if self.lane == "E" else math.pi


# IDM parameters (shared).
IDM = dict(a=1.4, b=2.0, s0=2.0, T=1.5, delta=4.0)


def idm_accel(v: float, v0: float, gap: float, dv: float) -> float:
    """Longitudinal acceleration from the Intelligent Driver Model.

    gap: bumper-to-bumper distance to the obstacle ahead (m).
    dv:  approach rate = v - v_lead (positive when closing in).
    """
    a, b, s0, T = IDM["a"], IDM["b"], IDM["s0"], IDM["T"]
    gap = max(gap, 0.1)
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / v0) ** IDM["delta"] - (s_star / gap) ** 2)


def pose(lane: str, u: float) -> tuple[float, float, float]:
    if lane == "E":
        return u, LANE_E_Y, 0.0
    return ROAD_LENGTH - u, LANE_W_Y, math.pi


class Simulator:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.next_vid = 0
        self.lights = [TrafficLight(x=LIGHT_XS[0], offset=0.0),
                       TrafficLight(x=LIGHT_XS[1], offset=9.0)]
        self.signs = [StopSign(x=x) for x in STOP_SIGN_XS]
        self.vehicles: list[Vehicle] = []
        self._lines = {"E": self._stop_lines("E"), "W": self._stop_lines("W")}

        # Ego starts near the beginning of the eastbound lane.
        self.ego = self._spawn("E", u=10.0, v=10.0, ego=True)
        self._seed_traffic()
        self.next_event = 3.0

    # -- setup helpers ----------------------------------------------------
    def _spawn(self, lane, u, v, ego=False) -> Vehicle:
        v0 = 14.0 if ego else float(self.rng.uniform(11.0, 15.0))
        color = "#19d3ff" if ego else self.rng.choice(
            ["#ff8c1a", "#f4f4f4", "#ff5b5b", "#8ce06a", "#c58cff"])
        veh = Vehicle(vid=self.next_vid, lane=lane, u=u, v=v, v0=v0,
                      is_ego=ego, color=str(color))
        self.next_vid += 1
        self.vehicles.append(veh)
        return veh

    def _seed_traffic(self) -> None:
        for u in np.arange(40.0, ROAD_LENGTH, 45.0):
            if abs(u - self.ego.u) > 20:
                self._spawn("E", u=float(u), v=10.0)
        for u in np.arange(30.0, ROAD_LENGTH, 55.0):
            self._spawn("W", u=float(u), v=10.0)

    def _stop_lines(self, lane: str):
        """Return sorted [(u_line, kind, ctrl)] the given lane must obey."""
        lines = []
        for lt in self.lights:
            x = lt.x
            u = (x - INTER_HALF) if lane == "E" else ROAD_LENGTH - (x + INTER_HALF)
            lines.append((u, "light", lt))
        if lane == "E":
            for sg in self.signs:
                lines.append((sg.x - INTER_HALF, "sign", sg))
        return sorted(lines, key=lambda z: z[0])

    # -- per-step obstacle logic -----------------------------------------
    def _lead_gap(self, veh, lane_sorted):
        """Gap and speed of the nearest vehicle ahead in the same lane."""
        best = (math.inf, 0.0)
        for other in lane_sorted:
            if other.u > veh.u + 1e-6:
                gap = (other.u - veh.u) - (veh.length + other.length) / 2.0
                return gap, other.v
        return best

    def _control_gap(self, veh):
        """Nearest stop line the vehicle must currently honor (gap, id)."""
        best_gap, best_id = math.inf, None
        for u_line, kind, ctrl in self._lines[veh.lane]:
            dist = u_line - veh.u
            if dist <= -1.0:
                continue  # already through the line
            if kind == "light":
                st = ctrl.state(self.t)
                if st == "green":
                    continue
                if st == "yellow" and dist < max(veh.v * 2.0, 6.0):
                    continue  # too close to stop safely -> clear the intersection
            else:  # stop sign
                if id(ctrl) in veh.cleared:
                    continue
            gap = dist - veh.length / 2.0
            if gap < best_gap:
                best_gap, best_id = gap, id(ctrl)
        return best_gap, best_id

    # -- main step --------------------------------------------------------
    def step(self, dt: float) -> None:
        self.t += dt
        by_lane = {"E": sorted((v for v in self.vehicles if v.lane == "E"),
                               key=lambda z: z.u),
                   "W": sorted((v for v in self.vehicles if v.lane == "W"),
                               key=lambda z: z.u)}

        for veh in self.vehicles:
            lead_gap, lead_v = self._lead_gap(veh, by_lane[veh.lane])
            ctrl_gap, ctrl_id = self._control_gap(veh)

            if ctrl_gap < lead_gap:
                gap, dv = ctrl_gap, veh.v          # obstacle is stationary
                # Register a full stop at a stop sign, then let the car proceed.
                if ctrl_id is not None and ctrl_gap < 3.0 and veh.v < 0.6:
                    veh.cleared.add(ctrl_id)
            else:
                gap, dv = lead_gap, veh.v - lead_v

            acc = idm_accel(veh.v, veh.v0, gap, dv)
            veh.v = max(0.0, veh.v + acc * dt)
            veh.u += veh.v * dt

        self._handle_exits_and_bounds()
        self._run_events()

    def _handle_exits_and_bounds(self) -> None:
        keep = []
        for veh in self.vehicles:
            if veh.is_ego:
                if veh.u > ROAD_LENGTH:            # loop the ego forever
                    veh.u = 0.0
                    veh.cleared.clear()
                keep.append(veh)
                continue
            if veh.exit_x is not None:
                u_exit = (veh.exit_x if veh.lane == "E"
                          else ROAD_LENGTH - veh.exit_x)
                if veh.u >= u_exit:
                    continue  # turned off the road
            if veh.u <= ROAD_LENGTH:
                keep.append(veh)
        self.vehicles = keep

    def _run_events(self) -> None:
        """Occasionally have a car enter from a side street or turn off."""
        if self.t < self.next_event:
            return
        self.next_event = self.t + float(self.rng.uniform(4.0, 8.0))
        if self.rng.random() < 0.5:
            self._event_enter()
        else:
            self._event_exit()

    def _event_enter(self) -> None:
        x = float(self.rng.choice(INTERSECTIONS))
        u_enter = x + 3.0
        clear = all(v.lane != "E" or abs(v.u - u_enter) > 7.0
                    for v in self.vehicles)
        if clear:
            veh = self._spawn("E", u=u_enter, v=3.0)
            veh.blink_until = self.t + 2.5     # turn signal while merging

    def _event_exit(self) -> None:
        cands = [v for v in self.vehicles if not v.is_ego and v.lane == "E"]
        self.rng.shuffle(cands)
        for veh in cands:
            for x in INTERSECTIONS:
                if x - veh.u > 18.0:           # far enough to signal + turn
                    veh.exit_x = x
                    return

    # -- queries for rendering / sensing ---------------------------------
    def light_state(self, lt: TrafficLight) -> str:
        return lt.state(self.t)

    def blinking(self, veh: Vehicle) -> bool:
        if veh.blink_until > self.t:
            return int(self.t * 4) % 2 == 0
        if veh.exit_x is not None:
            u_exit = (veh.exit_x if veh.lane == "E"
                      else ROAD_LENGTH - veh.exit_x)
            if 0 < u_exit - veh.u < 20:
                return int(self.t * 4) % 2 == 0
        return False
