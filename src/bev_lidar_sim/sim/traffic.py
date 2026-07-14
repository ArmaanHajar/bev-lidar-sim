"""Shared traffic machinery: paths, vehicles, controls, and the sim core.

Every moving vehicle follows a `Path` — a polyline with arc-length lookup —
at a longitudinal position `u`, so all driving logic stays one-dimensional.
x/y/heading exist only through `Path.pose(u)` for rendering, sensing, and
intersection box checks; never write behavior code against x/y.

Behavior shared by every scenario:
  * Car-following with the Intelligent Driver Model (IDM); a red light or an
    un-granted stop line acts as a virtual stopped car, so the same code that
    follows a lead vehicle also produces smooth stops at signals.
  * Signals run real phases (main green -> yellow -> all-red -> cross green
    -> ...), so one axis flows while the other waits and vice versa.
  * Yellow-light dilemma: a vehicle continues through a yellow only if it can
    no longer stop comfortably before the line.
  * All-way stops are negotiated in arrival order: stop fully, queue, and
    proceed when the intersection box is clear and it is your turn.

`Simulator` is the scenario-independent core. Subclasses — the legacy
arterial in `arterial.py` and the generated city in `city/` — build the path
network, controls, static world, and spawning, and inherit car-following,
control stops, stepping, and the ego-frame sensor query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sensors.scene import Box, Scene

# --- shared appearance and surface constants -------------------------------
CAR_COLORS = ["#c85a4e", "#d9a441", "#79a86b", "#8f7fce",
              "#c9cdd4", "#5b87b0", "#b06a92"]
EGO_COLOR = "#22d3ee"

# Reflectivity by surface type (what the LiDAR intensity channel sees).
REFL = {"curb": 0.20, "building": 0.55, "tree": 0.45, "pole": 0.60,
        "sign": 0.95, "car": 0.85, "truck": 0.75}

# Comfortable turn-entry speed used when braking for an upcoming turn.
TURN_SPEED = 4.6


# --- Paths ----------------------------------------------------------------
class Path:
    """A polyline with arc-length parametrization: u (meters) -> pose."""

    def __init__(self, pts, name: str = "", is_turn: bool = False,
                 speed: float = 13.5):
        self.pts = np.asarray(pts, dtype=float)
        d = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.seg_heading = np.arctan2(d[:, 1], d[:, 0])
        self.name = name
        self.is_turn = is_turn
        self.speed = speed          # speed limit on this path
        self.stops = []             # [(u_line, control), ...] sorted by u

    def pose(self, u: float):
        u = min(max(u, 0.0), self.length)
        i = int(np.searchsorted(self.cum, u, side="right")) - 1
        i = max(0, min(i, len(self.seg_len) - 1))
        f = (u - self.cum[i]) / max(self.seg_len[i], 1e-9)
        p = self.pts[i] + f * (self.pts[i + 1] - self.pts[i])
        return float(p[0]), float(p[1]), float(self.seg_heading[i])


# --- Traffic controls -----------------------------------------------------
@dataclass
class Signal:
    """A two-phase signal: main-approach phase, then cross-approach phase."""

    x: float
    offset: float = 0.0
    g_main: float = 10.0
    y_main: float = 2.6
    clear1: float = 1.2      # all-red after the main phase
    g_cross: float = 7.0
    y_cross: float = 2.6
    clear2: float = 1.2      # all-red after the cross phase

    @property
    def period(self) -> float:
        return (self.g_main + self.y_main + self.clear1
                + self.g_cross + self.y_cross + self.clear2)

    def state(self, t: float, approach: str) -> str:
        p = (t + self.offset) % self.period
        if approach == "main":
            if p < self.g_main:
                return "green"
            if p < self.g_main + self.y_main:
                return "yellow"
            return "red"
        q = p - (self.g_main + self.y_main + self.clear1)
        if q < 0:
            return "red"
        if q < self.g_cross:
            return "green"
        if q < self.g_cross + self.y_cross:
            return "yellow"
        return "red"


class AllWayStop:
    """First-come-first-served arbitration for a 4-way stop."""

    BOX = 5.4  # half-size of the conflict box around the center

    def __init__(self, x: float, y: float = 0.0):
        self.x = x
        self.y = y
        self.queue: list = []      # vids in arrival order
        self.granted = None        # vid currently allowed to go
        self._entered = False
        self._grant_t = 0.0

    def register(self, vid: int) -> None:
        if vid not in self.queue:
            self.queue.append(vid)

    def forget(self, vid: int) -> None:
        if vid in self.queue:
            self.queue.remove(vid)
        if self.granted == vid:
            self.granted = None

    def _inside(self, veh) -> bool:
        x, y, _ = veh.pose()
        return abs(x - self.x) < self.BOX and abs(y - self.y) < self.BOX

    def update(self, sim) -> None:
        vmap = {v.vid: v for v in sim.vehicles}
        if self.granted is not None:
            veh = vmap.get(self.granted)
            if veh is None:
                self.forget(self.granted)
            else:
                if self._inside(veh):
                    self._entered = True
                stale = sim.t - self._grant_t > 12.0
                if (self._entered and not self._inside(veh)) or stale:
                    self.forget(self.granted)
        if self.granted is None:
            self.queue = [vid for vid in self.queue if vid in vmap]
            if self.queue and not any(self._inside(v) for v in sim.vehicles):
                self.granted = self.queue[0]
                self._entered = False
                self._grant_t = sim.t


# --- Vehicles -------------------------------------------------------------
@dataclass
class Vehicle:
    vid: int
    path: Path
    u: float
    v: float
    v0: float = 13.0          # desired speed (m/s)
    length: float = 4.5
    width: float = 1.9
    kind: str = "car"         # "car" | "truck"
    color: str = "#c85a4e"
    is_ego: bool = False
    route: list = field(default_factory=list)   # [(Path, u_start), ...]
    turn_at_u: float = None   # divert onto route[0] at this u (else at path end)
    acc: float = 0.0          # last commanded acceleration (for brake lights)

    def pose(self):
        return self.path.pose(self.u)


# IDM parameters (shared).
IDM = dict(a=1.5, b=2.2, s0=2.0, T=1.4, delta=4.0)


def idm_accel(v: float, v0: float, gap: float, dv: float) -> float:
    """Longitudinal acceleration from the Intelligent Driver Model."""
    a, b, s0, T = IDM["a"], IDM["b"], IDM["s0"], IDM["T"]
    if not math.isfinite(gap):
        return a * (1.0 - (v / max(v0, 0.1)) ** IDM["delta"])
    gap = max(gap, 0.1)
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / max(v0, 0.1)) ** IDM["delta"] - (s_star / gap) ** 2)


# --- Simulator core ---------------------------------------------------------
class Simulator:
    """Scenario-independent traffic engine.

    Subclass contract: `__init__` must build `self.world` (curbs +
    static_boxes shared by renderer and LiDAR), the path network with
    per-path `stops`, `self.signals` / `self.stop_mgrs`, and `self.ego`,
    then implement `_transitions` (path-end handling) and `_run_events`
    (spawning).
    """

    def __init__(self, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0.0
        self.next_vid = 0
        self.vehicles: list = []
        self.ego_status = "cruising"
        self.manual = False        # user drives the ego (throttle/brake only)
        self.manual_cmd = 0.0      # +1 accelerate, 0 coast, -1 brake
        self.signals: dict = {}
        self.stop_mgrs: dict = {}

    # -- spawning ------------------------------------------------------------
    def _spawn(self, path, u, v, kind="car", is_ego=False):
        if kind == "truck":
            v0 = float(self.rng.uniform(9.5, 11.5))
            length, width = 7.6, 2.4
            color = "#aeb6c2"
        else:
            v0 = 13.5 if is_ego else float(self.rng.uniform(10.5, 14.5))
            length, width = 4.5, 1.9
            color = EGO_COLOR if is_ego else str(self.rng.choice(CAR_COLORS))
        veh = Vehicle(vid=self.next_vid, path=path, u=u, v=v, v0=v0,
                      length=length, width=width, kind=kind, color=color,
                      is_ego=is_ego)
        self.next_vid += 1
        self.vehicles.append(veh)
        return veh

    # -- perception of the road ahead -----------------------------------------
    def _leader_gap(self, veh, by_path):
        """(gap, closing speed) to the nearest vehicle ahead along the route."""
        best_ds, lead_v, lead_len = math.inf, 0.0, 4.5
        for other in by_path.get(id(veh.path), []):
            if other is not veh and other.u > veh.u + 0.01:
                best_ds = other.u - veh.u
                lead_v, lead_len = other.v, other.length
                break
        limit = veh.turn_at_u if veh.turn_at_u is not None else veh.path.length
        off = limit - veh.u
        for p, u0 in veh.route[:2]:
            if off >= best_ds:
                break
            for other in by_path.get(id(p), []):
                if other is veh:
                    continue
                if other.u >= u0 - 0.1:
                    ds = off + (other.u - u0)
                    if 0.01 < ds < best_ds:
                        best_ds, lead_v, lead_len = ds, other.v, other.length
                    break
            off += p.length - u0
        gap = best_ds - (veh.length + lead_len) / 2.0
        return gap, veh.v - lead_v

    def _control_gap(self, veh):
        """(gap, label) to the nearest stop line the vehicle must honor."""
        best, label = math.inf, None
        for u_line, control in veh.path.stops:
            dist = u_line - veh.u
            if dist < -1.0 or dist > 90.0:
                continue
            if control[0] == "light":
                _, sig, approach = control
                st = sig.state(self.t, approach)
                if st == "green":
                    continue
                if st == "yellow" and dist < veh.v * veh.v / 6.0 + 1.5:
                    continue    # cannot stop comfortably -> clear the junction
                if st == "red" and dist < 1.0 and veh.v > 7.0:
                    continue    # already committed
                lbl = "red light" if st == "red" else "yellow light"
            else:
                mgr = control[1]
                if mgr.granted == veh.vid:
                    continue
                # A full stop just short of the line puts you in the queue
                # (bumper gap, so trucks qualify too).
                if dist - veh.length / 2.0 < 3.6 and veh.v < 0.5:
                    mgr.register(veh.vid)
                lbl = "all-way stop"
            gap = dist - veh.length / 2.0
            if gap < best:
                best, label = gap, lbl
        return best, label

    # -- main step -------------------------------------------------------------
    def step(self, dt: float) -> None:
        self.t += dt
        for mgr in self.stop_mgrs.values():
            mgr.update(self)

        by_path = {}
        for v in self.vehicles:
            by_path.setdefault(id(v.path), []).append(v)
        for lst in by_path.values():
            lst.sort(key=lambda z: z.u)

        for veh in self.vehicles:
            lead_gap, lead_dv = self._leader_gap(veh, by_path)
            ctrl_gap, ctrl_label = self._control_gap(veh)

            if ctrl_gap < lead_gap:
                gap, dv, source = ctrl_gap, veh.v, ctrl_label
            else:
                gap, dv = lead_gap, lead_dv
                source = "traffic" if math.isfinite(lead_gap) else "free"

            v0 = min(veh.v0, veh.path.speed)
            acc = idm_accel(veh.v, v0, gap, dv)

            # Slow down in time for an upcoming turn.
            if veh.turn_at_u is not None:
                d = veh.turn_at_u - veh.u
                if 0.0 < d < 45.0:
                    allow = math.sqrt(TURN_SPEED ** 2
                                      + 2.0 * 1.8 * max(d - 2.0, 0.0))
                    if veh.v > allow:
                        acc = min(acc, -2.2)
            elif veh.route and veh.route[0][0].is_turn:
                # General lane networks transition onto a turn connector at
                # the end of the current path rather than using turn_at_u.
                d = veh.path.length - veh.u
                if 0.0 < d < 45.0:
                    allow = math.sqrt(TURN_SPEED ** 2
                                      + 2.0 * 1.8 * max(d - 2.0, 0.0))
                    if veh.v > allow:
                        acc = min(acc, -2.2)
            veh.acc = max(acc, -6.0)   # physical braking limit
            if veh.is_ego:
                if self.manual:
                    # The user is the driver: throttle/brake override the IDM
                    # (the ego still follows its lane; steering is the path's).
                    if self.manual_cmd > 0:
                        veh.acc = 2.6
                    elif self.manual_cmd < 0:
                        veh.acc = -6.0
                    else:
                        veh.acc = -0.8 if veh.v > 0 else 0.0
                    if veh.v > 19.0:            # ~70 km/h cap
                        veh.acc = min(veh.acc, 0.0)
                    self.ego_status = "manual control"
                else:
                    self.ego_status = self._status(source, veh)

        for veh in self.vehicles:
            veh.v = max(0.0, veh.v + veh.acc * dt)
            veh.u += veh.v * dt

        self._transitions()
        self._run_events()

    def _status(self, source, veh) -> str:
        if source in (None, "free"):
            return "cruising"
        if source == "traffic":
            return "following traffic"
        if veh.v < 0.5:
            return "waiting — " + source
        return "stopping — " + source

    # -- scenario hooks ----------------------------------------------------
    def _transitions(self) -> None:
        """Move vehicles whose u passed the end of their path (per scenario)."""
        raise NotImplementedError

    def _run_events(self) -> None:
        """Timed spawning and other scenario events."""
        raise NotImplementedError

    # -- queries for rendering / sensing ---------------------------------------
    def blinking(self, veh) -> bool:
        """True when the vehicle's (right) turn signal is lit this instant."""
        soon = (veh.turn_at_u is not None
                and 0.0 <= veh.turn_at_u - veh.u < 30.0)
        if soon or veh.path.is_turn:
            return int(self.t * 3.4) % 2 == 0
        return False

    def light_state(self, key, approach: str) -> str:
        return self.signals[key].state(self.t, approach)

    def ego_scene(self, max_range: float = 48.0) -> Scene:
        """The world in the ego frame (ego at origin facing +x) for the LiDAR."""
        ex, ey, eh = self.ego.pose()
        c, s = math.cos(-eh), math.sin(-eh)

        def tf(x, y):
            dx, dy = x - ex, y - ey
            return dx * c - dy * s, dx * s + dy * c

        scene = Scene()
        r = max_range + 6.0
        for a, b, refl in self.world.curbs:
            if max(a[0], b[0]) < ex - r or min(a[0], b[0]) > ex + r:
                continue
            if max(a[1], b[1]) < ey - r or min(a[1], b[1]) > ey + r:
                continue
            scene.add_wall(tf(*a), tf(*b), refl=refl)
        for box in self.world.static_boxes():
            if abs(box.cx - ex) > r + 12.0 or abs(box.cy - ey) > r + 12.0:
                continue
            bx, by = tf(box.cx, box.cy)
            scene.boxes.append(Box(bx, by, box.heading - eh, box.length,
                                   box.width, box.label, box.refl))
        for veh in self.vehicles:
            if veh.is_ego:
                continue
            vx, vy, vh = veh.pose()
            if abs(vx - ex) > r or abs(vy - ey) > r:
                continue
            bx, by = tf(vx, vy)
            scene.boxes.append(Box(bx, by, vh - eh, veh.length, veh.width,
                                   veh.kind, REFL.get(veh.kind, 0.85)))
        return scene
