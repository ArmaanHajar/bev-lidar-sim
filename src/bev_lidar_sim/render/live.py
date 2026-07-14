"""Pygame rendering for the traffic sim: chase view of the street + ego LiDAR.

The chase view draws the same world geometry the LiDAR senses (curbs broken
at intersections, buildings, trees, poles) plus road furniture: lane paint,
stop lines, crosswalks, per-approach signal heads, and stop signs. Vehicles
get headlights, brake lights, and turn signals driven by the sim state.

Everything renders into a plain pygame Surface, so the same code powers the
live 60 fps window and headless GIF export.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
from pygame import gfxdraw

from ..sensors.lidar import Lidar2D
from ..sim import arterial as A


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


# --- palette ---------------------------------------------------------------
GROUND = _hex("#0c1013")
ASPHALT = _hex("#282d33")
SIDEWALK = _hex("#3b424c")
CURB = _hex("#5a6270")
LANE_YELLOW = _hex("#d8b13f")
MARK_WHITE = _hex("#cdd5dd")
BUILDING = _hex("#1c2530")
BUILDING_EDGE = _hex("#35404e")
TREE = _hex("#2a5c3c")
TREE_EDGE = _hex("#3d7f53")
GLASS = _hex("#131b26")
LIDAR_BG = _hex("#06090f")
RING = _hex("#182233")
RING_TEXT = _hex("#3d4c60")
HUD_TEXT = _hex("#9fb0c3")
EGO_CYAN = _hex("#22d3ee")
BOX_COLORS = {"car": _hex("#f0b429"), "truck": _hex("#ff6b5e")}

# --- layout ------------------------------------------------------------------
W = 1280
WORLD_H = 430            # top-down street panel height (px)
LIDAR_H = 430            # north-up LiDAR panel height (px)
GAP = 12
HEIGHT = WORLD_H + GAP + LIDAR_H
PXM_W = 6.2              # street view px per meter (~206 m x 69 m)
PXM_L = 4.8              # LiDAR view px per meter (~267 m x 90 m)

_LIDAR = Lidar2D(n_beams=540, max_range=45.0, noise_std=0.03, dropout=0.012)

# Point-color LUT (plasma), indexed by sqrt(intensity).
from matplotlib import colormaps  # noqa: E402  (only used to build the LUT)
_PLASMA = (np.asarray(colormaps["plasma"](np.linspace(0.05, 0.95, 256)))[:, :3]
           * 255).astype(int)

_fonts = {}


def _font(size, mono=False):
    key = (size, mono)
    if key not in _fonts:
        if not pygame.font.get_init():
            pygame.font.init()
        name = "menlo,monaco,couriernew" if mono else "helveticaneue,arial"
        _fonts[key] = pygame.font.SysFont(name, size,
                                          bold=mono)
    return _fonts[key]


def _text(surf, s, pos, color, size=13, mono=False, anchor="topleft"):
    img = _font(size, mono).render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)


# --- shape helpers ------------------------------------------------------------
def _rounded_rect_pts(length, width, r=0.45):
    """Rounded-rectangle polygon centered at the origin, long axis = x."""
    r = min(r, 0.45 * width, 0.45 * length)
    hl, hw = length / 2.0, width / 2.0
    pts = []
    corners = [(hl - r, hw - r, 0.0), (-hl + r, hw - r, math.pi / 2),
               (-hl + r, -hw + r, math.pi), (hl - r, -hw + r, -math.pi / 2)]
    for cx, cy, a0 in corners:
        for th in np.linspace(a0, a0 + math.pi / 2, 5):
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return np.array(pts)


def _place(pts, cx, cy, heading):
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([cx, cy])


def _offset(cx, cy, heading, fwd, right):
    c, s = math.cos(heading), math.sin(heading)
    return (cx + c * fwd + s * right, cy + s * fwd - c * right)


def _poly(surf, pts, color, edge=None):
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    gfxdraw.filled_polygon(surf, ipts, color)
    gfxdraw.aapolygon(surf, ipts, edge if edge else color)


def _circle(surf, x, y, r, color):
    x, y, r = int(round(x)), int(round(y)), max(int(round(r)), 1)
    gfxdraw.filled_circle(surf, x, y, r, color)
    if len(color) == 3:
        gfxdraw.aacircle(surf, x, y, r, color)


def _shade(color, f):
    return tuple(min(255, int(c * f)) for c in color)


# --- chase view -----------------------------------------------------------------
class WorldView:
    """World -> screen transform for the chase panel (camera follows ego)."""

    def __init__(self, ego_x, ego_y=0.0):
        self.ego_x = ego_x
        self.ego_y = ego_y

    def pt(self, x, y):
        return (W / 2.0 + (x - self.ego_x) * PXM_W,
                WORLD_H / 2.0 - (y - self.ego_y) * PXM_W)

    def rect(self, x, y, w, h):
        """World-space rect (x, y = lower-left corner) -> pygame Rect."""
        px, py = self.pt(x, y + h)
        return pygame.Rect(int(px), int(py),
                           max(int(w * PXM_W), 1), max(int(h * PXM_W), 1))


def draw_world(surf, sim, fps=None):
    if getattr(sim.world, "is_city", False):
        _draw_city_world(surf, sim, fps)
        return

    ego_x = sim.ego.pose()[0]
    cam = WorldView(ego_x)
    half_m = W / 2.0 / PXM_W
    x_lo, x_hi = ego_x - half_m, ego_x + half_m
    y_lim = WORLD_H / 2.0 / PXM_W

    surf.set_clip(pygame.Rect(0, 0, W, WORLD_H))
    surf.fill(GROUND)
    visible = [ix for ix in A.INTERSECTIONS if abs(ix - ego_x) < half_m + 20]

    # Sidewalks, then asphalt over them.
    for sy in (1, -1):
        surf.fill(SIDEWALK, cam.rect(x_lo, min(sy * 4, sy * 7), 2 * half_m, 3))
    for ix in visible:
        for sx in (-7, 4):
            surf.fill(SIDEWALK, cam.rect(ix + sx, -y_lim, 3, 2 * y_lim))
    surf.fill(ASPHALT, cam.rect(x_lo, -A.ROAD_HALF, 2 * half_m,
                                2 * A.ROAD_HALF))
    for ix in visible:
        surf.fill(ASPHALT, cam.rect(ix - A.ROAD_HALF, -y_lim,
                                    2 * A.ROAD_HALF, 2 * y_lim))

    # Lane paint.
    for xd in np.arange(math.floor(x_lo / 6.0) * 6.0, x_hi, 6.0):
        if any(abs(xd + 1.5 - ix) < 7.5 for ix in A.INTERSECTIONS):
            continue
        pygame.draw.line(surf, LANE_YELLOW, cam.pt(xd, 0),
                         cam.pt(xd + 3.0, 0), 2)
    for ix in visible:
        for yd in np.arange(-y_lim, y_lim, 6.0):
            if abs(yd + 1.5) < 7.5:
                continue
            pygame.draw.line(surf, LANE_YELLOW, cam.pt(ix, yd),
                             cam.pt(ix, yd + 3.0), 2)
        _crosswalk(surf, cam, ix)
        _stop_lines(surf, cam, ix)

    # Curbs (segments already have intersection gaps).
    for a, b, _ in sim.world.curbs:
        if max(a[0], b[0]) < x_lo or min(a[0], b[0]) > x_hi:
            continue
        pygame.draw.line(surf, CURB, cam.pt(*a), cam.pt(*b), 2)

    # Buildings and trees.
    for box in sim.world.buildings:
        if box.cx + box.length / 2 < x_lo or box.cx - box.length / 2 > x_hi:
            continue
        r = cam.rect(box.cx - box.length / 2, box.cy - box.width / 2,
                     box.length, box.width)
        surf.fill(BUILDING, r)
        pygame.draw.rect(surf, BUILDING_EDGE, r, 2)
    for box in sim.world.trees:
        if x_lo < box.cx < x_hi:
            px, py = cam.pt(box.cx, box.cy)
            _circle(surf, px, py, 1.15 * PXM_W, TREE)
            gfxdraw.aacircle(surf, int(px), int(py), int(1.15 * PXM_W),
                             TREE_EDGE)
    for box in sim.world.poles:
        if x_lo < box.cx < x_hi:
            px, py = cam.pt(box.cx, box.cy)
            _circle(surf, px, py, 2.5, _hex("#4a5563"))

    # Traffic controls: one signal head / sign per approach.
    for ix in visible:
        if ix in sim.signals:
            _signal_head(surf, cam, ix - 6.1, -6.4, sim.light_state(ix, "main"))
            _signal_head(surf, cam, ix + 6.1, 6.4, sim.light_state(ix, "main"))
            _signal_head(surf, cam, ix + 6.1, -6.4,
                         sim.light_state(ix, "cross"))
            _signal_head(surf, cam, ix - 6.1, 6.4, sim.light_state(ix, "cross"))
        else:
            for px, py in [(ix - 5.9, -6.1), (ix + 5.9, 6.1),
                           (ix + 6.1, -5.9), (ix - 6.1, 5.9)]:
                _stop_sign(surf, cam, px, py)

    for veh in sim.vehicles:
        _draw_vehicle(surf, cam, sim, veh)

    _draw_hud(surf, sim, fps)
    surf.set_clip(None)


def _draw_city_world(surf, sim, fps=None):
    """Draw a local north-up window of the generated city around the ego."""
    ego_x, ego_y, _ = sim.ego.pose()
    cam = WorldView(ego_x, ego_y)
    half_w = W / 2.0 / PXM_W
    half_h = WORLD_H / 2.0 / PXM_W
    bounds = (ego_x - half_w, ego_y - half_h,
              ego_x + half_w, ego_y + half_h)

    surf.set_clip(pygame.Rect(0, 0, W, WORLD_H))
    surf.fill(GROUND)

    # Sidewalk slabs sit below asphalt so crossings and corners read clearly.
    for rect in sim.world.sidewalk_rects:
        if _rect_visible(rect, bounds):
            surf.fill(SIDEWALK, cam.rect(*rect))
    for rect in sim.world.road_rects:
        if _rect_visible(rect, bounds):
            surf.fill(ASPHALT, cam.rect(*rect))

    for a, b in sim.world.lane_dashes:
        if _segment_visible(a, b, bounds):
            pygame.draw.line(surf, LANE_YELLOW, cam.pt(*a), cam.pt(*b), 2)
    for rect in sim.world.crosswalk_bars:
        if _rect_visible(rect, bounds):
            surf.fill(MARK_WHITE, cam.rect(*rect))
    for a, b in sim.world.stop_lines:
        if _segment_visible(a, b, bounds):
            pygame.draw.line(surf, MARK_WHITE, cam.pt(*a), cam.pt(*b), 3)
    for a, b, _ in sim.world.curbs:
        if _segment_visible(a, b, bounds):
            pygame.draw.line(surf, CURB, cam.pt(*a), cam.pt(*b), 2)

    for box in sim.world.buildings:
        if not _box_visible(box, bounds):
            continue
        rect = cam.rect(box.cx - box.length / 2.0,
                        box.cy - box.width / 2.0,
                        box.length, box.width)
        surf.fill(BUILDING, rect)
        pygame.draw.rect(surf, BUILDING_EDGE, rect, 2)
        # A small inset roof gives dense blocks some visual separation.
        inner = rect.inflate(-max(4, int(PXM_W)), -max(4, int(PXM_W)))
        if inner.width > 3 and inner.height > 3:
            pygame.draw.rect(surf, _hex("#222e3a"), inner, 1)

    for box in sim.world.trees:
        if _box_visible(box, bounds, margin=2.0):
            px, py = cam.pt(box.cx, box.cy)
            _circle(surf, px, py, 0.95 * PXM_W, TREE)
            gfxdraw.aacircle(surf, int(px), int(py), int(0.95 * PXM_W),
                             TREE_EDGE)
    for box in sim.world.poles:
        if _box_visible(box, bounds, margin=1.0):
            px, py = cam.pt(box.cx, box.cy)
            _circle(surf, px, py, 2.2, _hex("#667180"))
    for i, box in enumerate(sim.world.parked):
        if _box_visible(box, bounds, margin=3.0):
            _draw_parked(surf, cam, box, i)

    for item in sim.world.intersections:
        if not (bounds[0] - 10 < item.x < bounds[2] + 10
                and bounds[1] - 10 < item.y < bounds[3] + 10):
            continue
        if item.kind == "signal":
            main = sim.light_state(item.node, "main")
            cross = sim.light_state(item.node, "cross")
            _signal_head(surf, cam, item.x - 7.0, item.y - 7.0, main)
            _signal_head(surf, cam, item.x + 7.0, item.y + 7.0, main)
            _signal_head(surf, cam, item.x + 7.0, item.y - 7.0, cross)
            _signal_head(surf, cam, item.x - 7.0, item.y + 7.0, cross)
        else:
            for px, py in ((item.x - 6.7, item.y - 6.7),
                           (item.x + 6.7, item.y + 6.7),
                           (item.x + 6.7, item.y - 6.7),
                           (item.x - 6.7, item.y + 6.7)):
                _stop_sign(surf, cam, px, py)

    for veh in sim.vehicles:
        _draw_vehicle(surf, cam, sim, veh)

    _draw_hud(surf, sim, fps)
    _text(surf, "N", (W - 25, 18), _hex("#8fa0b5"), size=13,
          mono=True, anchor="topright")
    pygame.draw.aaline(surf, _hex("#8fa0b5"), (W - 31, 42), (W - 31, 23))
    pygame.draw.aaline(surf, _hex("#8fa0b5"), (W - 31, 23), (W - 35, 29))
    pygame.draw.aaline(surf, _hex("#8fa0b5"), (W - 31, 23), (W - 27, 29))
    surf.set_clip(None)


def _rect_visible(rect, bounds):
    x, y, w, h = rect
    return x + w >= bounds[0] and x <= bounds[2] \
        and y + h >= bounds[1] and y <= bounds[3]


def _segment_visible(a, b, bounds):
    return max(a[0], b[0]) >= bounds[0] and min(a[0], b[0]) <= bounds[2] \
        and max(a[1], b[1]) >= bounds[1] and min(a[1], b[1]) <= bounds[3]


def _box_visible(box, bounds, margin=0.0):
    r = max(box.length, box.width) / 2.0 + margin
    return bounds[0] - r <= box.cx <= bounds[2] + r \
        and bounds[1] - r <= box.cy <= bounds[3] + r


def _draw_parked(surf, cam, box, index):
    colors = ("#536b78", "#806b65", "#616a53", "#77717f", "#8a8174")
    body = _hex(colors[index % len(colors)])
    pts = [cam.pt(*p) for p in box.corners()]
    _poly(surf, pts, body, _shade(body, 0.52))
    c, s = math.cos(box.heading), math.sin(box.heading)
    wcx = box.cx - 0.15 * box.length * c
    wcy = box.cy - 0.15 * box.length * s
    glass = BoxProxy(wcx, wcy, box.heading, box.length * 0.45,
                     box.width * 0.72)
    _poly(surf, [cam.pt(*p) for p in glass.corners()], GLASS)


class BoxProxy:
    """Tiny rectangle helper used only for parked-car window rendering."""

    def __init__(self, cx, cy, heading, length, width):
        self.cx, self.cy = cx, cy
        self.heading, self.length, self.width = heading, length, width

    def corners(self):
        hl, hw = self.length / 2.0, self.width / 2.0
        pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        return _place(pts, self.cx, self.cy, self.heading)


def _crosswalk(surf, cam, ix):
    for sx in (-1, 1):
        x0 = ix + sx * 4.2 + (0 if sx > 0 else -1.6)
        for yy in np.arange(-3.5, 3.51, 1.1):
            surf.fill(MARK_WHITE, cam.rect(x0, yy - 0.28, 1.6, 0.56))
    for sy in (-1, 1):
        y0 = sy * 4.2 + (0 if sy > 0 else -1.6)
        for xx in np.arange(ix - 3.5, ix + 3.51, 1.1):
            surf.fill(MARK_WHITE, cam.rect(xx - 0.28, y0, 0.56, 1.6))


def _stop_lines(surf, cam, ix):
    w = 3
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix - 5.3, -A.ROAD_HALF),
                     cam.pt(ix - 5.3, 0), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix + 5.3, 0),
                     cam.pt(ix + 5.3, A.ROAD_HALF), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix, -5.3),
                     cam.pt(ix + A.ROAD_HALF, -5.3), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix - A.ROAD_HALF, 5.3),
                     cam.pt(ix, 5.3), w)


def _signal_head(surf, cam, x, y, state):
    px, py = cam.pt(x, y)
    box = pygame.Rect(0, 0, int(1.1 * PXM_W), int(3.2 * PXM_W))
    box.center = (int(px), int(py))
    pygame.draw.rect(surf, _hex("#0e1216"), box, border_radius=3)
    pygame.draw.rect(surf, _hex("#39424e"), box, 1, border_radius=3)
    off = {"red": _hex("#341113"), "yellow": _hex("#33290e"),
           "green": _hex("#0e2c19")}
    on = {"red": _hex("#ff3b30"), "yellow": _hex("#ffcc00"),
          "green": _hex("#34d466")}
    for i, name in enumerate(["red", "yellow", "green"]):
        cy = py + (i - 1) * PXM_W
        lit = state == name
        if lit:
            gfxdraw.filled_circle(surf, int(px), int(cy), int(0.75 * PXM_W),
                                  on[name] + (60,))
        _circle(surf, px, cy, 0.34 * PXM_W, on[name] if lit else off[name])


def _stop_sign(surf, cam, x, y):
    px, py = cam.pt(x, y)
    r = 1.05 * PXM_W
    pts = [(px + r * math.cos(a), py + r * math.sin(a))
           for a in np.arange(math.pi / 8, 2 * math.pi, math.pi / 4)]
    _poly(surf, pts, _hex("#c0212e"), (255, 255, 255))
    _text(surf, "STOP", (px, py), (255, 255, 255), size=7, mono=True,
          anchor="center")


def _draw_vehicle(surf, cam, sim, veh):
    x, y, h = veh.pose()
    if (abs(x - cam.ego_x) > W / 2.0 / PXM_W + 8
            or abs(y - cam.ego_y) > WORLD_H / 2.0 / PXM_W + 8):
        return
    L, Wd = veh.length, veh.width
    px, py = cam.pt(x, y)
    color = _hex(veh.color)
    edge = (255, 255, 255) if veh.is_ego else _shade(color, 0.45)

    # Transform: local (m, +y left) -> screen px around (px, py), y flipped.
    def tf(pts_local, fwd=0.0, right=0.0):
        cx, cy = _offset(x, y, h, fwd, right)
        spx, spy = cam.pt(cx, cy)
        pts = _place(pts_local, 0, 0, h) * np.array([1, -1]) * PXM_W
        return pts + np.array([spx, spy])

    _poly(surf, tf(_rounded_rect_pts(L, Wd, 0.5)), color, edge)

    if veh.kind == "truck":
        _poly(surf, tf(_rounded_rect_pts(L * 0.24, Wd * 0.92, 0.35),
                       L * 0.36), _shade(color, 0.7), edge)
        _poly(surf, tf(_rounded_rect_pts(L * 0.62, Wd * 0.98, 0.25),
                       -L * 0.14), _hex("#d5dae2"), _hex("#7d8694"))
    else:
        _poly(surf, tf(_rounded_rect_pts(L * 0.5, Wd * 0.76, 0.4),
                       -L * 0.04), GLASS)

    for side in (-1, 1):
        hx, hy = cam.pt(*_offset(x, y, h, L * 0.46, side * Wd * 0.30))
        _circle(surf, hx, hy, 0.16 * PXM_W, _hex("#f6f0c8"))
    braking = veh.acc < -0.7
    for side in (-1, 1):
        tx, ty = cam.pt(*_offset(x, y, h, -L * 0.46, side * Wd * 0.30))
        if braking:
            gfxdraw.filled_circle(surf, int(tx), int(ty), int(0.5 * PXM_W),
                                  (255, 45, 32, 70))
        _circle(surf, tx, ty, (0.22 if braking else 0.14) * PXM_W,
                _hex("#ff2d20") if braking else _hex("#7a1f1a"))

    if sim.blinking(veh):
        direction = sim.turn_signal(veh) if hasattr(sim, "turn_signal") else "right"
        side = -1.0 if direction == "left" else 1.0
        for fwd in (L * 0.46, -L * 0.46):
            bx, by = cam.pt(*_offset(x, y, h, fwd, side * Wd * 0.42))
            gfxdraw.filled_circle(surf, int(bx), int(by), int(0.5 * PXM_W),
                                  (255, 176, 32, 80))
            _circle(surf, bx, by, 0.22 * PXM_W, _hex("#ffb020"))

    if veh.is_ego:
        _circle(surf, px, py, 0.5 * PXM_W, _hex("#0b2530"))
        gfxdraw.aacircle(surf, int(px), int(py), int(0.5 * PXM_W),
                         _hex("#67e8f9"))
        a = -sim.t * 9.0
        pygame.draw.aaline(surf, _hex("#67e8f9"), (px, py),
                           (px + 0.48 * PXM_W * math.cos(a),
                          py - 0.48 * PXM_W * math.sin(a)))


def _draw_hud(surf, sim, fps=None):
    hud = pygame.Surface((272, 88), pygame.SRCALPHA)
    hud.fill((11, 15, 20, 228))
    pygame.draw.rect(hud, _hex("#273140"), hud.get_rect(), 1, border_radius=6)
    surf.blit(hud, (12, 12))
    _text(surf, "%3.0f km/h" % (sim.ego.v * 3.6), (24, 22), EGO_CYAN,
          size=20, mono=True)
    _text(surf, sim.ego_status, (24, 53), HUD_TEXT, size=13)
    district = getattr(sim, "district_name", "ARTERIAL")
    _text(surf, district, (24, 76), _hex("#5c6b80"), size=10, mono=True)
    _text(surf, "t %5.1fs" % sim.t, (270, 24), _hex("#5c6b80"), size=11,
          mono=True, anchor="topright")
    if fps is not None:
        _text(surf, "%3.0f fps" % fps, (270, 42), _hex("#5c6b80"), size=11,
              mono=True, anchor="topright")
    keys = "space pause   b boxes   r rays   m drive (up/down)"
    _text(surf, keys, (W - 14, WORLD_H - 10), _hex("#4a5766"), size=11,
          anchor="bottomright")


# --- ego LiDAR panel --------------------------------------------------------------
def _ego_to_world_points(points, ego_heading):
    """Rotate ego-frame points into a north-up display frame."""
    points = np.asarray(points, dtype=float)
    c, s = math.cos(ego_heading), math.sin(ego_heading)
    rotation = np.array(((c, -s), (s, c)))
    return points @ rotation.T


def draw_lidar(surf, sim, show_boxes=True, show_rays=False):
    top = WORLD_H + GAP
    cx, cy = W / 2.0, top + LIDAR_H / 2.0

    def lt(x, y):
        return (cx + x * PXM_L, cy - y * PXM_L)

    surf.set_clip(pygame.Rect(0, top, W, LIDAR_H))
    surf.fill(LIDAR_BG)

    # Range rings + crosshair.
    for r in (10, 20, 30, 40):
        gfxdraw.aacircle(surf, int(cx), int(cy), int(r * PXM_L), RING)
        _text(surf, "%dm" % r, (cx + r * PXM_L - 6, cy - 12), RING_TEXT,
              size=10, anchor="topright")
    pygame.draw.line(surf, RING, (0, int(cy)), (W, int(cy)))
    pygame.draw.line(surf, RING, (int(cx), top), (int(cx), top + LIDAR_H))

    scene = sim.ego_scene(max_range=_LIDAR.max_range)
    seg_a, seg_b = scene.segments()
    scan = _LIDAR.scan((0.0, 0.0), seg_a, seg_b,
                       seg_refl=scene.reflectivity())

    _, _, ego_heading = sim.ego.pose()
    pts = _ego_to_world_points(scan.points[scan.hit], ego_heading)
    inten = scan.intensity[scan.hit]
    if show_rays:
        for p in pts:
            pygame.draw.aaline(surf, (13, 58, 52), lt(0, 0), lt(*p))

    ci = np.clip(np.sqrt(inten) * 255, 0, 255).astype(int)
    for (x, y), c in zip(pts, ci):
        px, py = lt(x, y)
        col = tuple(_PLASMA[c])
        gfxdraw.filled_circle(surf, int(px), int(py), 2, col)

    if show_boxes:
        for box in scene.boxes:
            color = BOX_COLORS.get(box.label)
            if color is None:
                continue
            corners = _ego_to_world_points(box.corners(), ego_heading)
            corners = [lt(*p) for p in corners]
            pygame.draw.aalines(surf, color, True, corners)
            tip = (box.cx + 0.5 * box.length * math.cos(box.heading),
                   box.cy + 0.5 * box.length * math.sin(box.heading))
            center, tip = _ego_to_world_points(
                ((box.cx, box.cy), tip), ego_heading)
            pygame.draw.aaline(surf, color, lt(*center), lt(*tip))

    # The panel is north-up, so the ego marker turns instead of the scan.
    wedge = [(2.6, 0.0), (-1.5, 1.15), (-0.8, 0.0), (-1.5, -1.15)]
    wedge = _ego_to_world_points(wedge, ego_heading)
    _poly(surf, [lt(*p) for p in wedge], EGO_CYAN, (255, 255, 255))

    _text(surf, "EGO LIDAR — north-up 2D point cloud, colored by return intensity",
          (14, top + 8), _hex("#8fa0b5"), size=13)
    _text(surf, "N", (W - 24, top + 11), _hex("#8fa0b5"), size=12,
          mono=True, anchor="topright")
    pygame.draw.aaline(surf, _hex("#8fa0b5"),
                       (W - 30, top + 42), (W - 30, top + 26))
    pygame.draw.aalines(surf, _hex("#8fa0b5"), False,
                        ((W - 34, top + 31), (W - 30, top + 26),
                         (W - 26, top + 31)))
    surf.set_clip(None)
