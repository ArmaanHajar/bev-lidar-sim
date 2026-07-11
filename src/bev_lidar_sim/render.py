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

from . import sim as S
from .lidar import Lidar2D


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
WORLD_H = 368            # chase-view panel height (px)
LIDAR_H = 400            # LiDAR panel height (px)
GAP = 12
HEIGHT = WORLD_H + GAP + LIDAR_H
PXM_W = 9.2              # chase view px per meter (view ~139 m x 40 m)
PXM_L = 13.2             # LiDAR view px per meter (~97 m x 30 m)

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

    def __init__(self, ego_x):
        self.ego_x = ego_x

    def pt(self, x, y):
        return (W / 2.0 + (x - self.ego_x) * PXM_W,
                WORLD_H / 2.0 - y * PXM_W)

    def rect(self, x, y, w, h):
        """World-space rect (x, y = lower-left corner) -> pygame Rect."""
        px, py = self.pt(x, y + h)
        return pygame.Rect(int(px), int(py),
                           max(int(w * PXM_W), 1), max(int(h * PXM_W), 1))


def draw_world(surf, sim, fps=None):
    ego_x = sim.ego.pose()[0]
    cam = WorldView(ego_x)
    half_m = W / 2.0 / PXM_W
    x_lo, x_hi = ego_x - half_m, ego_x + half_m
    y_lim = WORLD_H / 2.0 / PXM_W

    surf.set_clip(pygame.Rect(0, 0, W, WORLD_H))
    surf.fill(GROUND)
    visible = [ix for ix in S.INTERSECTIONS if abs(ix - ego_x) < half_m + 20]

    # Sidewalks, then asphalt over them.
    for sy in (1, -1):
        surf.fill(SIDEWALK, cam.rect(x_lo, min(sy * 4, sy * 7), 2 * half_m, 3))
    for ix in visible:
        for sx in (-7, 4):
            surf.fill(SIDEWALK, cam.rect(ix + sx, -y_lim, 3, 2 * y_lim))
    surf.fill(ASPHALT, cam.rect(x_lo, -S.ROAD_HALF, 2 * half_m,
                                2 * S.ROAD_HALF))
    for ix in visible:
        surf.fill(ASPHALT, cam.rect(ix - S.ROAD_HALF, -y_lim,
                                    2 * S.ROAD_HALF, 2 * y_lim))

    # Lane paint.
    for xd in np.arange(math.floor(x_lo / 6.0) * 6.0, x_hi, 6.0):
        if any(abs(xd + 1.5 - ix) < 7.5 for ix in S.INTERSECTIONS):
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
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix - 5.3, -S.ROAD_HALF),
                     cam.pt(ix - 5.3, 0), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix + 5.3, 0),
                     cam.pt(ix + 5.3, S.ROAD_HALF), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix, -5.3),
                     cam.pt(ix + S.ROAD_HALF, -5.3), w)
    pygame.draw.line(surf, MARK_WHITE, cam.pt(ix - S.ROAD_HALF, 5.3),
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
    if abs(x - cam.ego_x) > W / 2.0 / PXM_W + 8:
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
        for fwd in (L * 0.46, -L * 0.46):
            bx, by = cam.pt(*_offset(x, y, h, fwd, Wd * 0.42))
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
    hud = pygame.Surface((250, 74), pygame.SRCALPHA)
    hud.fill((11, 15, 20, 228))
    pygame.draw.rect(hud, _hex("#273140"), hud.get_rect(), 1, border_radius=6)
    surf.blit(hud, (12, 12))
    _text(surf, "%3.0f km/h" % (sim.ego.v * 3.6), (24, 22), EGO_CYAN,
          size=20, mono=True)
    _text(surf, sim.ego_status, (24, 58), HUD_TEXT, size=13)
    _text(surf, "t %5.1fs" % sim.t, (250, 24), _hex("#5c6b80"), size=11,
          mono=True, anchor="topright")
    if fps is not None:
        _text(surf, "%3.0f fps" % fps, (250, 42), _hex("#5c6b80"), size=11,
              mono=True, anchor="topright")
    keys = "space pause   b boxes   r rays   m drive (up/down)"
    _text(surf, keys, (W - 14, WORLD_H - 10), _hex("#4a5766"), size=11,
          anchor="bottomright")


# --- ego LiDAR panel --------------------------------------------------------------
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

    pts = scan.points[scan.hit]
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
            corners = [lt(*p) for p in box.corners()]
            pygame.draw.aalines(surf, color, True, corners)
            tip = (box.cx + 0.5 * box.length * math.cos(box.heading),
                   box.cy + 0.5 * box.length * math.sin(box.heading))
            pygame.draw.aaline(surf, color, lt(box.cx, box.cy), lt(*tip))

    # Ego wedge.
    wedge = [(2.6, 0.0), (-1.5, 1.15), (-0.8, 0.0), (-1.5, -1.15)]
    _poly(surf, [lt(*p) for p in wedge], EGO_CYAN, (255, 255, 255))

    _text(surf, "EGO LIDAR — live 2D point cloud, colored by return intensity",
          (14, top + 8), _hex("#8fa0b5"), size=13)
    surf.set_clip(None)
