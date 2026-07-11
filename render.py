"""Rendering for the traffic sim: a chase view of the road + the ego's LiDAR."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, RegularPolygon

import sim as S
import viz
from lidar import Lidar2D
from scene import Box, Scene

ROAD_COLOR = "#2b2f3a"
_LIDAR = Lidar2D(n_beams=480, max_range=45.0, noise_std=0.03, dropout=0.01)


def _rect_corners(cx, cy, heading, length, width):
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]])
    c, s = math.cos(heading), math.sin(heading)
    return local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def _draw_vehicle(ax, veh, t, zorder=6):
    x, y, h = S.pose(veh.lane, veh.u)
    corners = _rect_corners(x, y, h, veh.length, veh.width)
    edge = "white" if veh.is_ego else "#0a0f1e"
    lw = 2.0 if veh.is_ego else 1.0
    ax.add_patch(Polygon(corners, closed=True, facecolor=veh.color,
                         edgecolor=edge, linewidth=lw, zorder=zorder))
    # Windshield hint: a small marker at the front.
    fx, fy = x + math.cos(h) * veh.length * 0.35, y + math.sin(h) * veh.length * 0.35
    ax.add_patch(Circle((fx, fy), 0.35, color="#0a0f1e", zorder=zorder + 1))


def _draw_light(ax, sim, lt):
    state = sim.light_state(lt)
    head_x, head_y = lt.x, S.LANE_HALF + 2.2
    ax.add_patch(Rectangle((head_x - 0.5, head_y - 1.8), 1.0, 3.6,
                           facecolor="#111", edgecolor="#333", zorder=7))
    colors = {"red": "#3a0d0d", "yellow": "#3a340d", "green": "#0d3a1a"}
    on = {"red": "#ff3b30", "yellow": "#ffcc00", "green": "#2fe36b"}
    for i, name in enumerate(["red", "yellow", "green"]):
        cy = head_y + 1.0 - i * 1.0
        c = on[name] if state == name else colors[name]
        ax.add_patch(Circle((head_x, cy), 0.35, color=c, zorder=8))


def _draw_stop_sign(ax, x):
    sx, sy = x - S.INTER_HALF, S.LANE_E_Y - S.LANE_HALF + 1.0
    ax.add_patch(RegularPolygon((sx, sy), numVertices=8, radius=1.2,
                                orientation=math.pi / 8, facecolor="#d0021b",
                                edgecolor="white", linewidth=1.2, zorder=7))
    ax.text(sx, sy, "STOP", color="white", ha="center", va="center",
            fontsize=5, fontweight="bold", zorder=8)


def draw_world(ax, sim):
    ax.clear()
    ego_x = S.pose(sim.ego.lane, sim.ego.u)[0]
    ax.set_facecolor("#0a0f1e")
    ax.set_xlim(ego_x - 70, ego_x + 70)
    ax.set_ylim(-18, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    # Road surface + cross streets + lane markings.
    ax.add_patch(Rectangle((ego_x - 80, -S.LANE_HALF), 160, 2 * S.LANE_HALF,
                           facecolor=ROAD_COLOR, zorder=1))
    for ix in S.INTERSECTIONS:
        ax.add_patch(Rectangle((ix - S.INTER_HALF, -16), 2 * S.INTER_HALF, 32,
                               facecolor=ROAD_COLOR, zorder=1))
    for xd in np.arange(ego_x - 80, ego_x + 80, 6):
        ax.plot([xd, xd + 3], [0, 0], color="#c9b21a", lw=1.2, zorder=2)

    for lt in sim.lights:
        _draw_light(ax, sim, lt)
    for x in S.STOP_SIGN_XS:
        _draw_stop_sign(ax, x)

    for veh in sim.vehicles:
        _draw_vehicle(ax, veh, sim.t)
        if sim.blinking(veh):
            x, y, h = S.pose(veh.lane, veh.u)
            ax.add_patch(Circle((x, y), 0.6, color="#ffae00", alpha=0.9, zorder=9))

    ax.text(ego_x - 68, 15.5, f"ego  {sim.ego.v * 3.6:4.0f} km/h",
            color="#19d3ff", fontsize=11, fontweight="bold", zorder=10)
    ax.text(ego_x - 68, 13.2, f"t = {sim.t:5.1f} s", color="#8a93a6",
            fontsize=9, zorder=10)


def _ego_scene(sim) -> Scene:
    """Build the ego-centric scene the LiDAR sees (ego at origin, facing +x)."""
    ex, ey, eh = S.pose(sim.ego.lane, sim.ego.u)
    scene = Scene()
    # Road curbs relative to ego.
    for edge in (-S.LANE_HALF, S.LANE_HALF):
        scene.add_wall((-45, edge - ey), (45, edge - ey))
    for veh in sim.vehicles:
        if veh.is_ego:
            continue
        vx, vy, vh = S.pose(veh.lane, veh.u)
        if abs(vx - ex) > 45:
            continue
        scene.boxes.append(Box(cx=vx - ex, cy=vy - ey, heading=vh - eh,
                               length=veh.length, width=veh.width, label="car"))
    return scene


def draw_lidar(ax, sim):
    scene = _ego_scene(sim)
    seg_a, seg_b = scene.segments()
    scan = _LIDAR.scan((0.0, 0.0), seg_a, seg_b)
    viz.plot_bev(ax, scene, scan, extent=45.0, yextent=11.0)
    ax.set_title("ego LiDAR  (bird's-eye view — what the car senses)",
                 color="white", fontsize=10, pad=6)
