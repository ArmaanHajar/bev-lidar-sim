"""Matplotlib bird's-eye-view rendering of a scene + LiDAR scan."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from ..sensors.lidar import ScanResult
from ..sensors.scene import Scene

BG = "#06090f"
RING = "#182233"
RING_TEXT = "#3d4c60"
BOX_COLORS = {"car": "#f0b429", "truck": "#ff6b5e"}


def _style_axes(ax, xext: float, yext: float) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(-xext, xext)
    ax.set_ylim(-yext, yext)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1a2330")
        spine.set_linewidth(0.8)


def draw_rings(ax, max_r: float = 40.0, step: float = 10.0) -> None:
    """Faint range rings + crosshair, radar style."""
    for r in np.arange(step, max_r + 0.1, step):
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=RING,
                            linewidth=0.8, zorder=1))
        ax.text(r - 0.6, 0.7, "%dm" % r, color=RING_TEXT, fontsize=6,
                ha="right", zorder=1)
    ax.axhline(0, color=RING, linewidth=0.6, zorder=1)
    ax.axvline(0, color=RING, linewidth=0.6, zorder=1)


def draw_ego(ax) -> None:
    """Sensor marker at the origin: a wedge pointing +x."""
    wedge = np.array([[2.6, 0.0], [-1.5, 1.15], [-0.8, 0.0], [-1.5, -1.15]])
    ax.add_patch(Polygon(wedge, closed=True, facecolor="#22d3ee",
                         edgecolor="white", linewidth=0.8, zorder=6))


def draw_boxes(ax, scene: Scene) -> None:
    """Ground-truth outlines for the dynamic objects (cars / trucks)."""
    for box in scene.boxes:
        color = BOX_COLORS.get(box.label)
        if color is None:
            continue
        ax.add_patch(Polygon(box.corners(), closed=True, fill=False,
                             edgecolor=color, linewidth=1.1, alpha=0.9,
                             zorder=4))
        tip = (box.cx + 0.5 * box.length * np.cos(box.heading),
               box.cy + 0.5 * box.length * np.sin(box.heading))
        ax.plot([box.cx, tip[0]], [box.cy, tip[1]], color=color,
                linewidth=0.9, alpha=0.7, zorder=4)


def plot_bev(ax, scene: Scene, scan: ScanResult, extent: float = 50.0,
             yextent: float = None, show_boxes: bool = True,
             show_rays: bool = False, rings: bool = True) -> None:
    """Render one BEV frame onto `ax`.

    `extent` is the half-width in x; `yextent` the half-height in y (defaults
    to `extent` for a square view). Points are colored by return intensity
    when the scan has one, else by range.
    """
    ax.clear()
    _style_axes(ax, extent, yextent if yextent is not None else extent)
    if rings:
        draw_rings(ax, max_r=min(extent, 40.0))

    origin = scan.points[0] - scan.ranges[0] * np.array(
        [np.cos(scan.angles[0]), np.sin(scan.angles[0])]
    )

    if show_rays:
        for p in scan.points[scan.hit]:
            ax.plot([origin[0], p[0]], [origin[1], p[1]],
                    color="#14b8a6", linewidth=0.25, alpha=0.14, zorder=2)

    pts = scan.points[scan.hit]
    if len(pts):
        if scan.intensity is not None:
            c = np.sqrt(scan.intensity[scan.hit])   # gamma-lift dim returns
            ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap="plasma", s=5.5,
                       vmin=0.05, vmax=0.95, zorder=3)
        else:
            rng = scan.ranges[scan.hit]
            ax.scatter(pts[:, 0], pts[:, 1], c=rng, cmap="viridis_r", s=5,
                       vmin=0, vmax=extent, zorder=3)

    if show_boxes:
        draw_boxes(ax, scene)
    draw_ego(ax)


def new_figure(size: float = 8.0):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor(BG)
    return fig, ax
