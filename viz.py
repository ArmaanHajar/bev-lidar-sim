"""Matplotlib bird's-eye-view rendering of a scene + LiDAR scan."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from lidar import ScanResult
from scene import Scene

# Class colors for ground-truth boxes.
BOX_COLORS = {"car": "#ffb000", "truck": "#ff3b30"}


def _style_axes(ax, xext: float, yext: float) -> None:
    ax.set_facecolor("#0a0f1e")
    ax.set_xlim(-xext, xext)
    ax.set_ylim(-yext, yext)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_ego(ax) -> None:
    """Draw the ego vehicle as a small rectangle at the origin, facing +x."""
    hl, hw = 2.3, 1.0
    corners = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]])
    ax.add_patch(Polygon(corners, closed=True, facecolor="#7CFC00",
                         edgecolor="white", linewidth=1.0, zorder=5))


def draw_boxes(ax, scene: Scene) -> None:
    for box in scene.boxes:
        color = BOX_COLORS.get(box.label, "#ffb000")
        ax.add_patch(Polygon(box.corners(), closed=True, fill=False,
                             edgecolor=color, linewidth=1.8, zorder=4))


def plot_bev(ax, scene: Scene, scan: ScanResult, extent: float = 50.0,
             yextent: float | None = None,
             show_boxes: bool = True, show_rays: bool = False) -> None:
    """Render one BEV frame onto `ax`.

    `extent` is the half-width in x; `yextent` the half-height in y (defaults
    to `extent` for a square view). Use a smaller `yextent` for a wide strip.
    """
    ax.clear()
    _style_axes(ax, extent, yextent if yextent is not None else extent)

    origin = scan.points[0] - scan.ranges[0] * np.array(
        [np.cos(scan.angles[0]), np.sin(scan.angles[0])]
    )

    if show_rays:
        for p in scan.points[scan.hit]:
            ax.plot([origin[0], p[0]], [origin[1], p[1]],
                    color="#1de9b6", linewidth=0.2, alpha=0.15, zorder=1)

    pts = scan.points[scan.hit]
    if len(pts):
        rng = scan.ranges[scan.hit]
        ax.scatter(pts[:, 0], pts[:, 1], c=rng, cmap="viridis_r", s=6,
                   vmin=0, vmax=extent, zorder=3)

    if show_boxes:
        draw_boxes(ax, scene)
    draw_ego(ax)


def new_figure(size: float = 8.0):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("#0a0f1e")
    return fig, ax
