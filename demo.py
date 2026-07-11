"""Run the 2D BEV LiDAR simulator.

    python demo.py              # render a single frame -> bev_frame.png
    python demo.py --animate    # render a moving scene   -> bev_scan.gif
    python demo.py --rays       # also draw the laser rays (nice for a still)
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import viz
from lidar import Lidar2D
from scene import Box, Scene


def scene_at(frame: int, n_frames: int) -> Scene:
    """Build the scene for a given animation frame.

    Ego stays at the origin. An oncoming car drives toward it, a truck
    pulls away, and two vehicles stay parked.
    """
    scene = Scene()
    road_half = 6.0
    scene.add_wall((-25.0, +road_half), (45.0, +road_half))
    scene.add_wall((-25.0, -road_half), (45.0, -road_half))
    scene.add_wall((10.0, -road_half - 2.0), (30.0, -road_half - 2.0))

    p = frame / max(n_frames - 1, 1)  # 0 -> 1

    oncoming_x = 40.0 - 45.0 * p      # drives from far ahead toward ego
    truck_x = 18.0 + 20.0 * p         # pulls away down the road
    scene.boxes.extend([
        Box(oncoming_x, -2.5, np.pi, 4.5, 2.0, "car"),   # oncoming, left lane
        Box(truck_x, 2.2, 0.05, 8.0, 2.4, "truck"),      # leading truck
        Box(12.0, 2.6, 0.0, 4.5, 2.0, "car"),            # parked ahead-right
        Box(6.0, -3.2, np.pi, 4.2, 1.9, "car"),          # parked behind-left
    ])
    return scene


def render_still(rays: bool) -> None:
    lidar = Lidar2D(n_beams=1080, max_range=50.0)
    scene = scene_at(0, 1)
    seg_a, seg_b = scene.segments()
    scan = lidar.scan((0.0, 0.0), seg_a, seg_b)

    fig, ax = viz.new_figure()
    viz.plot_bev(ax, scene, scan, extent=50.0, show_rays=rays)
    ax.set_title("2D BEV LiDAR scan", color="white", fontsize=14, pad=12)
    fig.savefig("bev_frame.png", dpi=130, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print("wrote bev_frame.png")


def render_animation(rays: bool, n_frames: int = 90) -> None:
    lidar = Lidar2D(n_beams=1080, max_range=50.0)
    fig, ax = viz.new_figure()

    def update(frame: int):
        scene = scene_at(frame, n_frames)
        seg_a, seg_b = scene.segments()
        scan = lidar.scan((0.0, 0.0), seg_a, seg_b)
        viz.plot_bev(ax, scene, scan, extent=50.0, show_rays=rays)
        ax.set_title("2D BEV LiDAR scan", color="white", fontsize=14, pad=12)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
    anim.save("bev_scan.gif", writer=PillowWriter(fps=20),
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    print("wrote bev_scan.gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animate", action="store_true", help="render a GIF")
    ap.add_argument("--rays", action="store_true", help="draw laser rays")
    args = ap.parse_args()

    if args.animate:
        render_animation(args.rays)
    else:
        render_still(args.rays)


if __name__ == "__main__":
    main()
