"""Live driving simulation.

    python drive.py                 # open a live window (chase cam + ego LiDAR)
    python drive.py --no-lidar      # just the road view
    python drive.py --save out.gif  # render a few seconds to a GIF (headless)

The ego car (cyan) drives the road, obeys traffic lights and the stop sign,
follows the car ahead, and reacts to vehicles entering/leaving the road.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

import render
from sim import Simulator

DT = 0.05          # physics step (s)
FPS = 20


def build_figure(show_lidar: bool):
    if show_lidar:
        fig = plt.figure(figsize=(13, 7))
        gs = GridSpec(2, 1, height_ratios=[3, 1.4], hspace=0.12)
        ax_world = fig.add_subplot(gs[0])
        ax_lidar = fig.add_subplot(gs[1])
    else:
        fig, ax_world = plt.subplots(figsize=(13, 5))
        ax_lidar = None
    fig.patch.set_facecolor("#0a0f1e")
    return fig, ax_world, ax_lidar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lidar", action="store_true", help="hide LiDAR panel")
    ap.add_argument("--save", metavar="PATH", help="render to a GIF and exit")
    ap.add_argument("--seconds", type=float, default=8.0, help="length when --save")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    show_lidar = not args.no_lidar
    sim = Simulator(seed=args.seed)
    fig, ax_world, ax_lidar = build_figure(show_lidar)

    def update(_frame):
        sim.step(DT)
        render.draw_world(ax_world, sim)
        if show_lidar:
            render.draw_lidar(ax_lidar, sim)

    if args.save:
        n = int(args.seconds * FPS)
        anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS)
        anim.save(args.save, writer=PillowWriter(fps=FPS),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        print(f"wrote {args.save}")
        return

    anim = FuncAnimation(fig, update, interval=1000 / FPS, cache_frame_data=False)
    fig._anim = anim  # keep a reference alive
    plt.show()


if __name__ == "__main__":
    main()
