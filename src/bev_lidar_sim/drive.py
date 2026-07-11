"""Live driving simulation (pygame).

    python drive.py                 # live 60 fps window (chase cam + ego LiDAR)
    python drive.py --no-lidar      # just the street view
    python drive.py --save out.gif  # render a few seconds to a GIF (headless)

Keys in the live window:
    space  pause / resume
    b      toggle ground-truth boxes in the LiDAR panel
    r      toggle laser rays in the LiDAR panel
    m      take the wheel (throttle ↑ / brake ↓); the ego keeps lane
    esc/q  quit

The ego car (cyan) loops along the arterial, obeying the signals, the all-way
stop, and the traffic around it — including cross traffic and cars turning on
and off the road at the side streets.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lidar", action="store_true", help="hide LiDAR panel")
    ap.add_argument("--save", metavar="PATH", help="render to a GIF and exit")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="length when --save")
    ap.add_argument("--width", type=int, default=1080,
                    help="GIF width in px when --save")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.save:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    from . import render
    from .sim import Simulator

    pygame.init()
    height = render.HEIGHT if not args.no_lidar else render.WORLD_H
    screen = pygame.display.set_mode((render.W, height))
    pygame.display.set_caption("bev-lidar-sim")
    sim = Simulator(seed=args.seed)

    if args.save:
        _save_gif(pygame, render, sim, screen, args)
        return

    clock = pygame.time.Clock()
    paused, boxes, rays = False, True, False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_b:
                    boxes = not boxes
                elif ev.key == pygame.K_r:
                    rays = not rays
                elif ev.key == pygame.K_m:
                    sim.manual = not sim.manual

        held = pygame.key.get_pressed()
        sim.manual_cmd = (1.0 if held[pygame.K_UP] else 0.0) \
            - (1.0 if held[pygame.K_DOWN] else 0.0)

        if not paused:
            sim.step(1.0 / 60.0)
        render.draw_world(screen, sim, fps=clock.get_fps())
        if not args.no_lidar:
            render.draw_lidar(screen, sim, show_boxes=boxes, show_rays=rays)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def _save_gif(pygame, render, sim, screen, args) -> None:
    """Headless render: 20 GIF fps, 0.05 s of sim time per frame."""
    from PIL import Image

    n = int(args.seconds * 20)
    frames = []
    for i in range(n):
        sim.step(0.05)
        render.draw_world(screen, sim)
        if not args.no_lidar:
            render.draw_lidar(screen, sim)
        arr = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        img = Image.fromarray(arr)
        if args.width and args.width != img.width:
            h = round(img.height * args.width / img.width)
            img = img.resize((args.width, h), Image.LANCZOS)
        frames.append(img)
        if i % 100 == 0:
            print("frame %d/%d" % (i, n), file=sys.stderr)

    # One shared palette (built from sample frames) compresses far better
    # than a palette per frame.
    samples = frames[:: max(len(frames) // 8, 1)]
    strip = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
    for k, f in enumerate(samples):
        strip.paste(f, (0, k * f.height))
    palette = strip.quantize(colors=255)
    frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    frames[0].save(args.save, save_all=True, append_images=frames[1:],
                   duration=50, loop=0, optimize=True)
    print(f"wrote {args.save}")
    pygame.quit()


if __name__ == "__main__":
    main()
