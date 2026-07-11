# BEV LiDAR + Traffic Simulator

A from-scratch autonomous-driving sandbox in two parts:

1. **A 2D bird's-eye-view LiDAR sensor** — a spinning sensor fires rays in a
   full 360° circle against a scene of 2D polygons; each ray stops at the first
   surface it hits, so occlusion and shadows fall out for free. Every return
   also carries an **intensity** (surface reflectivity attenuated with range),
   so stop signs glow and concrete curbs stay dim, just like real LiDAR.
2. **A path-based traffic microsimulator** — a small urban arterial with two
   signalized intersections and an all-way stop, crossed by real side streets
   with real cross traffic. Cars follow the Intelligent Driver Model, obey
   signal phases, resolve the 4-way stop in arrival order, and turn on and off
   the arterial along real turn arcs. The ego's live LiDAR view is rendered
   below the street so you can see what the car senses.

![driving demo](assets/drive_demo.gif)

## Run the driving sim

```bash
python drive.py              # live 60 fps window: chase cam + ego LiDAR
python drive.py --no-lidar   # street view only
python drive.py --save out.gif --seconds 20   # headless render to a GIF
```

After installing the package, the equivalent console command is `bev-drive`.

The live window runs on **pygame at 60 fps**; GIF export renders headless
(no window) through the exact same code path and takes seconds.

Keys in the live window: `space` pause · `b` toggle ground-truth boxes ·
`r` toggle laser rays · `m` take the wheel (throttle `↑` / brake `↓`;
the ego steers itself along its lane) · `esc` quit.

## Run just the LiDAR sensor

![example frame](assets/bev_frame.png)

```bash
python demo.py              # single frame  -> outputs/bev_frame.png
python demo.py --rays       # single frame with laser rays drawn
python demo.py --animate    # moving scene  -> outputs/bev_scan.gif
```

After installing the package, the equivalent console command is `bev-demo`.

## Why this is relevant to autonomous driving

Bird's-eye view is the working representation of modern AV perception:
detectors like PointPillars rasterize LiDAR into a top-down grid before running
a CNN. This project builds that world from the sensor up — and because the
simulator knows the true pose of every object, every frame comes with free
ground-truth boxes for training.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Files

| Path | Role |
|------|------|
| `src/bev_lidar_sim/scene.py` | `Box` (oriented rectangle) + `Scene`; everything becomes line segments with per-surface reflectivity |
| `src/bev_lidar_sim/lidar.py` | `Lidar2D` — vectorized ray/segment intersection, nearest hit, range noise, dropout, intensity channel |
| `src/bev_lidar_sim/viz.py` | Matplotlib BEV rendering for stills: range rings, intensity-colored points, GT boxes |
| `src/bev_lidar_sim/sim.py` | Traffic microsim: path network, IDM car-following, signal phases, all-way stop arbitration, turns, spawning; also generates the static world shared by renderer and sensor |
| `src/bev_lidar_sim/render.py` | Pygame chase-cam street view + ego LiDAR panel |
| `src/bev_lidar_sim/drive.py` | Driving simulation CLI implementation |
| `src/bev_lidar_sim/demo.py` | Static LiDAR demo CLI implementation |
| `drive.py`, `demo.py` | Thin root wrappers for the old commands |
| `assets/` | README/demo images and GIFs |

## How the driving behaves

Every lane, side street, and turn is a **path** (a polyline with arc-length
lookup), so all driving logic is 1-D: a vehicle is `(path, u, v)` and only
rendering and sensing care about x/y.

Vehicles drive with the **Intelligent Driver Model (IDM)**, the standard
car-following model: accelerate toward a desired speed, brake to keep a safe,
speed-dependent gap to whatever is ahead — where "ahead" looks across path
transitions, so a car follows smoothly through a turn. Traffic controls are
handled the way real microsimulators do it: a red light or an un-granted stop
line becomes a *virtual stopped car* at the line, so the same IDM code that
follows the lead vehicle also produces smooth stops at signals.

On top of that:

- **Signal phases**: main green → yellow → all-red → cross green → …, so the
  side streets actually flow while the arterial waits.
- **Yellow-light dilemma**: a car runs the yellow only when it can no longer
  stop comfortably before the line.
- **All-way stop**: come to a full stop to join the queue; proceed when the
  intersection box is clear and it's your turn (first-come, first-served).
- **Turns**: cars enter and leave the arterial along real right-turn arcs,
  slowing to a comfortable turning speed, with turn signals and brake lights.

## How the sensor works

Each beam is a ray `O + t·d`. For a segment `A → B` we solve for the ray
parameter `t` and segment parameter `u` with 2D cross products:

```
t = (A − O) × e / (d × e)      u = (A − O) × d / (d × e)      e = B − A
```

A hit is valid when `t > 0` and `0 ≤ u ≤ 1`; the smallest valid `t` across all
segments is the measured range. All beams × all segments are computed in one
broadcast, so a full sweep is a couple of numpy operations. The winning
segment's reflectivity, attenuated by `exp(-r/55)` plus noise, becomes the
return intensity.

## Ideas to extend

- Rasterize hits into a **BEV occupancy grid** (the input format a BEV detector eats)
- Train a small **detector/segmenter** on the free ground-truth boxes (sim-to-real)
- Drive the **ego along a path** and accumulate scans into a map (mini-SLAM)
- Replace the ego's scripted IDM with a **learned or planned controller** that
  drives from the LiDAR scan alone
