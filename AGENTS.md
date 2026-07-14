# bev-lidar-sim

Operational guide for AI agents (and new contributors) working in this repo.
The README explains *what* the project is; this file explains *how to work on
it without breaking it*.

## What this is

A from-scratch 2D autonomous-driving sandbox: a simulated spinning LiDAR
(vectorized ray casting with occlusion, range noise, dropout, and
reflectivity-based intensity) senses a live traffic microsimulation. The
default scenario is a seeded 4×4 intersection city with lane-graph routes,
generated blocks, signals, all-way stops, parked vehicles, and smooth turns;
the original arterial remains available with `--scenario arterial`. Runs as a
60 fps pygame app and exports GIFs headless through the same render path.

It is a **research-oriented sandbox and portfolio project**: reproducibility,
clear experimental boundaries, visual quality, and readable code all matter.
The intended roadmap is perception and planning work built on top of the sim
(temporal occupancy, tracking, stress testing, federated learning, and learned
ego control), so keep the sensor/sim layers clean and renderer-agnostic.

## Layout and dependency direction

```
src/bev_lidar_sim/          (five layers; __init__ re-exports the headless API)
  sensors/      Sensor layer. Pure numpy.
    scene.py      Box + Scene primitives; everything reduces to line
                  segments with per-segment reflectivity. No sim knowledge.
    lidar.py      Lidar2D sensor: one broadcasted ray x segment intersection
                  per sweep; returns ranges, hit mask, intensity, per-beam
                  segment index. Takes raw arrays; independent of scene.py.
  maps/         Map layer. Pure numpy + stdlib; no engine imports.
    roadgraph.py  Neutral map schema: RoadNode / LaneDef / ConnectorDef /
                  RoadGraph, with validate() and JSON round-trip. Map
                  *sources* (sim/city/graph.py, future netconvert/WOMD
                  importers — put importers in this package) emit a
                  RoadGraph; the simulator consumes one. This is the seam
                  for real-map import.
  sim/          Simulation layer.
    traffic.py    Shared machinery: Path, Vehicle, IDM, Signal, AllWayStop,
                  REFL/colors, and the scenario-independent `Simulator` core
                  (car-following, control stops, step, ego_scene).
                  Subclasses supply network, world, spawning, _transitions.
    arterial.py   Legacy arterial scenario: WorldGeometry + ArterialSimulator
                  (`--scenario arterial`). Keeps the turn_at_u mechanism.
    city/         Default generated city.
      graph.py      Grid layout as pure map data: build_city_roadgraph()
                    (no RNG — identical across processes). City geometry
                    and routing constants live at the top of this file.
      world.py      CityWorld static render/LiDAR geometry (seeded blocks,
                    trees, parked cars, paint). Still assumes the grid
                    layout; generalizing it to imported RoadGraphs is the
                    planned next step for map import.
      simulator.py  CitySimulator: builds signals/stops/Paths from any
                    RoadGraph (`graph=` kwarg), BFS boundary routing,
                    spawning.
  render/       Rendering layer (keep live and stills separate).
    live.py       Pygame renderer: ego-following city/arterial view + LiDAR
                  panel. Depends on sim.arterial (constants), sensors.lidar.
                  The ONLY module (plus cli/drive.py) that may import pygame.
    stills.py     Matplotlib renderer for publication-style stills/GIFs of
                  the raw sensor (used by cli/demo.py).
  cli/          Entry points (the console scripts in pyproject point here).
    drive.py      Driving CLI: live 60 fps window or headless GIF export.
    demo.py       Static-scene sensor CLI: bev_frame.png / bev_scan.gif.
drive.py, demo.py (repo root)
              Thin sys.path wrappers so `python drive.py` works uninstalled.
              Keep them 10-ish lines; all logic lives in the package.
assets/       Committed media referenced by README (drive_demo.gif, etc.).
outputs/      Gitignored scratch output dir (demo.py default target).
tests/        Built-in unittest coverage: city graph/connectors/determinism,
              roadgraph schema round-trip + graph-driven sim equivalence,
              arterial smoke, local LiDAR geometry. Tests import from the
              top-level `bev_lidar_sim` API.
```

Import relationships (`A -> B` means A imports B):

```
maps.roadgraph     -> (numpy/stdlib only)
sensors.*          -> (numpy only)
sim.traffic        -> sensors.scene
sim.arterial       -> sim.traffic, sensors.scene
sim.city.graph     -> maps.roadgraph
sim.city.world     -> sim.traffic, sensors.scene, sim.city.graph
sim.city.simulator -> maps.roadgraph, sim.traffic, sim.city.graph/world
render.live        -> sim.arterial, sensors.lidar
render.stills      -> sensors
cli.drive          -> sim, render
cli.demo           -> sensors, render.stills
```

Layer rule: sensors and maps import nothing from the package; sim imports
sensors (for shared Box/Scene geometry) and maps; render imports sim and
sensors; cli imports everything. Never point an arrow backwards.
`maps/roadgraph.py` must stay pure data — no traffic/render/sensor imports —
so importers and saved JSON scenarios stay decoupled from the engine. Never
import pygame or matplotlib into `sensors/`, `maps/`, or `sim/` (exception:
`render/live.py` builds its point-color LUT from matplotlib's plasma
colormap — that stays in render). This separation is what makes headless
RL / training loops possible later.

The intended flow for new road environments: emit a `RoadGraph` (from a
generator or importer), validate it, and pass it to
`CitySimulator(seed, graph=...)`. `RoadGraph.save_json/load_json` round-trips
scenarios as plain files.

## Environment

- Venv at `.venv/` on **Python 3.13** (Homebrew). `pyproject.toml` requires
  `>=3.10`. Do not recreate the venv with the Xcode/system Python (3.9) —
  that's what it was accidentally on before.
- The package is currently **not** pip-installed into the venv; the root
  wrappers make everything work anyway. `pip install -e .` additionally
  provides the `bev-drive` / `bev-demo` console scripts. Either path is fine;
  don't assume the editable install exists.
- Dependencies: numpy, matplotlib, pillow, pygame — feel free to add new deps, 
  but ask the user before doing so.
- Run everything with `.venv/bin/python`; there are many other Pythons on
  this machine (anaconda, python.org, Homebrew) and the wrong one will miss
  the deps.

## Running and rendering

```bash
.venv/bin/python drive.py                      # live 60 fps window (needs a display)
.venv/bin/python drive.py --save out.gif --seconds 24 --seed 1
.venv/bin/python drive.py --scenario arterial  # legacy straight-road scenario
.venv/bin/python demo.py [--rays|--animate]    # stills -> outputs/
.venv/bin/python -m unittest discover -s tests -v
```

- `--save` sets `SDL_VIDEODRIVER=dummy` itself; GIF export is fully headless
  and CI/agent-safe. The live window obviously isn't — never try to "test"
  it by launching it from an agent session; render frames to PNG instead
  (see verification below).
- GIF timeline is fixed: 20 fps, `sim.step(0.05)` per frame — one GIF second
  equals one sim second. The live loop steps `1/60` per rendered frame.
- Live keys: `space` pause, `b` GT boxes, `r` rays, `m` manual ego
  (throttle ↑ / brake ↓), `esc` quit.
- README media lives in `assets/` (committed). Regenerate the hero GIF with:
  `.venv/bin/python drive.py --save assets/drive_demo.gif --seconds 24 --seed 1`
  — takes ~20 s, expect ~10–12 MB at the default `--width 1080`. If a change
  affects visuals, regenerate the assets so the README doesn't lie.

## Verification playbook

Run the fast unit tests first, then verify simulation behavior and visuals.
Before claiming a sim/render change works, do all four:

1. **Unit tests** — run `.venv/bin/python -m unittest discover -s tests -v`.
   Add focused tests when changing route connectivity, connector geometry,
   determinism, or local LiDAR scene construction.
2. **Traffic audit** — run the sim headless for 180 sim-seconds across a few
   seeds and assert the invariants:
   - zero oriented-box vehicle overlaps (center distance alone misses
     perpendicular conflicts inside city intersections),
   - no vehicle stalled (< 0.2 m/s) for more than ~40 s continuously
     (normal worst-case queue wait is ~10–20 s),
   - vehicle count stays bounded (~27 max with default spawn caps),
   - the all-way-stop manager's queue drains (granted vehicle changes).
   Write this as a throwaway script (pattern: step 0.05, track pairwise
   distances + per-vid stall clocks). A multi-minute jam almost always means
   a control-registration or leader-gap regression, not a rendering issue.
3. **Frame inspection** — render 3–4 frames at different sim times to PNG
   with `SDL_VIDEODRIVER=dummy` (`pygame.image.save(screen, ...)`) and
   actually look at them: signal states consistent with moving traffic,
   stop lines/crosswalks aligned, cars on lane centerlines, LiDAR panel
   points landing on the drawn geometry.
4. **Perf check** — time `sim.step` + `draw_world` + `draw_lidar`. A warmed-up
   city frame is currently ≈ 2.3 ms. A frame over ~16 ms breaks the 60 fps
   promise; that is a regression even if output is correct.

Determinism: `CitySimulator(seed=N)` / `ArterialSimulator(seed=N)`
deterministically seed the traffic and world. Since the RoadGraph refactor
the city lane order is construction-ordered (no set iteration), so city
traffic is reproducible across processes too — historical runs from before
the refactor used a different (hash-dependent) lane order and won't replay
bit-identically. `Lidar2D` creates its own unseeded `default_rng` unless one
is supplied, and the renderer's module-level LiDAR currently uses that
default. Traffic audits are reproducible per simulator seed, but scan noise
and complete GIF bytes are not guaranteed reproducible unless the sensor RNG
is explicitly seeded. Seed 1 is the canonical traffic/world demo seed.

## Domain invariants and known traps

These encode real bugs that were already found and fixed — do not reintroduce:

- **LiDAR data stays in the ego frame; only the live display is north-up.**
  `Simulator.ego_scene()` and `Lidar2D.scan()` return ego/sensor-relative
  geometry for research and training. `render.live.draw_lidar()` rotates a
  copy of points and boxes by the ego heading so the lower panel matches the
  fixed-world orientation of the upper panel. Do not rotate sensor outputs to
  implement display behavior.

- **All driving logic is 1-D.** A vehicle is `(path, u, v)`; x/y/heading
  exist only via `Path.pose(u)` for rendering, sensing, and the intersection
  box checks. Never write behavior code against x/y.
- **Controls are virtual stopped cars.** Red lights and un-granted stop lines
  feed the same IDM as a lead vehicle (`gap = dist_to_line - length/2`,
  closing speed = own speed). Keep new control types in this pattern.
- **Stop-line registration must use bumper gap, not raw distance.** IDM
  equilibrium leaves the *bumper* ~2 m from the line, so the *center*
  distance scales with vehicle length. The registration window
  (`dist - length/2 < 3.6 and v < 0.5` in `_control_gap`) was once a raw
  `dist < 5.5` — trucks never registered at the all-way stop and traffic
  deadlocked permanently. Any tightening here must be re-audited with trucks.
- **Commanded acceleration is clamped to −6 m/s².** Raw IDM can emit −600
  when a light change traps a car past its comfortable stopping point; the
  clamp also keeps brake-light logic (`acc < -0.7`) sane.
- **`turn_at_u` is a legacy-arterial mechanism.** City routes transition at
  lane ends through explicit connector paths. Both forms slow for an upcoming
  turn in `Simulator.step`; keep that dual behavior when changing braking.
- **Leader search looks across at most 2 route hops** (`route[:2]` with
  accumulated offsets). Cross-path conflicts inside junction boxes are
  prevented by *signal phasing + all-red clearance* and the all-way-stop
  manager. The city permits only straight movements at signals; turns occur
  at serialized all-way stops. There is no runtime geometric collision engine.
  Adding signal turns, permissive movements, or right-on-red requires
  movement-specific conflict handling and an oriented-box traffic audit.
- **Scenario endings differ.** The legacy ego wraps on the eastbound arterial.
  The city ego receives a new boundary-to-boundary route after completing a
  trip; non-ego city traffic despawns at its destination boundary.
- **Ego routes must contain a turn.** City routing is shortest-hop BFS, which
  favors straight lines; ego route requests pass `min_turns=1, skip=2` so
  every ego trip visibly turns (seed 1 once drove straight for 3+ minutes).
  `skip` matters: the ego spawns at `paths[2]`, so turns in the skipped stub
  don't count. Covered by `test_ego_route_always_contains_a_turn`.
- **RoadGraph geometry is load-bearing.** Lanes at controlled nodes end *at
  their stop line* (the control applies at `u = lane.length`), and connector
  polylines must start/end exactly at the endpoints of the lanes they join —
  `RoadGraph.validate()` enforces this; run it on anything an importer emits.
- **Renderer and sensor share obstacle geometry.** `WorldGeometry` and
  `CityWorld` are generated once and consumed by both `render.draw_world` and
  `Simulator.ego_scene`.
  Decorative lane paint, crosswalks, stop lines, signal heads, and lighting
  effects are intentionally renderer-only. New physical obstacles should be
  added to the shared world rather than drawn only in the renderer.
- **Intensity model:** per-surface reflectivity (`traffic.REFL`) × `exp(-r/55)`
  × noise, clipped 0..1; display applies `sqrt()` gamma before the plasma
  LUT so dim curbs stay visible. Signs are intentionally near-retroreflective
  (0.95) — a future "detect signs by intensity" demo depends on that.
- **numpy ≥ 2.0:** 2-argument `np.cross` on 2-D vectors is deprecated — use
  the module-local `_cross2` helpers, never `np.cross`, for 2-D work.
- **Pygame specifics:** screen y is flipped (all transforms negate y);
  `gfxdraw` needs int coords and is the only draw API that alpha-blends RGBA
  tuples (used for glows); scale constants are `PXM_W` / `PXM_L` px-per-meter.
  Fonts must survive machines without Menlo (SysFont fallback list).
- **GIF size:** frames are quantized to **one shared adaptive palette**
  (sampled across the run) before saving — per-frame palettes triple the
  file size. Keep that code path when touching `_save_gif`.

## Conventions

- Style: matches the existing code — module docstrings that explain the
  *model* (not just the API), `from __future__ import annotations`, section
  comments (`# --- ... ---`), ~80-col lines, dataclasses for value types,
  no type-checking ceremony beyond annotations.
- Comments state constraints and non-obvious physics ("bumper gap, so trucks
  qualify too"), never narration of the next line.
- New tunables go next to their peers: legacy world constants at the top of
  `sim/arterial.py`, city geometry/routing constants at the top of
  `sim/city/graph.py`, IDM params in the `IDM` dict in `sim/traffic.py`, and
  colors in `render/live.py` / `render/stills.py`.
- Keep `render/stills.py` (matplotlib, feeds cli/demo.py) and
  `render/live.py` (pygame live) separate — they serve different artifacts;
  don't try to unify them.
- Git: don't commit unless asked. If asked, note that `assets/` binaries belong
  in the commit but `outputs/` and `.venv/` never do.

## Performance baselines (Apple Silicon, Python 3.13)

| Operation | Budget / measured |
|---|---|
| City `sim.step(dt)` | well below 0.1 ms in normal traffic |
| City LiDAR local scene | commonly ~80 segments; varies with blocks/traffic |
| City `step` + world + LiDAR rendering | ~2.3 ms headless after warm-up (16 ms frame budget) |
| 24 s GIF export (480 frames, 1080 px) | ~20 s, ~12 MB |

Headless training loops (future RL/detector work) should call `sim.step` +
`Lidar2D.scan` directly and skip rendering entirely — that path sustains
thousands of steps per second.

## Updating this guide

As you work on this repository, you must update this file to reflect any new architectures, tools, or workflows you introduce or modify. Consider this file the single source of truth for repository health. Every significant structural change or new constraint must be appended or updated here before completing a task.
