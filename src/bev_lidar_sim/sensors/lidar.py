"""A vectorized 2D spinning-LiDAR sensor.

The sensor sits at `origin` and fires `n_beams` rays evenly around 360 degrees.
Each ray is intersected against every scene segment in one numpy operation; the
nearest valid hit becomes the measured range. Occlusion is automatic — a ray
stops at the first surface, so objects cast shadows behind them.

Besides range, the sensor returns a per-beam *intensity*: the reflectivity of
the surface it hit, attenuated with distance and perturbed with noise —
the same extra channel a real LiDAR gives you (retroreflective signs bright,
concrete curbs dim).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cross2(ax, ay, bx, by):
    """z-component of the 2D cross product a x b (broadcasting)."""
    return ax * by - ay * bx


@dataclass
class ScanResult:
    angles: np.ndarray      # (N,) ray azimuths, radians
    ranges: np.ndarray      # (N,) measured range per ray (max_range where no hit)
    points: np.ndarray      # (N, 2) hit points in world frame
    hit: np.ndarray         # (N,) bool, True where a real surface was hit
    intensity: np.ndarray = None   # (N,) return intensity 0..1 (0 where no hit)
    seg_idx: np.ndarray = None     # (N,) index of the segment each beam hit


@dataclass
class Lidar2D:
    n_beams: int = 720
    max_range: float = 50.0
    angle_min: float = 0.0
    angle_max: float = 2.0 * np.pi
    noise_std: float = 0.02   # range noise (meters, 1-sigma)
    dropout: float = 0.02     # fraction of hits randomly lost
    rng: np.random.Generator = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    def scan(self, origin, seg_a: np.ndarray, seg_b: np.ndarray,
             seg_refl: np.ndarray = None) -> ScanResult:
        """Cast all beams from `origin` against segments (seg_a -> seg_b)."""
        origin = np.asarray(origin, dtype=float)
        angles = np.linspace(
            self.angle_min, self.angle_max, self.n_beams, endpoint=False
        )
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N, 2)

        e = seg_b - seg_a                 # segment vectors        (M, 2)
        ao = seg_a - origin               # origin -> seg start    (M, 2)

        # 2D cross products, broadcast over (N rays, M segments).
        # denom = dir x e ;  t = (ao x e) / denom ;  u = (ao x dir) / denom
        denom = _cross2(dirs[:, None, 0], dirs[:, None, 1],
                        e[None, :, 0], e[None, :, 1])              # (N, M)
        ao_x_e = _cross2(ao[:, 0], ao[:, 1], e[:, 0], e[:, 1])     # (M,)
        ao_x_d = _cross2(ao[None, :, 0], ao[None, :, 1],
                         dirs[:, None, 0], dirs[:, None, 1])       # (N, M)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = ao_x_e[None, :] / denom                            # (N, M)
            u = ao_x_d / denom                                     # (N, M)

        eps = 1e-9
        valid = (np.abs(denom) > eps) & (t > eps) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)

        seg_idx = np.argmin(t, axis=1)                             # (N,)
        ranges = t[np.arange(self.n_beams), seg_idx]               # (N,)
        hit = np.isfinite(ranges) & (ranges <= self.max_range)

        # Sensor imperfections.
        if self.noise_std > 0:
            ranges = ranges + self.rng.normal(0.0, self.noise_std, ranges.shape)
        if self.dropout > 0:
            hit &= self.rng.random(ranges.shape) >= self.dropout

        ranges = np.where(hit, ranges, self.max_range)
        points = origin + ranges[:, None] * dirs

        if seg_refl is not None:
            refl = np.asarray(seg_refl, dtype=float)[seg_idx]
            intensity = refl * np.exp(-np.abs(ranges) / 55.0)
            intensity *= 1.0 + self.rng.normal(0.0, 0.06, ranges.shape)
            intensity = np.clip(np.where(hit, intensity, 0.0), 0.0, 1.0)
        else:
            intensity = None

        return ScanResult(angles=angles, ranges=ranges, points=points,
                          hit=hit, intensity=intensity, seg_idx=seg_idx)
