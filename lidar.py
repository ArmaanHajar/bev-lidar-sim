"""A vectorized 2D spinning-LiDAR sensor.

The sensor sits at `origin` and fires `n_beams` rays evenly around 360 degrees.
Each ray is intersected against every scene segment in one numpy operation; the
nearest valid hit becomes the measured range. Occlusion is automatic — a ray
stops at the first surface, so objects cast shadows behind them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanResult:
    angles: np.ndarray   # (N,) ray azimuths, radians
    ranges: np.ndarray   # (N,) measured range per ray (max_range where no hit)
    points: np.ndarray   # (N, 2) hit points in world frame
    hit: np.ndarray      # (N,) bool, True where a real surface was hit


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

    def scan(self, origin, seg_a: np.ndarray, seg_b: np.ndarray) -> ScanResult:
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
        denom = np.cross(dirs[:, None, :], e[None, :, :])          # (N, M)
        ao_x_e = np.cross(ao, e)                                   # (M,)
        ao_x_d = (ao[None, :, 0] * dirs[:, None, 1]
                  - ao[None, :, 1] * dirs[:, None, 0])             # (N, M)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = ao_x_e[None, :] / denom                           # (N, M)
            u = ao_x_d / denom                                    # (N, M)

        eps = 1e-9
        valid = (np.abs(denom) > eps) & (t > eps) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)

        ranges = t.min(axis=1)                                    # (N,)
        hit = np.isfinite(ranges) & (ranges <= self.max_range)

        # Sensor imperfections.
        if self.noise_std > 0:
            ranges = ranges + self.rng.normal(0.0, self.noise_std, ranges.shape)
        if self.dropout > 0:
            lost = self.rng.random(ranges.shape) < self.dropout
            hit &= ~lost

        ranges = np.where(hit, ranges, self.max_range)
        points = origin + ranges[:, None] * dirs
        return ScanResult(angles=angles, ranges=ranges, points=points, hit=hit)
