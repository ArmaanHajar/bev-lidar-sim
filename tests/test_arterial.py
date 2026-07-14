"""Smoke checks for the legacy arterial scenario after the module split."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bev_lidar_sim import ArterialSimulator


class ArterialSimulatorTest(unittest.TestCase):
    def test_arterial_runs_and_ego_progresses(self):
        sim = ArterialSimulator(seed=1)
        start_u = sim.ego.u
        for _ in range(600):        # 30 sim-seconds
            sim.step(0.05)
        self.assertGreater(sim.ego.u, start_u)
        self.assertTrue(sim.vehicles)
        self.assertLess(len(sim.vehicles), 30)

    def test_ego_scene_sees_arterial_geometry(self):
        sim = ArterialSimulator(seed=1)
        for _ in range(100):
            sim.step(0.05)
        scene = sim.ego_scene(max_range=45.0)
        self.assertGreater(len(scene.wall_refl), 0)
        self.assertTrue(any(b.label == "building" for b in scene.boxes))

    def test_seed_reproduces_traffic(self):
        a = ArterialSimulator(seed=4)
        b = ArterialSimulator(seed=4)
        for _ in range(400):
            a.step(0.05)
            b.step(0.05)
        state_a = [(v.vid, v.path.name, v.u, v.v) for v in a.vehicles]
        state_b = [(v.vid, v.path.name, v.u, v.v) for v in b.vehicles]
        self.assertEqual(state_a, state_b)


if __name__ == "__main__":
    unittest.main()
