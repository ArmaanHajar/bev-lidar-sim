"""Fast structural checks for the procedural city environment."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bev_lidar_sim import CitySimulator
from bev_lidar_sim import Lidar2D


class CitySimulatorTest(unittest.TestCase):
    def test_driving_modes_change_ego_pace_and_can_switch_live(self):
        speeds = []
        for mode in ("safe", "normal", "daring"):
            sim = CitySimulator(seed=1, driving_mode=mode)
            sim.vehicles = [sim.ego]
            for _ in range(40):
                sim.step(0.05)
            speeds.append(sim.ego.v)

        self.assertLess(speeds[0], speeds[1])
        self.assertLess(speeds[1], speeds[2])

        sim.set_driving_mode("daring")
        self.assertEqual(sim.driving_mode, "daring")
        with self.assertRaises(ValueError):
            sim.set_driving_mode("reckless")

    def test_city_has_connected_routes_and_varied_geometry(self):
        sim = CitySimulator(seed=1)

        self.assertEqual(len(sim.intersections), 16)
        self.assertEqual(len(sim.lanes), 80)
        self.assertGreater(len(sim.world.buildings), 20)
        self.assertGreater(len(sim.world.parked), 8)
        ego_lanes = 1 + sum(p.name.startswith("lane:")
                            for p, _ in sim.ego.route)
        self.assertGreaterEqual(ego_lanes, 4)

        for start in sim.incoming_boundary:
            self.assertTrue(any(sim._route_between(start, goal) is not None
                                for goal in sim.outgoing_boundary))

    def test_ego_route_always_contains_a_turn(self):
        # Shortest-hop routing favors straight lines; the ego must still
        # demonstrably turn on every trip (min_turns=1 in _random_route).
        for seed in (1, 2, 3, 7):
            sim = CitySimulator(seed=seed)
            for trip in range(3):
                route = [sim.ego.path] + [p for p, _ in sim.ego.route]
                self.assertTrue(any(p.is_turn for p in route),
                                f"straight-only ego trip {trip} seed {seed}")
                # "new destination" is set the step the route renews and
                # holds until the next step overwrites it.
                for _ in range(12000):          # up to 600 sim-seconds
                    sim.step(0.05)
                    if sim.ego_status == "new destination":
                        break
                else:
                    self.fail(f"ego never finished a trip (seed {seed})")

    def test_connectors_join_lane_endpoints(self):
        sim = CitySimulator(seed=2)
        for lane in sim.lanes.values():
            for connector, outgoing, _ in sim.transitions.get(id(lane), []):
                np.testing.assert_allclose(connector.pts[0], lane.pts[-1])
                np.testing.assert_allclose(connector.pts[-1], outgoing.pts[0])

    def test_seed_reproduces_traffic(self):
        a = CitySimulator(seed=7)
        b = CitySimulator(seed=7)
        for _ in range(500):
            a.step(0.05)
            b.step(0.05)

        state_a = [(v.vid, v.path.name, v.u, v.v) for v in a.vehicles]
        state_b = [(v.vid, v.path.name, v.u, v.v) for v in b.vehicles]
        self.assertEqual(state_a, state_b)

    def test_lidar_scans_local_city_geometry(self):
        sim = CitySimulator(seed=3)
        for _ in range(600):
            sim.step(0.05)
        scene = sim.ego_scene(max_range=45.0)
        seg_a, seg_b = scene.segments()
        lidar = Lidar2D(n_beams=360, max_range=45.0,
                        noise_std=0.0, dropout=0.0,
                        rng=np.random.default_rng(3))
        scan = lidar.scan((0.0, 0.0), seg_a, seg_b, scene.reflectivity())

        self.assertGreater(int(scan.hit.sum()), 25)
        self.assertTrue(np.all(np.isfinite(scan.ranges)))
        self.assertTrue(any(box.label == "building" for box in scene.boxes))


if __name__ == "__main__":
    unittest.main()
