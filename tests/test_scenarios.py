"""Scenario checks: maps validate, missions route and complete."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bev_lidar_sim import SCENARIOS, make_sim


class ScenarioTest(unittest.TestCase):
    def test_every_scenario_builds_a_valid_graph(self):
        for name, sc in SCENARIOS.items():
            graph = sc.build()
            graph.validate()
            self.assertTrue(graph.roads, f"{name}: no road centerlines")
            self.assertTrue(graph.boundary_in(), name)
            self.assertTrue(graph.boundary_out(), name)

    def test_missions_start_at_od_and_complete(self):
        for name, sc in SCENARIOS.items():
            if sc.ego_od is None:
                continue
            sim = make_sim(name, seed=1)
            self.assertEqual(sim.ego.path.start_node, sc.ego_od[0], name)
            route_ends = ([p for p, _ in sim.ego.route] or [sim.ego.path])
            self.assertEqual(route_ends[-1].end_node, sc.ego_od[1], name)
            arrived = False
            for _ in range(9000):           # up to 450 sim-seconds
                sim.step(0.05)
                if sim.ego_status == "arrived — restarting trip":
                    arrived = True
                    break
            self.assertTrue(arrived, f"{name}: mission never completed")
            # The restarted trip begins back at the origin.
            self.assertEqual(sim.ego.path.start_node, sc.ego_od[0], name)

    def test_riverside_main_road_is_curved(self):
        graph = SCENARIOS["riverside"].build()
        main = [l for l in graph.lanes.values() if l.id == "M1>M2"][0]
        self.assertGreater(len(main.pts), 4)     # a real polyline, not a line
        ys = main.pts[:, 1]
        self.assertGreater(ys.max() - ys.min(), 3.0)

    def test_roadworks_leaves_closed_block_out_of_the_graph(self):
        graph = SCENARIOS["roadworks"].build()
        self.assertNotIn("1-2>2-2", graph.lanes)
        self.assertNotIn("2-2>1-2", graph.lanes)
        sim = make_sim("roadworks", seed=1)
        self.assertTrue(sim.world.barriers)


if __name__ == "__main__":
    unittest.main()
