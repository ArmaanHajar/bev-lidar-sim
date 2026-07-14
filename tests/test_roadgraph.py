"""Schema checks: grid graph validity, JSON round-trip, sim equivalence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bev_lidar_sim import CitySimulator, build_city_roadgraph
from bev_lidar_sim import RoadGraph


def _traffic_state(sim):
    return [(v.vid, v.path.name, round(v.u, 9), round(v.v, 9))
            for v in sim.vehicles]


class RoadGraphTest(unittest.TestCase):
    def test_grid_graph_is_valid_and_deterministic(self):
        a = build_city_roadgraph()
        b = build_city_roadgraph()
        a.validate()

        self.assertEqual(len(a.lanes), 80)
        self.assertEqual(sum(n.kind != "boundary" for n in a.nodes.values()), 16)
        self.assertEqual(len(a.boundary_in()), 16)
        self.assertEqual(len(a.boundary_out()), 16)
        # Straight-only at signals, turns at stops -> every lane into an
        # intersection has at least one movement out of it.
        for lane in a.lanes.values():
            if a.nodes[lane.end].kind != "boundary":
                self.assertTrue(a.connectors_from(lane.id))
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_json_round_trip_preserves_graph(self):
        graph = build_city_roadgraph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "city.json"
            graph.save_json(path)
            loaded = RoadGraph.load_json(path)
        loaded.validate()
        self.assertEqual(graph.to_dict(), loaded.to_dict())
        # The file itself is plain JSON (shareable scenario format).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "city.json"
            graph.save_json(path)
            self.assertIn("nodes", json.loads(path.read_text()))

    def test_simulator_runs_identically_from_loaded_graph(self):
        graph = RoadGraph.from_dict(build_city_roadgraph().to_dict())
        native = CitySimulator(seed=11)
        from_schema = CitySimulator(seed=11, graph=graph)
        for _ in range(400):
            native.step(0.05)
            from_schema.step(0.05)
        self.assertEqual(_traffic_state(native), _traffic_state(from_schema))


if __name__ == "__main__":
    unittest.main()
