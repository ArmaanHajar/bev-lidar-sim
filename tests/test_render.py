import math
import unittest

import numpy as np

from bev_lidar_sim.render.live import _ego_to_world_points


class RenderTransformTests(unittest.TestCase):
    def test_ego_forward_rotates_to_world_heading(self):
        point = _ego_to_world_points((1.0, 0.0), math.pi / 2)

        np.testing.assert_allclose(point, (0.0, 1.0), atol=1e-12)

    def test_rotation_preserves_ranges(self):
        points = np.array(((3.0, 4.0), (-2.0, 5.0), (0.0, 0.0)))
        rotated = _ego_to_world_points(points, -0.73)

        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=1),
            np.linalg.norm(points, axis=1),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
