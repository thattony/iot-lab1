import math
import unittest

import numpy as np

from mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, Pose, RangeReading


class OccupancyGridTest(unittest.TestCase):
    def test_forward_hit_marks_endpoint_and_free_ray(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=1)
        pose = Pose()
        grid.update_scan(pose, [RangeReading(0, 10)], max_range_cm=50, sensor_offset_cm=0)
        endpoint = grid.world_to_cell(10, 0)
        midpoint = grid.world_to_cell(5, 0)
        self.assertEqual(grid.cells[endpoint[1], endpoint[0]], OCCUPIED)
        self.assertEqual(grid.cells[midpoint[1], midpoint[0]], FREE)

    def test_invalid_echo_does_not_claim_free_space(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=1)
        grid.update_scan(Pose(), [RangeReading(0, None)], sensor_offset_cm=0)
        self.assertTrue(np.all(grid.cells == UNKNOWN))

    def test_long_reading_marks_clear_ray_without_obstacle(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=1)
        grid.update_scan(Pose(), [RangeReading(0, 80)], max_range_cm=30, sensor_offset_cm=0)
        self.assertEqual(int((grid.cells == OCCUPIED).sum()), 0)
        self.assertEqual(grid.cells[grid.world_to_cell(20, 0)[1], grid.world_to_cell(20, 0)[0]], FREE)

    def test_adjacent_similar_returns_are_interpolated(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=1)
        grid.update_scan(
            Pose(),
            [RangeReading(-10, 20), RangeReading(10, 20)],
            max_range_cm=50,
            sensor_offset_cm=0,
        )
        center = grid.world_to_cell(20 * math.cos(math.radians(10)), 0)
        self.assertEqual(grid.cells[center[1], center[0]], OCCUPIED)

    def test_pose_integration(self):
        pose = Pose().integrate(velocity_cm_s=10, yaw_rate_rad_s=0, dt_s=2)
        self.assertAlmostEqual(pose.x_cm, 20)
        self.assertAlmostEqual(pose.y_cm, 0)
        turned = Pose().integrate(10, math.pi / 2, 1)
        self.assertGreater(turned.x_cm, 0)
        self.assertGreater(turned.y_cm, 0)

    def test_bicycle_motion_curves_left(self):
        pose = Pose().advance_bicycle(10, math.radians(20), wheelbase_cm=12)
        self.assertGreater(pose.x_cm, 0)
        self.assertGreater(pose.y_cm, 0)
        self.assertGreater(pose.yaw_rad, 0)

    def test_obstacle_inflation_adds_clearance(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=2)
        x, y = grid.world_to_cell(10, 0)
        grid.cells[y, x] = OCCUPIED
        blocked = grid.inflated_obstacles(6)
        near_x, near_y = grid.world_to_cell(14, 0)
        far_x, far_y = grid.world_to_cell(20, 0)
        self.assertTrue(blocked[near_y, near_x])
        self.assertFalse(blocked[far_y, far_x])


if __name__ == "__main__":
    unittest.main()
