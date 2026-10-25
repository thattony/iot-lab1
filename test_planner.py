import unittest

from mapping import OCCUPIED, OccupancyGrid, Pose
from planner import HybridAStar


class HybridAStarTest(unittest.TestCase):
    def test_plans_forward_route(self):
        grid = OccupancyGrid(size_cm=200, resolution_cm=2)
        route = HybridAStar(grid, clearance_cm=10, step_cm=8).plan(Pose(), (60, 0))
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 0)
        self.assertLessEqual(((route[-1].pose.x_cm - 60) ** 2 + route[-1].pose.y_cm**2) ** 0.5, 10)

    def test_routes_around_inflated_obstacle(self):
        grid = OccupancyGrid(size_cm=200, resolution_cm=2)
        # A short wall blocks the direct path but leaves room above and below.
        for y_cm in range(-15, 16, 2):
            x, y = grid.world_to_cell(30, y_cm)
            grid.cells[y, x] = OCCUPIED
        planner = HybridAStar(grid, clearance_cm=8, step_cm=6)
        route = planner.plan(Pose(), (65, 0))
        self.assertIsNotNone(route)
        self.assertTrue(any(abs(motion.pose.y_cm) > 20 for motion in route))
        for motion in route:
            cell = grid.world_to_cell(motion.pose.x_cm, motion.pose.y_cm)
            self.assertFalse(planner.blocked[cell[1], cell[0]])

    def test_rejects_blocked_goal(self):
        grid = OccupancyGrid(size_cm=100, resolution_cm=2)
        x, y = grid.world_to_cell(20, 0)
        grid.cells[y, x] = OCCUPIED
        self.assertIsNone(HybridAStar(grid, clearance_cm=5).plan(Pose(), (20, 0)))


if __name__ == "__main__":
    unittest.main()

