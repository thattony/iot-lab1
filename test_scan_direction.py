import unittest
from unittest.mock import Mock, patch

from mapping import OccupancyGrid, Pose, OCCUPIED
from scan_map import sweep


class ScanDirectionTest(unittest.TestCase):
    @patch("scan_map.time.sleep")
    @patch("scan_map.sample_distance", return_value=20.0)
    def test_negative_servo_maps_to_left_with_or_without_legacy_flag(self, sample, sleep):
        for legacy_flag in (False, True):
            car = Mock()
            readings = sweep(car, [-40, 0, 40], sample_count=3,
                             sample_delay_s=0, settle_s=0, invert_pan=legacy_flag)
            self.assertEqual([r.angle_deg for r in readings], [40, 0, -40])
            grid = OccupancyGrid(size_cm=100, resolution_cm=1)
            grid.update_scan(Pose(), readings[:1], sensor_offset_cm=0)
            import numpy as np
            rows, _ = np.where(grid.cells == OCCUPIED)
            self.assertTrue(all(grid.origin_cm + row * grid.resolution_cm > 0 for row in rows))
            car.set_cam_pan_angle.assert_called_with(0)
            car.stop.assert_called_once()
