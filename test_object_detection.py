import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from object_detection import Detection, load_labels, traffic_action


class ObjectDetectionHelpersTest(unittest.TestCase):
    def test_coco_placeholder_is_removed(self):
        with TemporaryDirectory() as folder:
            labels = Path(folder) / "labels.txt"
            labels.write_text("???\nperson\nbicycle\n")
            self.assertEqual(load_labels(labels), ["person", "bicycle"])

    def test_person_and_stop_sign_require_stop(self):
        person = Detection("person", 0.9, (0, 0, 10, 10))
        stop_sign = Detection("stop sign", 0.8, (0, 0, 10, 10))
        car = Detection("car", 0.9, (0, 0, 10, 10))
        self.assertEqual(traffic_action([person]), "STOP")
        self.assertEqual(traffic_action([stop_sign]), "STOP")
        self.assertEqual(traffic_action([car]), "CLEAR")


if __name__ == "__main__":
    unittest.main()
