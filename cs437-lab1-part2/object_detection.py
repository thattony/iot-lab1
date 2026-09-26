#!/usr/bin/env python3
"""Real-time PiCar-X object detection for CS 437 Lab 1, Step 7."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # Lets non-camera helper tests run on a development laptop.
    cv2 = None


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    # Pixel coordinates in (left, top, right, bottom) order.
    box: tuple[int, int, int, int]


def load_labels(path: str | Path) -> list[str]:
    labels = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    # The original COCO SSD label file starts with a placeholder, but this
    # model returns zero-based class IDs (class 0 is "person").
    if labels and labels[0].lower() in {"???", "background", "__background__"}:
        labels = labels[1:]
    return labels


def traffic_action(detections: list[Detection]) -> str:
    """Return the conservative driving action used by Step 9 later."""
    labels = {d.label.lower() for d in detections}
    if labels & {"person", "stop sign", "traffic light"}:
        return "STOP"
    return "CLEAR"


class ObjectDetector:
    def __init__(self, model_path: str | Path, label_path: str | Path, *, threads: int = 4):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed in the active Python environment.")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "LiteRT is not installed. Activate ~/cs437-lab1-part2-env first."
            ) from exc

        self.labels = load_labels(label_path)
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()

        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected one model input, found {len(inputs)}")
        self.input = inputs[0]
        shape = self.input["shape"]
        self.input_height, self.input_width = int(shape[1]), int(shape[2])

        outputs = self.interpreter.get_output_details()
        self.outputs = {item["name"]: item for item in outputs}
        self.boxes_output = self._find_output(outputs, suffix="PostProcess", last_size=4)
        self.classes_output = self._find_output(outputs, suffix="PostProcess:1")
        self.scores_output = self._find_output(outputs, suffix="PostProcess:2")
        self.count_output = self._find_output(outputs, suffix="PostProcess:3")

    @staticmethod
    def _find_output(outputs: list[dict[str, Any]], *, suffix: str, last_size: int | None = None):
        matches = [item for item in outputs if item["name"].endswith(suffix)]
        if last_size is not None:
            matches = [item for item in matches if int(item["shape"][-1]) == last_size]
        if len(matches) != 1:
            names = [item["name"] for item in outputs]
            raise RuntimeError(f"Cannot identify model output {suffix!r}; outputs are {names}")
        return matches[0]

    def detect(self, rgb_frame: np.ndarray, threshold: float = 0.45) -> list[Detection]:
        height, width = rgb_frame.shape[:2]
        resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        tensor = np.expand_dims(resized, axis=0)
        if tensor.dtype != self.input["dtype"]:
            tensor = tensor.astype(self.input["dtype"])

        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.boxes_output["index"])[0]
        classes = self.interpreter.get_tensor(self.classes_output["index"])[0]
        scores = self.interpreter.get_tensor(self.scores_output["index"])[0]
        count = int(self.interpreter.get_tensor(self.count_output["index"])[0])

        detections: list[Detection] = []
        for box, class_id, score in zip(boxes[:count], classes[:count], scores[:count]):
            if float(score) < threshold:
                continue
            top, left, bottom, right = (float(value) for value in box)
            x1 = max(0, min(width - 1, round(left * width)))
            y1 = max(0, min(height - 1, round(top * height)))
            x2 = max(0, min(width - 1, round(right * width)))
            y2 = max(0, min(height - 1, round(bottom * height)))
            index = int(class_id)
            label = self.labels[index] if 0 <= index < len(self.labels) else f"class-{index}"
            detections.append(Detection(label, float(score), (x1, y1, x2, y2)))
        return detections


def annotate(rgb_frame: np.ndarray, detections: list[Detection], fps: float) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed in the active Python environment.")
    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (40, 40, 230) if detection.label.lower() in {
            "person", "stop sign", "traffic light"
        } else (40, 210, 80)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection.label} {detection.score:.0%}"
        cv2.putText(image, caption, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(image, f"{fps:.1f} FPS  action={traffic_action(detections)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 60), 2, cv2.LINE_AA)
    return image


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "models" / "detect.tflite")
    parser.add_argument("--labels", type=Path, default=root / "models" / "labelmap.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "runtime")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after this many frames; 0 runs until Ctrl+C")
    parser.add_argument("--preview", action="store_true",
                        help="Open a local display window (normally omit over SSH)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if cv2 is None:
        print("OpenCV is missing. Activate ~/cs437-lab1-part2-env first.", file=sys.stderr)
        return 2
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 is missing; run this program on the Raspberry Pi.", file=sys.stderr)
        return 2

    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("--threshold must be in (0, 1]")

    detector = ObjectDetector(args.model, args.labels, threads=args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_image = args.output_dir / "latest.jpg"
    latest_json = args.output_dir / "detections.json"

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": 15},
        buffer_count=4,
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)  # Let automatic exposure settle.

    running = True

    def stop(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    frame_count = 0
    started = time.monotonic()
    last_report = started
    try:
        while running and (args.max_frames == 0 or frame_count < args.max_frames):
            rgb_frame = camera.capture_array("main")
            detections = detector.detect(rgb_frame, args.threshold)
            frame_count += 1
            now = time.monotonic()
            fps = frame_count / max(now - started, 1e-6)
            image = annotate(rgb_frame, detections, fps)
            cv2.imwrite(str(latest_image), image)

            payload = {
                "timestamp": time.time(),
                "fps": round(fps, 3),
                "action": traffic_action(detections),
                "detections": [asdict(item) for item in detections],
            }
            atomic_write_json(latest_json, payload)

            if now - last_report >= 1.0 or args.max_frames == 1:
                summary = ", ".join(
                    f"{item.label}={item.score:.0%}" for item in detections
                ) or "none"
                print(f"FPS {fps:.1f} | action {payload['action']} | detections: {summary}", flush=True)
                last_report = now

            if args.preview:
                cv2.imshow("CS 437 Step 7", image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        camera.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Stopped cleanly. Latest image: {latest_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
