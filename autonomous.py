#!/usr/bin/env python3
"""Steps 8-9: repeatedly map, plan with Hybrid A*, and drive the PiCar-X."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from mapping import OccupancyGrid, Pose
from object_detection import ObjectDetector, annotate, atomic_write_json
from planner import HybridAStar, Motion
from scan_map import sample_distance, sweep


class VisionGuard:
    """Camera policy: wait for people and make a timed stop at stop signs."""

    def __init__(self, root: Path, threshold: float, threads: int = 4):
        from picamera2 import Picamera2

        self.threshold = threshold
        self.detector = ObjectDetector(
            root / "models" / "detect.tflite",
            root / "models" / "labelmap.txt",
            threads=threads,
        )
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 15},
            buffer_count=4,
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1.0)
        self.output_dir = root / "runtime" / "autonomous" / "vision"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_stop_sign_time = -math.inf

    def observe(self):
        import cv2

        started = time.monotonic()
        frame = self.camera.capture_array("main")
        detections = self.detector.detect(frame, self.threshold)
        fps = 1.0 / max(time.monotonic() - started, 1e-6)
        image = annotate(frame, detections, fps)
        cv2.imwrite(str(self.output_dir / "latest.jpg"), image)
        payload = {
            "timestamp": time.time(),
            "fps": round(fps, 3),
            "detections": [asdict(item) for item in detections],
        }
        atomic_write_json(self.output_dir / "detections.json", payload)
        return detections

    def enforce(self, car, *, stop_seconds: float, person_timeout_s: float) -> bool:
        detections = self.observe()
        labels = {item.label.lower() for item in detections}

        if "person" in labels:
            car.stop()
            print("VISION: person detected; waiting until the path is clear.", flush=True)
            deadline = time.monotonic() + person_timeout_s
            while "person" in labels:
                if time.monotonic() >= deadline:
                    print("VISION: person did not clear before timeout; aborting.", file=sys.stderr)
                    return False
                time.sleep(0.25)
                labels = {item.label.lower() for item in self.observe()}
            print("VISION: person cleared; replanning.", flush=True)

        now = time.monotonic()
        if "stop sign" in labels and now - self.last_stop_sign_time >= 8.0:
            car.stop()
            print(f"VISION: stop sign detected; stopping for {stop_seconds:.1f} seconds.", flush=True)
            time.sleep(stop_seconds)
            self.last_stop_sign_time = time.monotonic()
        return True

    def close(self):
        self.camera.stop()


class PicarXDriver:
    def __init__(
        self,
        car,
        *,
        power: int,
        cm_per_second: float,
        emergency_distance_cm: float,
        invert_steering: bool,
    ):
        self.car = car
        self.power = power
        self.cm_per_second = cm_per_second
        self.emergency_distance_cm = emergency_distance_cm
        self.invert_steering = invert_steering

    def drive(self, motion: Motion) -> float:
        """Execute one short arc and return estimated distance actually traveled."""
        steering = -motion.steering_deg if self.invert_steering else motion.steering_deg
        duration = motion.distance_cm / self.cm_per_second
        self.car.set_dir_servo_angle(steering)
        started = time.monotonic()
        blocked = False
        try:
            self.car.forward(self.power)
            while time.monotonic() - started < duration:
                distance = float(self.car.get_distance())
                if 0 < distance < self.emergency_distance_cm:
                    blocked = True
                    print(f"SAFETY: obstacle at {distance:.1f} cm; emergency stop.", flush=True)
                    break
                time.sleep(0.05)
        finally:
            self.car.stop()
            self.car.set_dir_servo_angle(0)
        elapsed = min(time.monotonic() - started, duration)
        traveled = elapsed * self.cm_per_second
        return max(0.0, traveled - (1.0 if blocked else 0.0))


def append_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-forward", type=float, default=60.0,
                        help="Goal x coordinate in cm from the starting pose")
    parser.add_argument("--goal-left", type=float, default=0.0,
                        help="Goal y coordinate in cm; negative values are right")
    parser.add_argument("--execute", action="store_true",
                        help="Actually drive; without this flag the program only scans and plans")
    parser.add_argument("--power", type=int, default=20)
    parser.add_argument("--cm-per-second", type=float, default=15.0,
                        help="Measured straight-line speed at the selected power")
    parser.add_argument("--step-cm", type=float, default=8.0)
    parser.add_argument("--wheelbase-cm", type=float, default=12.0)
    parser.add_argument("--clearance-cm", type=float, default=18.0)
    parser.add_argument("--goal-tolerance-cm", type=float, default=10.0)
    parser.add_argument("--emergency-distance-cm", type=float, default=22.0)
    parser.add_argument("--max-cycles", type=int, default=80)
    parser.add_argument("--scan-step", type=int, default=15)
    parser.add_argument("--scan-samples", type=int, default=3)
    parser.add_argument("--max-range", type=float, default=150.0)
    parser.add_argument("--vision-threshold", type=float, default=0.45)
    parser.add_argument("--stop-seconds", type=float, default=3.0)
    parser.add_argument("--person-timeout", type=float, default=30.0)
    parser.add_argument("--invert-pan", action="store_true")
    parser.add_argument("--invert-steering", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=root / "runtime" / "autonomous")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.power <= 0 or args.power > 100:
        raise SystemExit("--power must be in 1..100")
    if args.cm_per_second <= 0 or args.step_cm <= 0:
        raise SystemExit("--cm-per-second and --step-cm must be positive")
    if math.hypot(args.goal_forward, args.goal_left) < args.goal_tolerance_cm:
        raise SystemExit("Goal is already inside the goal tolerance")


def main() -> int:
    args = parse_args()
    validate_args(args)
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from picarx import Picarx
    except ImportError:
        print("PiCar-X library missing; activate ~/env.", file=sys.stderr)
        return 2

    car = Picarx()
    car.stop()
    stopped = False

    def stop(_signum=None, _frame=None):
        nonlocal stopped
        stopped = True
        car.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    grid = OccupancyGrid(size_cm=400, resolution_cm=2)
    pose = Pose()
    goal = (args.goal_forward, args.goal_left)
    angles = list(range(-70, 71, args.scan_step))
    if angles[-1] != 70:
        angles.append(70)
    driver = PicarXDriver(
        car,
        power=args.power,
        cm_per_second=args.cm_per_second,
        emergency_distance_cm=args.emergency_distance_cm,
        invert_steering=args.invert_steering,
    )
    vision = None
    log_path = args.output_dir / "run.jsonl"

    if args.execute:
        print("EXECUTE enabled. Starting autonomous run in 5 seconds; Ctrl+C stops immediately.")
        for remaining in range(5, 0, -1):
            print(remaining, flush=True)
            time.sleep(1)
        vision = VisionGuard(root, args.vision_threshold)
    else:
        print("PLAN-ONLY mode: the drive motors will remain stopped.")

    try:
        for cycle in range(args.max_cycles):
            if stopped:
                return 130
            remaining = math.hypot(pose.x_cm - goal[0], pose.y_cm - goal[1])
            if remaining <= args.goal_tolerance_cm:
                print(f"GOAL reached within {remaining:.1f} cm after {cycle} replans.")
                return 0

            print(f"\nCycle {cycle}: pose=({pose.x_cm:.1f}, {pose.y_cm:.1f}) "
                  f"yaw={math.degrees(pose.yaw_rad):.1f} deg; rescanning.")
            readings = sweep(
                car,
                angles,
                sample_count=args.scan_samples,
                sample_delay_s=0.03,
                settle_s=0.08,
                invert_pan=args.invert_pan,
            )
            grid.update_scan(pose, readings, max_range_cm=args.max_range, sensor_offset_cm=8)
            planner = HybridAStar(
                grid,
                clearance_cm=args.clearance_cm,
                wheelbase_cm=args.wheelbase_cm,
                step_cm=args.step_cm,
            )
            route = planner.plan(pose, goal, goal_tolerance_cm=args.goal_tolerance_cm)
            if not route:
                grid.save(args.output_dir, pose=pose, readings=readings, goal=goal)
                print("No safe route found. Move the goal or obstacles and try again.", file=sys.stderr)
                return 4

            grid.save(
                args.output_dir,
                pose=pose,
                readings=readings,
                route=[pose] + [item.pose for item in route],
                goal=goal,
            )
            print(f"Hybrid A*: {len(route)} segments; first steering={route[0].steering_deg:+.0f} deg")
            append_log(
                log_path,
                {
                    "timestamp": time.time(),
                    "cycle": cycle,
                    "pose": asdict(pose),
                    "goal": goal,
                    "route_segments": len(route),
                    "valid_ranges": sum(item.distance_cm is not None for item in readings),
                },
            )

            if not args.execute:
                print(f"Plan saved to {args.output_dir / 'map.png'}")
                return 0

            if not vision.enforce(
                car,
                stop_seconds=args.stop_seconds,
                person_timeout_s=args.person_timeout,
            ):
                return 5

            first = route[0]
            traveled = driver.drive(first)
            pose = pose.advance_bicycle(
                traveled,
                math.radians(first.steering_deg),
                args.wheelbase_cm,
            )
            print(f"Moved an estimated {traveled:.1f} cm; rescanning before next segment.")
        print("Maximum navigation cycles reached before the goal.", file=sys.stderr)
        return 6
    finally:
        car.stop()
        car.set_dir_servo_angle(0)
        car.set_cam_pan_angle(0)
        if vision is not None:
            vision.close()


if __name__ == "__main__":
    raise SystemExit(main())
