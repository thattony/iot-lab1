#!/usr/bin/env python3
"""Sweep the PiCar-X ultrasonic sensor and generate the Step 6 map."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

from mapping import OccupancyGrid, Pose, RangeReading


def sample_distance(car, *, sample_count: int, sample_delay_s: float) -> float | None:
    samples: list[float] = []
    for _ in range(sample_count):
        value = float(car.get_distance())
        if math.isfinite(value) and value > 0:
            samples.append(value)
        time.sleep(sample_delay_s)
    # Requiring multiple echoes rejects isolated electrical/noise spikes.
    if len(samples) < max(2, sample_count // 2):
        return None
    return round(float(statistics.median(samples)), 2)


def sweep(
    car,
    angles: list[int],
    *,
    sample_count: int,
    sample_delay_s: float,
    settle_s: float,
    invert_pan: bool,
) -> list[RangeReading]:
    readings: list[RangeReading] = []
    try:
        for servo_angle in angles:
            car.set_cam_pan_angle(servo_angle)
            time.sleep(settle_s)
            distance = sample_distance(
                car, sample_count=sample_count, sample_delay_s=sample_delay_s
            )
            map_angle = -servo_angle if invert_pan else servo_angle
            readings.append(RangeReading(map_angle, distance))
            display = "timeout" if distance is None else f"{distance:.1f} cm"
            print(f"servo {servo_angle:+4d} deg -> {display}", flush=True)
    finally:
        car.set_cam_pan_angle(0)
        car.stop()
    return readings


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "runtime" / "mapping")
    parser.add_argument("--min-angle", type=int, default=-70)
    parser.add_argument("--max-angle", type=int, default=70)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-delay", type=float, default=0.035)
    parser.add_argument("--settle", type=float, default=0.12)
    parser.add_argument("--max-range", type=float, default=150.0)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--sensor-offset", type=float, default=8.0)
    parser.add_argument("--invert-pan", action="store_true",
                        help="Mirror scan angles if positive servo motion points right")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step <= 0 or args.min_angle >= args.max_angle:
        raise SystemExit("Require --step > 0 and --min-angle < --max-angle")
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if not (-80 <= args.min_angle <= 80 and -80 <= args.max_angle <= 80):
        raise SystemExit("Keep scan angles between -80 and +80 degrees")

    try:
        from picarx import Picarx
    except ImportError:
        print("PiCar-X library missing; activate ~/cs437-lab1-part2-env.", file=sys.stderr)
        return 2

    angles = list(range(args.min_angle, args.max_angle + 1, args.step))
    if angles[-1] != args.max_angle:
        angles.append(args.max_angle)

    car = Picarx()
    car.stop()
    pose = Pose()
    grid = OccupancyGrid(size_cm=args.size, resolution_cm=args.resolution)
    print(f"Scanning {len(angles)} angles. Motors remain stopped.")
    readings = sweep(
        car,
        angles,
        sample_count=args.samples,
        sample_delay_s=args.sample_delay,
        settle_s=args.settle,
        invert_pan=args.invert_pan,
    )
    grid.update_scan(
        pose,
        readings,
        max_range_cm=args.max_range,
        sensor_offset_cm=args.sensor_offset,
    )
    grid.save(args.output_dir, pose=pose, readings=readings)

    valid = sum(reading.distance_cm is not None for reading in readings)
    occupied = int((grid.cells == 1).sum())
    print(f"Valid readings: {valid}/{len(readings)}; occupied cells: {occupied}")
    print(f"Map: {args.output_dir / 'map.png'}")
    print(f"Binary grid: {args.output_dir / 'obstacles.csv'}")
    if valid == 0:
        print("No valid ultrasonic echoes. Check sensor wiring and power.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

