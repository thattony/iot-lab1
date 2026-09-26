#!/usr/bin/env python3
"""Low-speed straight-line calibration for PiCar-X localization."""

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Required to start the motors")
    parser.add_argument("--power", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not args.execute:
        print("Dry run only. Add --execute after placing the car on a clear measured track.")
        return 0
    if not 0 < args.power <= 100 or not 0 < args.seconds <= 3:
        raise SystemExit("Use power 1..100 and duration 0..3 seconds")

    from picarx import Picarx

    car = Picarx()
    car.stop()
    car.set_dir_servo_angle(0)
    print("Starting in 5 seconds. Measure the distance traveled; Ctrl+C stops the car.")
    try:
        for remaining in range(5, 0, -1):
            print(remaining, flush=True)
            time.sleep(1)
        car.forward(args.power)
        time.sleep(args.seconds)
    finally:
        car.stop()
    print(f"Divide measured centimeters by {args.seconds:.2f} to get --cm-per-second.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

