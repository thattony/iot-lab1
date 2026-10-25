# CS 437 Lab 1 Part 2

## Installation

On the Raspberry Pi, follow [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md).
Python dependencies are recorded in `requirements.txt`; system packages and
the optional Robot HAT I2S amplifier require the separate setup steps.

## Step 6: advanced ultrasonic mapping

The mapper sweeps the ultrasonic sensor from -70 to +70 degrees, takes five
readings at each angle, filters noise with the median, and converts valid
returns into a non-probabilistic 2-D occupancy grid. It marks observed free
space as `0`, obstacles as `1`, and keeps unknown cells as `-1` internally.
Adjacent obstacle returns are interpolated only when their ranges are similar,
which prevents the mapper from drawing a wall across a doorway.

On the Raspberry Pi, leave the car stationary with the ultrasonic sensor on
the pan servo, place at least one box beside or in front of it, and run:

```bash
cd ~/project/cs437-lab1-part2
source ~/env/bin/activate
python scan_map.py
```

Outputs are written under `runtime/mapping/`:

- `map.png`: visual map for the demo and report
- `obstacles.csv`: rubric-friendly binary 0/1 map
- `occupancy.npy`: full unknown/free/occupied grid for Step 8
- `scan.json`: raw angle/distance data and pose metadata

The mapping program keeps the drive motors stopped. If the resulting map is
mirrored, rerun it with `python scan_map.py --invert-pan`.

## Step 7: object detection

The detector uses the Pi camera, OpenCV, and the quantized COCO SSD MobileNet
model. It prints detections and measured FPS, saves an annotated frame to
`runtime/latest.jpg`, and writes structured output to
`runtime/detections.json`. A person, stop sign, or traffic light produces a
`STOP` action for the later self-driving integration.

On the Raspberry Pi:

```bash
cd ~/project/cs437-lab1-part2
source ~/env/bin/activate

# Safe one-frame camera/model check; this does not move the car.
python object_detection.py --max-frames 1

# Continuous detector. Press Ctrl+C to stop.
python object_detection.py
```

For the lab demo, hold a stop-sign image in the camera view. The terminal
should report `stop sign`, `action STOP`, and at least 1 FPS. Copy
`runtime/latest.jpg` into the report as evidence. If detection is unstable,
use even lighting, fill more of the image with the sign, and try
`--threshold 0.35`.

The camera code never starts the PiCar-X motors.

## Step 8: Hybrid A* routing

`planner.py` implements Hybrid A*, an A* derivation whose graph state contains
the car's grid position and heading. Its edges are short arcs at the five
steering angles supported by the controller. Obstacles are expanded by an
18 cm clearance radius before planning so the route accounts for the physical
width of the car.

Test mapping and routing without moving the car:

```bash
python autonomous.py --goal-forward 60 --goal-left 30
```

The program performs a new ultrasonic scan, computes a route, and writes the
route over the map in `runtime/autonomous/map.png`. Yellow is the planned
route and the white circle is the goal.

Before the first driving run, measure speed on a clear straight track:

```bash
python calibrate_speed.py --execute --power 20 --seconds 1
```

Measure the centimeters traveled. Use that value as `--cm-per-second`.

## Step 9: integrated self-driving

Use at least two solid obstacles and keep a hand near the power switch. This
command enables the motors after a five-second countdown:

```bash
python autonomous.py \
  --execute \
  --goal-forward 60 \
  --goal-left 30 \
  --power 20 \
  --cm-per-second MEASURED_VALUE
```

The controller rescans and reruns Hybrid A* after every 8 cm motion segment.
The ultrasonic sensor triggers an immediate stop inside 22 cm. The camera is
checked before each motion: a person makes the car wait until the person is
gone, and a stop sign makes it halt for three seconds. `Ctrl+C` stops the
motors and centers both steering and pan servos.

For the two routing demonstrations required by the rubric, use different
obstacle layouts and destinations, for example:

```bash
# Run 1: destination left of the starting heading
python autonomous.py --execute --goal-forward 60 --goal-left 30 \
  --power 20 --cm-per-second MEASURED_VALUE

# Run 2: destination right of the starting heading
python autonomous.py --execute --goal-forward 70 --goal-left -30 \
  --power 20 --cm-per-second MEASURED_VALUE
```

If steering bends opposite the yellow route, add `--invert-steering`. If the
map itself is mirrored, add `--invert-pan`.
