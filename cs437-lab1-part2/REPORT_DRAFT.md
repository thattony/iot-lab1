# CS 437 Lab 1 Part 2 Report

**Group members and NetIDs:** [ADD NAMES AND NETIDS]

**Private demo video:** [ADD VIDEO LINK]

## System overview

Our PiCar-X software combines four modules: an ultrasonic occupancy mapper,
a quantized neural-network object detector, a Hybrid A* route planner, and a
closed-loop controller that rescans and replans after every short movement.
The implementation uses centimeters in a world coordinate frame where the
car begins at `(0, 0)`, positive x is forward, and positive y is left.

## Step 6: advanced mapping

The ultrasonic sensor is mounted on the camera pan servo. The mapper sweeps
from -70 to +70 degrees and samples five echoes per direction. A median filter
rejects isolated noisy values. A valid return marks the cells along the beam
as free and its endpoint as occupied. A timeout leaves cells unknown because
a missing echo is not proof that the space is clear. Nearby endpoints are
interpolated only when their distances differ by at most 15 cm, which avoids
drawing a false wall across large gaps.

The map is a 200 by 200 NumPy array with 2 cm cells. Internally `-1` means
unknown, `0` means observed free, and `1` means occupied. We also export a
binary CSV in the exact zero/one form requested by the rubric. The first
hardware scan produced 11 valid readings out of 15 and 38 occupied cells,
with measured obstacle distances of approximately 52-79 cm.

Localization uses a bicycle motion model. After each commanded movement, the
pose is advanced using measured travel speed, elapsed time, steering angle,
and the 12 cm wheelbase. This is open-loop odometry, so short movements and
frequent rescans limit accumulated error.

## Step 7: object detection

The camera captures 640 by 480 RGB frames through Picamera2. OpenCV resizes
each frame to the 300 by 300 input required by the quantized COCO SSD
MobileNet model. LiteRT runs inference with four CPU threads and the XNNPACK
delegate. The program converts normalized boxes to camera pixels, draws
labels and confidence scores, and publishes JSON for the driving controller.

Our sustained 40-frame hardware test achieved approximately 15.7-16.5 FPS,
well above the requested 1 FPS. The model correctly placed boxes around
chairs and plants in the test scene. For autonomous driving, a detected
person causes the car to wait until the path clears and a stop sign causes a
three-second halt.

### Performance questions

1. **Would hardware acceleration help?** Yes. Convolutional inference is
   highly parallel. Our setup uses XNNPACK's optimized CPU kernels but does
   not use a dedicated GPU, NPU, Coral Edge TPU, or Movidius accelerator. A
   compatible quantized model and hardware delegate could offload inference
   and reduce CPU load further.
2. **Would multithreading help?** It helps inside the inference runtime because
   independent convolution work can use the Pi 5's CPU cores. We use four
   interpreter threads. Adding Python threads around every pipeline stage
   could hurt performance through scheduling overhead, memory bandwidth
   contention, and camera synchronization complexity.
3. **Frame rate versus accuracy:** Lower camera/model resolutions and higher
   confidence thresholds reduce work and false positives, but can miss small
   or distant signs. We retain a 640 by 480 camera frame for useful spatial
   detail, use the model's required 300 by 300 tensor, and default to a 0.45
   confidence threshold. The threshold can be lowered to 0.35 when testing a
   small printed sign.

## Step 8: self-driving navigation

We use Hybrid A*, a derivation of A* that searches over `(x, y, heading)`.
Each graph edge is a physically feasible 8 cm PiCar-X motion at one of five
steering angles: -28, -14, 0, +14, or +28 degrees. The heuristic is Euclidean
distance to the goal. Extra costs discourage unexplored cells, sharp steering,
and unnecessary steering changes.

Before planning, occupied cells are expanded by an 18 cm clearance radius.
This keeps the centerline route far enough from obstacles for the entire car
to pass. The controller executes only the first short route segment, rescans
the environment, updates its pose, and reruns Hybrid A*. This handles newly
visible space and obstacles that move after the initial plan.

Automated tests verify straight routing, routing around an inflated wall,
blocked-goal rejection, collision checking along steering arcs, and pose
integration.

**Physical run 1:** [ADD DESTINATION, OBSTACLE LAYOUT, RESULT, AND SCREENSHOT]

**Physical run 2:** [ADD DIFFERENT DESTINATION, LAYOUT, RESULT, AND SCREENSHOT]

## Step 9: full self-driving

The final controller runs mapping, routing, motion, and vision in one process.
Before every movement it checks the camera. A person blocks motion until the
person disappears. A stop sign triggers a full three-second stop with a
cooldown so the same sign is not treated as a new intersection every frame.
During each motion segment the forward ultrasonic reading is checked every
50 ms; a valid return inside 22 cm stops the motors immediately. Signal and
exception cleanup always stops both motors and centers the steering and pan
servos.

**Traffic sign used:** Stop sign.

**Required response:** Stop completely for three seconds, then resume mapping,
planning, and navigation.

**Integrated physical result:** [ADD COURSE DESCRIPTION, RESULT, AND SCREENSHOT]

## Challenges and design decisions

Ultrasonic timeouts occurred at several center angles during the first scan.
We addressed this by taking multiple samples, using the median, and treating
timeouts as unknown. Open-loop localization is sensitive to battery voltage,
floor friction, and motor calibration, so the controller uses measured speed,
short motion segments, large obstacle clearance, and frequent replanning.
The PiCar-X cannot rotate in place, which made four-neighbor grid instructions
physically awkward; Hybrid A* preserves the rubric's A* requirement while
planning steering arcs the chassis can execute.

## Reproduction commands

```bash
cd ~/project/cs437-lab1-part2
source ~/env/bin/activate
python scan_map.py
python object_detection.py
python autonomous.py --goal-forward 60 --goal-left 30
```

After speed calibration, add `--execute`, `--power 20`, and the measured
`--cm-per-second` value for a physical navigation run.
