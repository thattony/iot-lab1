# Part 2 demo checklist

Keep the final video under 10 minutes and show code while explaining it.

## 1. Advanced mapping

- Place at least one obstacle in the ultrasonic sensor's peripheral sweep.
- Keep the car and obstacle visible while running `python scan_map.py`.
- Show the sensor sweeping and then show `runtime/mapping/map.png`.
- Explain green free cells, red obstacles, dark unknown cells, median filtering,
  and interpolation.

## 2. Object detection

- Run `python object_detection.py`.
- Hold a stop-sign image where the camera can see it.
- Show the bounding box, label, confidence, FPS, and `action STOP` output.
- Explain 640x480 capture, 300x300 model input, four LiteRT threads, and the
  confidence-threshold tradeoff.

## 3. Routing demonstrations

- Calibrate centimeters per second before recording.
- Mark the destination visibly and state its coordinates aloud.
- Use at least two nontrivial obstacles.
- Run destination 1 with `--goal-forward 60 --goal-left 30`.
- Rearrange the obstacles, visibly mark destination 2, and run with
  `--goal-forward 70 --goal-left -30`.
- Show the yellow route and explain obstacle clearance, Hybrid A*, repeated
  rescans, and pose updates.

## 4. Full self-driving

- Use at least one physical obstacle and one printed stop sign.
- State that the traffic rule is a complete three-second stop.
- Show the car route around the obstacle, detect the sign, stop, and continue
  toward the marked destination.
- Keep a hand near the power switch and press Ctrl+C if behavior is unsafe.

## 5. Code walkthrough

- `mapping.py`: rays, interpolation, pose model, obstacle inflation
- `object_detection.py`: preprocessing, LiteRT inference, labels and boxes
- `planner.py`: Hybrid A* state, steering actions, heuristic, clearance
- `autonomous.py`: rescan/replan loop, vision policy, emergency stop, cleanup

## 6. Report finishing items

- Add every group member's name and NetID.
- Add the private video link and verify course staff can access it.
- Add screenshots and results from both physical routing runs.
- Add a screenshot/result from the integrated stop-sign run.
- Export the final report as PDF.
