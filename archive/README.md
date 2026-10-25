# Archive

Kept from the older `cs437-lab1` copy when it was merged into this folder
(2026-10-25). Everything else in that copy was identical to the current code.

- `autonomous_with_person_detection.py`: earlier `autonomous.py` (Sep 27 13:44)
  that still waited for people in the camera view (`--person-timeout`). The
  current `autonomous.py` only reacts to stop signs.
- `runtime/autonomous/`: output of the run made with that version (Sep 27 14:01).
  It stays on the Raspberry Pi only, because `runtime/` is git-ignored.

To run the old version, copy it to the project root first; it imports
`scan_map`, `mapping`, `planner`, and `object_detection` from there.
