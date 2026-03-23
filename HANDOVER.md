# Tennis Tagger MVP — Handover

## What Happened This Session

Analyzed the Tennis Tagger app's current state and created a full plan to reach MVP (replacing human taggers with ML).

### Key Discovery

Ran the real pipeline on test videos and found **5 cascading failures**:

| Gap | Current | Target |
|-----|---------|--------|
| Ball detection | ~10% of frames | >50% |
| Court homography | Identity matrix (broken) | Valid court coords |
| Shot segmentation | 1 mega-point for 60s | Multiple points |
| Hitter ID | Always player 1 | Alternates |
| Speed | 0.3 km/h | 80-260 km/h |

### What Was Produced

1. **Spec:** `docs/superpowers/specs/2026-03-23-mvp-detection-fix-stack-design.md` — Approved design for 5-layer fix stack
2. **Plan:** `docs/superpowers/plans/2026-03-23-mvp-detection-fix-stack.md` — 9 tasks (Task 0-8), TDD, bite-sized steps with exact code

### Uncommitted Files (3)

```
M ml/trajectory.py              — segmentation params, velocity helpers
M ml/tests/test_trajectory.py   — tests for dedup & shot boundary
M testdata/tennis_mid2_60s.mp4_output.csv
```

## What To Do Next

Execute the plan task by task. Start with **Task 0** (commit uncommitted work), then proceed 1 through 8 in order.

```
Task 0: Commit uncommitted work (3 files above)
Task 1: Lower TrackNet threshold from 0.5 to 0.3 for BallTrackerNet
Task 2: Remove frame skip in rpc_tracknet_batch
Task 3: Add set_background_reference RPC + sample 30 frames across video
Task 4: Add broadcast fallback homography when court detection fails
Task 5: Implement segment_detections() + wire multi-trajectory fitting
Task 6: Add Vy field to TrajectoryResult + velocity-based hitter fallback
Task 7: Add speed_valid field + sanity clamping (5-300 km/h)
Task 8: End-to-end validation on 60s and full match video
```

**Layers must go in order:** 1→2→3 (detection), then 4 (homography), then 5 (segmentation), then 6 (hitter), then 7 (speed), then 8 (validate).

## How To Resume

Tell Claude: "Read HANDOVER.md and the plan at docs/superpowers/plans/2026-03-23-mvp-detection-fix-stack.md, then execute starting from Task 0 using subagent-driven development."
