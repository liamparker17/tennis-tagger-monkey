# Tennis Tagger MVP — Detection Fix Stack Design

**Date:** 2026-03-23
**Status:** Approved
**Goal:** Fix the 5 critical gaps preventing Tennis Tagger from replacing human taggers

## Context

Tennis Tagger v2 is a Go + Python hybrid that processes tennis match video and exports 62-column Dartfish-compatible CSV. The target customer is a company currently using human taggers. The CSV export format is already compatible.

Testing on `tennis_60s.mp4` revealed 5 cascading failures:

| Gap | Metric | Current | Target |
|-----|--------|---------|--------|
| Ball detection rate | Frames with ball detected | ~10% | >50% |
| Court homography | Court-space position validity | Broken (cy=-3.20) | 0 ≤ cx ≤ 8.23, 0 ≤ cy ≤ 23.77 |
| Shot segmentation | Points in 60s clip | 1 (all shots lumped) | >1 |
| Hitter identification | Hitter alternation | Always player 1 | Alternates 0/1 |
| Speed estimation | Serve speed | 0.3 km/h | 80-260 km/h |

## Implementation Order

**Layers MUST be implemented in order 1 → 2 → 3 → 4 → 5.** Each layer depends on the one before it:

- Layer 1 (detection) must work before Layer 3 (segmentation) has enough data to segment
- Layer 2 (homography) must work before Layer 5 (speed) produces meaningful values
- Layer 3 (segmentation) must work before Layer 4 (hitter ID) can alternate between players

## Architecture

No architectural changes. All fixes are within existing components:

- `ml/tracknet.py` — TrackNetDetector inference wrapper
- `ml/bridge_server.py` — JSON-RPC dispatch (background subtraction, trajectory fitting)
- `ml/trajectory.py` — TrajectoryFitter + segmentation (partially implemented)
- `internal/point/shot.go` — Shot segmentation and hitter ID
- `internal/pipeline/concurrent.go` — Pipeline orchestration (no changes expected)

## Layer 1: Fix Background Subtraction & TrackNet Threshold

### Problem
- Background is built from the same 32-frame batch it subtracts from, so diff signal is near-zero
- `_PEAK_THRESHOLD = 0.5` is too high for BallTrackerNet's 256-class softmax output
- TrackNet skips every other frame (`(i - 4) % 2 != 0`), capping detection at 50%

### Fix

**File: `ml/bridge_server.py` — `rpc_tracknet_batch`**
- Accept a `reference_frames` parameter (list of frame dicts) for building background reference
- Build background reference once from frames sampled across the full video (e.g., 30 evenly spaced), passed in during the court detection phase
- Persist the `BackgroundSubtractor` instance across batch calls (already done)

**File: `ml/tracknet.py` — `_peak_detect`**
- Add a `classifier_threshold` parameter, default 0.3, used when `_is_classifier=True`
- Keep 0.5 for TrackNetV2 sigmoid output

**File: `ml/bridge_server.py` — `rpc_tracknet_batch`**
- Remove the `(i - 4) % 2 != 0` skip — run TrackNet on every frame with a full triplet (i >= 4)

### Pass Criteria
Ball detected in >50% of frames on `tennis_60s.mp4`.

## Layer 2: Fix Court Homography

### Problem
When `detect_court` returns identity matrix or fails, pixel coords pass through as court metres. All downstream values (in/out, speed, placement) are garbage.

### Fix

**File: `ml/bridge_server.py` — `rpc_detect_court`**
- After `detect_court`, check if returned homography is identity (or near-identity)
- If so, substitute a hardcoded broadcast fallback homography computed from standard court dimensions and typical broadcast camera position
- Log a warning: "Court detection failed, using broadcast fallback homography"

**File: `cmd/tagger/main.go`**
- Add `--court-corners` CLI flag accepting 4 pixel coordinate pairs for manual homography computation
- Pass through config to bridge

**Fallback homography derivation:**
Standard broadcast angle: camera at ~10m height, ~5m behind baseline, centered. Map the 4 court corners (pixel positions for 640x360 frame) to normalised court coords [0,1].

### Pass Criteria
All court positions satisfy `0 ≤ cx ≤ 8.23` and `0 ≤ cy ≤ 23.77` (with small margin for off-court balls).

## Layer 3: Wire Segmentation Into Pipeline

### Problem
`rpc_fit_trajectories` fits one trajectory to all ball positions. The uncommitted `deduplicate_detections` and `is_same_shot` logic isn't called.

### Fix

**File: `ml/trajectory.py`**
- Add `segment_detections(detections, fps)` function:
  1. Call `deduplicate_detections(detections)`
  2. Sort by frame_index
  3. Split into segments at gaps > `_MIN_GAP_FRAMES` or where `is_same_shot` returns False
  4. Return list of detection segments

**File: `ml/bridge_server.py` — `rpc_fit_trajectories`**
- Import and call `segment_detections` before fitting
- Call `fitter.fit()` per segment
- Return list of trajectories (one per segment)

### Pass Criteria
Multiple trajectories and >1 point recognized in 60s clip.

## Layer 4: Fix Hitter Identification

### Problem
With one mega-trajectory, hitter is always player 1. Even with segmentation, hitter logic needs trajectory direction relative to net.

### Existing Code
`internal/point/shot.go` already has `hitterFromBounce(cy)` which uses bounce court-Y position relative to the net:
- `cy < NetY` → far player (hitter=1)
- `cy >= NetY` → near player (hitter=0)

This position-based approach is valid but only works when bounces are detected. It also doesn't handle cases where bounces are missing.

### Fix

**File: `internal/point/shot.go` — `SegmentShots`**
- Keep existing `hitterFromBounce` as primary signal when bounces are available
- Add velocity-based fallback for shots without detected bounces:
  - `vy > 0` (moving toward far baseline) → near player (hitter=0) hit it
  - `vy < 0` (moving toward near baseline) → far player (hitter=1) hit it
- Add alternation sanity check: within a point, if two consecutive shots have the same hitter, flag the second as low-confidence

### Pass Criteria
Hitter alternates between 0 and 1 within a rally.

## Layer 5: Fix Speed Estimation

### Problem
Speed = `sqrt(vx² + vy²) * 3.6` is correct, but broken homography produces wrong vx/vy values.

### Fix

**Automatically fixed by Layer 2** — correct homography yields correct metre/second velocities.

**File: `ml/trajectory.py` — `Trajectory` dataclass**
- Add `speed_valid: bool = True` field

**File: `ml/trajectory.py` — `TrajectoryFitter.fit`**
- Add sanity clamping after speed calculation:
  - Serve (first shot of point): 80-260 km/h
  - Rally: 20-180 km/h
  - Outside range: set `speed_kph=0.0` and `speed_valid=False`
- Update `to_dict()` to include `speedValid` key

### Pass Criteria
Serve speeds in 80-260 km/h range, rally shots 20-180 km/h.

## Out of Scope

| Component | Reason |
|-----------|--------|
| Parabolic trajectory fitting | Linear fit sufficient for MVP bounce detection |
| Height estimation | Requires camera calibration, not needed for 2D placement |
| Desktop UI (Wails/Svelte) | Customer needs CSV, not GUI |
| YOLO fine-tuning | Optimization for later |
| Stroke classification (3DCNN) | Nice-to-have, not blocking core tagging |
| Scoreboard OCR | Score state machine works from point recognition |

## Final MVP Gate

Run on full match video (`Andrew Johnson vs Alberto Pulido Moreno.mp4`):
- Correct number of points (compare to video)
- Reasonable rally lengths (2-20 shots per point)
- Hitter alternation within rallies
- Plausible placement zones
- Realistic serve/rally speeds
