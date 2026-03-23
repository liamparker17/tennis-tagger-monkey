# Gap-Aware Trajectory Segmentation

**Date:** 2026-03-23
**Status:** Approved
**Context:** Tennis ball goes off-screen at the apex of parabolic arcs in side-on broadcast footage. Camera angles and shot heights vary (lobs vs flat drives vs serves), so gap duration varies widely.

## Problem

The current system sends ALL ball detections as a single flat list to `TrajectoryFitter.fit()`, producing exactly 1 trajectory for the entire video. This makes shot segmentation, speed estimation, and bounce detection nearly useless — they all depend on per-shot trajectories.

The ball is off-screen for a significant portion of each rally. In side-on footage:
- Ball is visible near racket contact (low in frame)
- Ball arcs upward and exits the top of frame
- Ball re-enters frame descending toward the bounce
- Duration off-screen varies: ~0.3s for flat drives, 1-2s for lobs

## Pre-Existing Bug Fix: Bounce Key Serialization

The current `trajectory.py` builds bounce dicts with snake_case keys (`frame_index`, `in_out`) but Go's `Bounce` struct expects camelCase (`frameIndex`, `inOut`). This means bounce `FrameIndex` and `InOut` deserialize as zero-values, silently breaking shot ordering and hitter assignment. **Must fix as part of this work** — change bounce dict keys to camelCase in `to_dict()` output.

## Design

### 0. Deduplicate Detections

**Location:** `ml/trajectory.py` (new function `deduplicate_detections()`)

`mergeBallPositions` in `concurrent.go` keeps both YOLO and TrackNet detections on the same frame. Before segmentation, we must deduplicate: for detections on the same `frame_index`, keep the one with higher confidence. Without this, same-frame duplicates (gap=0) corrupt the rolling median of inter-detection spacing, driving the adaptive threshold to near-zero and shattering the detection list into useless 1-2 point segments.

### 1. Adaptive Segment Splitter

**Location:** `ml/trajectory.py` (new function `segment_detections()`)

Splits a sorted, deduplicated list of ball detections into segments, where each segment represents one continuous arc (one shot). Uses adaptive gap detection rather than a fixed frame threshold.

**Algorithm:**

```
Deduplicate detections (one per frame, highest confidence wins)
Sort detections by frame_index
Compute recent_spacing = rolling median of inter-detection gaps (window=5)
For each consecutive pair of detections:
  gap = frame_diff between them
  If gap > max(adaptive_multiplier * recent_spacing, min_gap_frames):
    Check if this is a same-shot off-screen arc or a new shot:
      - Estimate exit velocity from last 2-3 points of current segment
      - Estimate entry velocity from first 2-3 points of next segment
      - If horizontal direction approximately the same AND vertical direction
        consistent (up->down parabola) AND expected off-screen duration
        roughly matches actual gap: SAME SHOT, merge segments
      - Otherwise: NEW SHOT
    Start new segment on NEW SHOT
```

**Parameters (with defaults):**
- `min_gap_frames`: 5 — absolute minimum gap to even consider splitting (prevents splitting on single missed detections)
- `adaptive_multiplier`: 5.0 — gap must be 5x the recent detection spacing
- `max_merge_gap_frames`: 75 — never merge segments across gaps larger than this (2.5s at 30fps), even if velocity says same shot — covers dead balls and high lobs up to ~2s off-screen

**Why adaptive:** A lob with sparse detections (every 3-4 frames) has a different "normal" spacing than a flat drive with dense detections (every 1-2 frames). The multiplier-on-median approach handles both without tuning.

### 2. Velocity-Based Shot Boundary Detection

**Location:** `ml/trajectory.py` (new function `is_same_shot()`)

Given the tail of segment A and the head of segment B, determine whether they're the same shot (ball went off-screen and came back) or different shots (opponent returned it).

**Signals:**
- **Horizontal direction consistency:** In side-on footage, same-shot arcs maintain approximately the same horizontal direction (ball continues traveling the same way). A reversal between exit and entry suggests the opponent returned it. Note: check direction between segment A exit and segment B entry, not within a segment.
- **Vertical consistency:** Ball exiting upward, re-entering downward = consistent parabola = same shot.
- **Gap duration vs expected:** Given exit velocity `vy_exit` (pixels/s upward) and effective gravity `g_eff` (pixels/s^2, constrained to 500-3000 range), expected time off-screen ≈ `2 * vy_exit / g_eff`. If actual gap is within 2x of expected, likely same shot.

**Fallback:** If we don't have enough points to estimate velocity (< 2 points in a segment), treat any gap > 15 frames (0.5s) as a new shot. This avoids splitting on brief detection gaps while still catching genuine shot boundaries.

### 3. Per-Segment Parabolic Trajectory Fitting

**Location:** `ml/trajectory.py` (modify `TrajectoryFitter.fit()`)

Change the fitter to accept a single segment (not the whole video) and fit a **parabolic** model instead of linear:

**Current (linear):**
```
cx(t) = cx0 + vx * t
cy(t) = cy0 + vy * t
```

**New (parabolic in pixel Y, linear in court X):**
```
px(t) = px0 + vx * t                    (linear — horizontal motion)
py(t) = py0 + vy * t + 0.5 * g * t^2    (parabolic — gravity on vertical axis)
```

Where `g` is effective gravity in pixels/s^2 (fitted from data, constrained to physically plausible range 500-3000 px/s^2 for typical broadcast cameras at 640x360). Court-coordinate transform still happens via homography, but the fit operates in pixel space first where the parabola is natural.

**Minimum points for parabolic fit: 4.** With exactly 3 points, the 3-parameter parabolic model interpolates perfectly (zero residuals) regardless of noise, giving false confidence and potentially wild `g_eff` estimates. Segments with exactly 3 points fall back to linear fit. Segments with < 3 points are discarded (existing `_MIN_DETECTIONS` behavior).

Bounce detection improves: instead of looking for velocity sign changes in noisy finite-difference data, we find where the fitted parabola intersects the court surface (bottom of frame / court plane).

**Trajectory dataclass update:** Add `g_eff: float = 0.0` field to `Trajectory` and include in `to_dict()` output. Needed for `is_same_shot()` across future segments and Phase 2 interpolation.

### 4. Orchestration Changes

**`ml/bridge_server.py` — `rpc_fit_trajectories()`:**
```python
# Before: one fit on all detections
traj = fitter.fit(detections)

# After: segment, then fit each
segments = segment_detections(detections, fps)
trajectories = []
for seg in segments:
    traj = fitter.fit(seg)
    if traj is not None:
        trajectories.append(traj)
return [t.to_dict() for t in trajectories]
```

No changes needed in Go code — `FitTrajectories` already returns `[]TrajectoryResult` and the pipeline already passes that list to `SegmentShots()`. Note: with per-shot trajectories, each trajectory should have ~1 bounce, so `SegmentShots` will naturally produce one shot per trajectory. The existing logic still works correctly.

### 5. Off-Screen Arc Interpolation (Phase 2)

Not in initial implementation. Once segmentation and parabolic fitting work, we can optionally:
- Connect segment N's exit to segment N+1's entry with a fitted parabola
- Estimate apex height, time-of-flight, and predicted bounce point for shots where the bounce itself was off-screen
- This would be additive — doesn't change the segmentation logic

## Known Limitations

- **Serve toss:** The upward toss before a serve produces detections that look like a separate arc. The segmentation may create a phantom segment for the toss. Acceptable for now — the toss segment will have low confidence and no bounce, so it won't produce a Shot.
- **Net shots:** Balls that hit the net and fall back may produce very short segments with ambiguous velocity. These will be discarded by the minimum detection threshold.

## Files Changed

| File | Change |
|------|--------|
| `ml/trajectory.py` | Add `deduplicate_detections()`, `segment_detections()`, `is_same_shot()`. Modify `TrajectoryFitter.fit()` to use parabolic model. Add `g_eff` to `Trajectory`. Fix bounce dict keys to camelCase. |
| `ml/bridge_server.py` | Update `rpc_fit_trajectories()` to segment before fitting. |

No Go changes required.

## Expected Impact

| Metric | Before | After (expected) |
|--------|--------|-------------------|
| Trajectories per 60s clip | 1 | 8-15 (one per shot) |
| Bounce detection accuracy | Poor (linear fit + noise) | Good (parabolic fit + court intersection) |
| Shot speed estimates | One average for whole video | Per-shot speeds |
| Shot direction | Meaningless (averaged) | Per-shot direction vectors |

## Testing

### Unit tests for `segment_detections()`
- Evenly spaced detections (no gaps) -> single segment
- Clear gap in the middle -> two segments
- Duplicate frame detections -> deduplicated before segmentation
- Lob pattern (sparse detections, long gap) -> correct split
- Fast rally (dense detections, short gaps) -> correct split
- Dead ball (3+ second gap) -> always splits regardless of velocity

### Unit tests for `is_same_shot()`
- Same horizontal direction, up->down vertical -> same shot
- Reversed horizontal direction -> new shot
- Insufficient points (< 2) -> falls back to gap threshold
- Gap exceeds max_merge_gap_frames -> new shot regardless

### Unit tests for parabolic fit
- Known parabolic points (4+) -> recovers g_eff within plausible range
- Exactly 3 points -> falls back to linear
- Noisy data -> g_eff stays within constrained range

### Integration
- Run on 60s test clip -> trajectory count increases from 1 to ~10
- Verify bounce frameIndex values are non-zero in output (serialization fix)
