# Gap-Aware Trajectory Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split ball detections into per-shot trajectory segments so the system produces one trajectory per shot instead of one for the entire video.

**Architecture:** Add a preprocessing layer (deduplicate + segment) before the existing `TrajectoryFitter.fit()`. Upgrade the fitter from linear to parabolic. Fix bounce serialization bug. All changes in two Python files — no Go changes.

**Tech Stack:** Python 3.11, NumPy, SciPy (curve_fit), pytest

**Spec:** `docs/superpowers/specs/2026-03-23-gap-aware-trajectory-segmentation-design.md`

---

## File Map

| File | Role |
|------|------|
| `ml/trajectory.py` | All trajectory logic: dedup, segmentation, shot boundary, parabolic fit, bounce detection |
| `ml/bridge_server.py` | RPC orchestration: calls segment + fit in loop |
| `ml/tests/test_trajectory.py` | All unit tests for the above |

---

### Task 1: Fix bounce key serialization bug

**Files:**
- Modify: `ml/trajectory.py:265-270` (bounce dict keys in `fit()`)
- Modify: `ml/trajectory.py:181-185` (bounce dict keys in `find_bounces()`)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing test for bounce camelCase keys**

Create `ml/tests/test_trajectory.py`:

```python
"""Tests for ml.trajectory module."""

import numpy as np
import pytest

from ml.trajectory import TrajectoryFitter


def _identity_homography():
    """Return a 3x3 identity homography (pixel coords = court coords)."""
    return np.eye(3, dtype=float)


def _make_detections(xs, ys, frames):
    """Helper to build detection dicts from parallel lists."""
    return [
        {"x": float(x), "y": float(y), "confidence": 0.9, "frame_index": int(f)}
        for x, y, f in zip(xs, ys, frames)
    ]


class TestBounceKeySerialization:
    """Bounce dicts must use camelCase keys to match Go's json tags."""

    def test_bounce_keys_are_camel_case(self):
        # Create detections with a clear velocity reversal to trigger a bounce
        # Ball moves forward then sharply reverses -> bounce detected
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        detections = _make_detections(
            xs=[1, 2, 3, 4, 5, 6, 7],
            ys=[1, 2, 3, 4, 3, 2, 1],  # reversal at index 3-4
            frames=[0, 1, 2, 3, 4, 5, 6],
        )
        traj = fitter.fit(detections)
        assert traj is not None
        # Check via to_dict() which is what gets serialized to Go
        d = traj.to_dict()
        assert len(d["bounces"]) > 0, "Expected at least one bounce to test key names"
        for bounce in d["bounces"]:
            assert "frameIndex" in bounce, f"Missing camelCase 'frameIndex', got keys: {list(bounce.keys())}"
            assert "inOut" in bounce, f"Missing camelCase 'inOut', got keys: {list(bounce.keys())}"
            assert "frame_index" not in bounce, "Found snake_case 'frame_index' — should be camelCase"
            assert "in_out" not in bounce, "Found snake_case 'in_out' — should be camelCase"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestBounceKeySerialization -v`

Expected: FAIL — bounce dicts currently use `frame_index` and `in_out`.

- [ ] **Step 3: Fix bounce keys to camelCase**

In `ml/trajectory.py`, change three locations:

Line 143-144 (`find_bounces` docstring) — update return type documentation:
```python
        ``{"frameIndex": int, "cx": float, "cy": float}``.
```

Line 181-185 (`find_bounces` return dict):
```python
                bounces.append({
                    "frameIndex": int(detections[i]["frame_index"]),
                    "cx": cx_b,
                    "cy": cy_b,
                })
```

Line 265-270 (`fit` bounce_with_inout construction):
```python
            bounces_with_inout.append({
                "frameIndex": b["frameIndex"],
                "cx": b["cx"],
                "cy": b["cy"],
                "inOut": in_out,
            })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestBounceKeySerialization -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "fix: use camelCase keys in bounce dicts to match Go json tags"
```

---

### Task 2: Add `g_eff` field to Trajectory dataclass

**Files:**
- Modify: `ml/trajectory.py:41-68` (Trajectory dataclass + to_dict)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing test**

Append to `ml/tests/test_trajectory.py`:

```python
from ml.trajectory import Trajectory


class TestTrajectoryDataclass:
    def test_g_eff_field_exists(self):
        t = Trajectory(start_frame=0, end_frame=10)
        assert hasattr(t, "g_eff")
        assert t.g_eff == 0.0

    def test_g_eff_in_to_dict(self):
        t = Trajectory(start_frame=0, end_frame=10, g_eff=1500.0)
        d = t.to_dict()
        assert "gEff" in d
        assert d["gEff"] == 1500.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestTrajectoryDataclass -v`

Expected: FAIL — `g_eff` field doesn't exist.

- [ ] **Step 3: Add g_eff field**

In `ml/trajectory.py`, modify the `Trajectory` dataclass (after line 54):

```python
    vy: float = 0.0                 # fitted y-velocity (m/s)
    g_eff: float = 0.0              # effective gravity (pixels/s^2), 0 = linear fit used
```

And add to `to_dict()` (after line 67):

```python
            "vy": self.vy,
            "gEff": self.g_eff,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestTrajectoryDataclass -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: add g_eff field to Trajectory dataclass"
```

---

### Task 3: Implement `deduplicate_detections()`

**Files:**
- Modify: `ml/trajectory.py` (add new function after constants, before Trajectory class ~line 36)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_trajectory.py`:

```python
from ml.trajectory import deduplicate_detections


class TestDeduplicateDetections:
    def test_no_duplicates_unchanged(self):
        dets = _make_detections([1, 2, 3], [4, 5, 6], [0, 1, 2])
        result = deduplicate_detections(dets)
        assert len(result) == 3

    def test_same_frame_keeps_higher_confidence(self):
        dets = [
            {"x": 10.0, "y": 20.0, "confidence": 0.5, "frame_index": 5},
            {"x": 12.0, "y": 22.0, "confidence": 0.9, "frame_index": 5},
        ]
        result = deduplicate_detections(dets)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9
        assert result[0]["x"] == 12.0

    def test_three_on_same_frame(self):
        dets = [
            {"x": 1.0, "y": 1.0, "confidence": 0.3, "frame_index": 0},
            {"x": 2.0, "y": 2.0, "confidence": 0.8, "frame_index": 0},
            {"x": 3.0, "y": 3.0, "confidence": 0.5, "frame_index": 0},
        ]
        result = deduplicate_detections(dets)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.8

    def test_preserves_order_by_frame(self):
        dets = [
            {"x": 1.0, "y": 1.0, "confidence": 0.9, "frame_index": 10},
            {"x": 2.0, "y": 2.0, "confidence": 0.9, "frame_index": 5},
        ]
        result = deduplicate_detections(dets)
        assert result[0]["frame_index"] == 5
        assert result[1]["frame_index"] == 10

    def test_empty_input(self):
        assert deduplicate_detections([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestDeduplicateDetections -v`

Expected: FAIL — `deduplicate_detections` not defined.

- [ ] **Step 3: Implement deduplicate_detections**

Add to `ml/trajectory.py` after the constants block (after line 34, before the Trajectory class):

```python
def deduplicate_detections(detections: List[dict]) -> List[dict]:
    """Keep one detection per frame_index, choosing the highest confidence.

    YOLO and TrackNet may both detect the ball on the same frame. Duplicates
    on the same frame_index corrupt the rolling-median gap calculation in
    segment_detections(), so they must be removed first.
    """
    if not detections:
        return []
    best: dict[int, dict] = {}
    for d in detections:
        fi = d["frame_index"]
        if fi not in best or d["confidence"] > best[fi]["confidence"]:
            best[fi] = d
    return sorted(best.values(), key=lambda d: d["frame_index"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestDeduplicateDetections -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: add deduplicate_detections to remove same-frame duplicates"
```

---

### Task 4: Implement `is_same_shot()`

**Files:**
- Modify: `ml/trajectory.py` (add new function after `deduplicate_detections`)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_trajectory.py`:

```python
from ml.trajectory import is_same_shot


class TestIsSameShot:
    """Velocity-based shot boundary detection."""

    def test_same_direction_parabola_is_same_shot(self):
        # Ball exits going right and up, re-enters going right and down
        tail = _make_detections([100, 110, 120], [200, 180, 160], [10, 11, 12])
        head = _make_detections([140, 150, 160], [160, 180, 200], [20, 21, 22])
        assert is_same_shot(tail, head, fps=30.0) is True

    def test_reversed_horizontal_is_new_shot(self):
        # Ball exits going right, re-enters going left
        tail = _make_detections([100, 110, 120], [200, 180, 160], [10, 11, 12])
        head = _make_detections([500, 490, 480], [160, 180, 200], [20, 21, 22])
        assert is_same_shot(tail, head, fps=30.0) is False

    def test_too_few_points_uses_gap_threshold(self):
        # Only 1 point in tail — can't estimate velocity
        tail = _make_detections([100], [200], [10])
        head = _make_detections([200], [200], [30])
        # Gap = 20 frames > 15 frame fallback threshold
        assert is_same_shot(tail, head, fps=30.0) is False

    def test_too_few_points_short_gap_is_same(self):
        tail = _make_detections([100], [200], [10])
        head = _make_detections([200], [200], [14])
        # Gap = 4 frames < 15 frame fallback threshold
        assert is_same_shot(tail, head, fps=30.0) is True

    def test_gap_exceeds_max_merge_is_new_shot(self):
        # Even with consistent direction, gap of 80 frames (>75) = new shot
        tail = _make_detections([100, 110, 120], [200, 180, 160], [10, 11, 12])
        head = _make_detections([140, 150, 160], [160, 180, 200], [92, 93, 94])
        assert is_same_shot(tail, head, fps=30.0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestIsSameShot -v`

Expected: FAIL — `is_same_shot` not defined.

- [ ] **Step 3: Implement is_same_shot**

Add to `ml/trajectory.py` after `deduplicate_detections`:

```python
# Segmentation parameters
_MIN_GAP_FRAMES = 5
_ADAPTIVE_MULTIPLIER = 5.0
_MAX_MERGE_GAP_FRAMES = 75
_FALLBACK_GAP_FRAMES = 15  # used when < 2 points to estimate velocity
_G_EFF_MIN = 500.0   # min plausible effective gravity (px/s^2)
_G_EFF_MAX = 3000.0   # max plausible effective gravity (px/s^2)


def _estimate_exit_velocity(detections: List[dict], fps: float) -> Optional[tuple]:
    """Estimate (vx, vy) in pixels/s from the last two detections (exit velocity)."""
    if len(detections) < 2:
        return None
    d0, d1 = detections[-2], detections[-1]
    dt = (d1["frame_index"] - d0["frame_index"]) / fps
    if dt <= 0:
        return None
    return ((d1["x"] - d0["x"]) / dt, (d1["y"] - d0["y"]) / dt)


def _estimate_entry_velocity(detections: List[dict], fps: float) -> Optional[tuple]:
    """Estimate (vx, vy) in pixels/s from the first two detections (entry velocity)."""
    if len(detections) < 2:
        return None
    d0, d1 = detections[0], detections[1]
    dt = (d1["frame_index"] - d0["frame_index"]) / fps
    if dt <= 0:
        return None
    return ((d1["x"] - d0["x"]) / dt, (d1["y"] - d0["y"]) / dt)


def is_same_shot(
    tail: List[dict],
    head: List[dict],
    fps: float,
    max_merge_gap: int = _MAX_MERGE_GAP_FRAMES,
) -> bool:
    """Determine if two detection segments belong to the same shot arc.

    Args:
        tail: Last few detections of the previous segment.
        head: First few detections of the next segment.
        fps:  Frames per second.
        max_merge_gap: Never merge across gaps larger than this (frames).

    Returns:
        True if likely the same shot (ball went off-screen and came back),
        False if likely a new shot (opponent returned the ball).
    """
    if not tail or not head:
        return False

    gap_frames = head[0]["frame_index"] - tail[-1]["frame_index"]

    # Hard cap: never merge across very large gaps
    if gap_frames > max_merge_gap:
        return False

    exit_vel = _estimate_exit_velocity(tail, fps)
    entry_vel = _estimate_entry_velocity(head, fps)

    # Fallback: not enough points to estimate velocity
    if exit_vel is None or entry_vel is None:
        return gap_frames <= _FALLBACK_GAP_FRAMES

    exit_vx, exit_vy = exit_vel
    entry_vx, entry_vy = entry_vel

    # Signal 1: horizontal direction should be approximately the same
    # (both moving right, or both moving left)
    if abs(exit_vx) > 5 and abs(entry_vx) > 5:
        if (exit_vx > 0) != (entry_vx > 0):
            return False  # horizontal reversal = new shot

    # Signal 2: vertical consistency — ball should exit upward (vy < 0 in
    # pixel coords where y increases downward) and re-enter downward (vy > 0)
    if exit_vy < 0 and entry_vy > 0:
        # Consistent parabola: up then down
        # Signal 3: check if gap duration is plausible for a parabolic arc
        # Expected time off-screen ≈ 2 * |vy_exit| / g_eff
        # Use midpoint of plausible g_eff range as estimate
        g_est = (_G_EFF_MIN + _G_EFF_MAX) / 2.0
        expected_gap_s = 2.0 * abs(exit_vy) / g_est
        actual_gap_s = gap_frames / fps
        if actual_gap_s <= expected_gap_s * 2.5:
            return True

    # Default: if gap is small enough, assume same shot
    return gap_frames <= _FALLBACK_GAP_FRAMES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestIsSameShot -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: add is_same_shot for velocity-based shot boundary detection"
```

---

### Task 5: Implement `segment_detections()`

**Files:**
- Modify: `ml/trajectory.py` (add new function after `is_same_shot`)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_trajectory.py`:

```python
from ml.trajectory import segment_detections


class TestSegmentDetections:
    def test_continuous_detections_single_segment(self):
        """No gaps -> one segment."""
        dets = _make_detections(
            xs=list(range(10)),
            ys=list(range(10)),
            frames=list(range(10)),
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == 10

    def test_clear_gap_splits(self):
        """Large gap in the middle -> two segments."""
        dets = _make_detections(
            xs=[1, 2, 3, 100, 101, 102],
            ys=[1, 2, 3, 100, 101, 102],
            frames=[0, 1, 2, 60, 61, 62],  # 58-frame gap, reversed direction
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 2
        assert len(segments[0]) == 3
        assert len(segments[1]) == 3

    def test_deduplicates_same_frame(self):
        """Same-frame duplicates should be deduplicated before segmentation."""
        dets = [
            {"x": 10.0, "y": 20.0, "confidence": 0.5, "frame_index": 0},
            {"x": 12.0, "y": 22.0, "confidence": 0.9, "frame_index": 0},
            {"x": 20.0, "y": 30.0, "confidence": 0.8, "frame_index": 1},
            {"x": 30.0, "y": 40.0, "confidence": 0.8, "frame_index": 2},
        ]
        segments = segment_detections(dets, fps=30.0)
        # After dedup: 3 detections on frames 0, 1, 2 — one segment
        assert len(segments) == 1
        assert len(segments[0]) == 3

    def test_dead_ball_always_splits(self):
        """Gap > max_merge_gap_frames always splits regardless of velocity."""
        # Same direction, consistent parabola, but gap is 80 frames (> 75 max)
        dets = _make_detections(
            xs=[100, 110, 120, 140, 150, 160],
            ys=[200, 180, 160, 160, 180, 200],
            frames=[0, 1, 2, 82, 83, 84],
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 2

    def test_empty_input(self):
        assert segment_detections([], fps=30.0) == []

    def test_fewer_than_min_detections_returned(self):
        """Even segments with < 3 detections should be returned (fitter discards them)."""
        dets = _make_detections([1, 2], [1, 2], [0, 1])
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == 2

    def test_lob_pattern_sparse_detections(self):
        """Sparse detections (every 3-4 frames) with a long gap should still split."""
        # Sparse segment 1: every 3 frames, then 60-frame gap, reversed direction
        dets = _make_detections(
            xs=[100, 110, 120, 130, 500, 490, 480, 470],
            ys=[300, 280, 260, 240, 240, 260, 280, 300],
            frames=[0, 3, 6, 9, 69, 72, 75, 78],  # 60-frame gap
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 2

    def test_fast_rally_dense_detections(self):
        """Dense detections (every frame) with a short gap should still split on direction reversal."""
        # Dense segment 1 going right, then 15-frame gap, segment 2 going left
        dets = _make_detections(
            xs=[100, 105, 110, 115, 120, 400, 395, 390, 385, 380],
            ys=[200, 195, 190, 185, 180, 180, 185, 190, 195, 200],
            frames=[0, 1, 2, 3, 4, 19, 20, 21, 22, 23],  # 15-frame gap
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestSegmentDetections -v`

Expected: FAIL — `segment_detections` not defined.

- [ ] **Step 3: Implement segment_detections**

Add to `ml/trajectory.py` after `is_same_shot`:

```python
def segment_detections(
    detections: List[dict],
    fps: float,
    min_gap: int = _MIN_GAP_FRAMES,
    adaptive_mul: float = _ADAPTIVE_MULTIPLIER,
    max_merge_gap: int = _MAX_MERGE_GAP_FRAMES,
) -> List[List[dict]]:
    """Split detections into per-shot segments using adaptive gap detection.

    1. Deduplicates (one detection per frame, highest confidence wins).
    2. Sorts by frame_index.
    3. Splits at gaps that exceed an adaptive threshold.
    4. Merges adjacent segments that appear to be the same shot arc
       (ball went off-screen and came back).

    Returns a list of detection lists, one per segment.
    """
    dets = deduplicate_detections(detections)
    if not dets:
        return []

    # Phase 1: split at every gap that exceeds the adaptive threshold
    segments: List[List[dict]] = []
    current: List[dict] = [dets[0]]
    recent_gaps: List[int] = []

    for i in range(1, len(dets)):
        gap = dets[i]["frame_index"] - dets[i - 1]["frame_index"]

        # Rolling median of recent gaps (window of 5)
        recent_gaps.append(gap)
        if len(recent_gaps) > 5:
            recent_gaps.pop(0)
        median_gap = float(sorted(recent_gaps)[len(recent_gaps) // 2])

        threshold = max(adaptive_mul * median_gap, min_gap)

        if gap > threshold:
            segments.append(current)
            current = [dets[i]]
            recent_gaps = []  # reset for new segment
        else:
            current.append(dets[i])

    segments.append(current)

    # Phase 2: merge adjacent segments that belong to the same shot arc
    if len(segments) <= 1:
        return segments

    merged: List[List[dict]] = [segments[0]]
    for seg in segments[1:]:
        tail = merged[-1][-3:]  # last 3 points of previous segment
        head = seg[:3]           # first 3 points of this segment
        if is_same_shot(tail, head, fps, max_merge_gap):
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestSegmentDetections -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: add segment_detections with adaptive gap detection"
```

---

### Task 6: Upgrade `TrajectoryFitter.fit()` to parabolic model

**Files:**
- Modify: `ml/trajectory.py:193-287` (the `fit()` method)
- Test: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Append to `ml/tests/test_trajectory.py`:

```python
class TestParabolicFit:
    def _parabolic_detections(self, n=8, g=1500.0):
        """Generate detections following a perfect parabola in pixel space.

        Ball starts at (100, 300), moves right at 200 px/s, and arcs upward
        then downward under effective gravity g (px/s^2).
        py(t) = 300 - 200*t + 0.5*g*t^2  (y=0 is top of frame)
        """
        fps = 30.0
        xs, ys, frames = [], [], []
        for i in range(n):
            t = i / fps
            px = 100 + 200 * t
            py = 300 - 200 * t + 0.5 * g * t * t
            xs.append(px)
            ys.append(py)
            frames.append(i)
        return _make_detections(xs, ys, frames), fps

    def test_parabolic_fit_recovers_g_eff(self):
        """With 8 points on a parabola, g_eff should be close to the true value."""
        dets, fps = self._parabolic_detections(n=8, g=1500.0)
        fitter = TrajectoryFitter(_identity_homography(), fps)
        traj = fitter.fit(dets)
        assert traj is not None
        # g_eff should be within the plausible range and close to 1500
        assert 500 <= traj.g_eff <= 3000
        assert abs(traj.g_eff - 1500.0) < 300  # within 20%

    def test_three_points_falls_back_to_linear(self):
        """With exactly 3 points, should use linear fit (g_eff = 0)."""
        dets = _make_detections([100, 200, 300], [100, 200, 300], [0, 1, 2])
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        traj = fitter.fit(dets)
        assert traj is not None
        assert traj.g_eff == 0.0  # linear fallback

    def test_four_points_uses_parabolic(self):
        """4+ points should attempt parabolic fit."""
        dets, fps = self._parabolic_detections(n=4, g=1500.0)
        fitter = TrajectoryFitter(_identity_homography(), fps)
        traj = fitter.fit(dets)
        assert traj is not None
        assert traj.g_eff > 0  # parabolic was used

    def test_g_eff_clamped_to_plausible_range(self):
        """If fitted g is outside 500-3000, it should be clamped."""
        # Nearly linear data -> g_eff would be tiny, should clamp to 0 (linear fallback)
        dets = _make_detections(
            [100, 110, 120, 130, 140],
            [100, 110, 120, 130, 140],
            [0, 1, 2, 3, 4],
        )
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        traj = fitter.fit(dets)
        assert traj is not None
        # For nearly-linear data, parabolic fit's g should be near 0,
        # which is outside plausible range -> falls back to linear
        assert traj.g_eff == 0.0 or 500 <= traj.g_eff <= 3000

    def test_noisy_parabolic_data_stays_in_range(self):
        """With noisy parabolic data, g_eff should still be within plausible range."""
        import random
        random.seed(42)
        fps = 30.0
        g_true = 1500.0
        xs, ys, frames = [], [], []
        for i in range(10):
            t = i / fps
            px = 100 + 200 * t + random.gauss(0, 3)  # 3px noise
            py = 300 - 200 * t + 0.5 * g_true * t * t + random.gauss(0, 3)
            xs.append(px)
            ys.append(py)
            frames.append(i)
        dets = _make_detections(xs, ys, frames)
        fitter = TrajectoryFitter(_identity_homography(), fps)
        traj = fitter.fit(dets)
        assert traj is not None
        assert 500 <= traj.g_eff <= 3000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py::TestParabolicFit -v`

Expected: FAIL — `g_eff` is always 0 (linear fit), parabolic not implemented.

- [ ] **Step 3: Rewrite fit() with parabolic model**

Replace the fitting section of `TrajectoryFitter.fit()` in `ml/trajectory.py` (lines 193-287). The full updated method:

```python
    def fit(self, detections: List[dict]) -> Optional[Trajectory]:
        """Fit a trajectory to a list of ball detections.

        Uses parabolic fit (py = py0 + vy*t + 0.5*g*t^2) for 4+ detections,
        linear fit for exactly 3. Returns None for < 3 detections.
        """
        if len(detections) < _MIN_DETECTIONS:
            return None

        detections = sorted(detections, key=lambda d: d["frame_index"])

        # Pixel coordinates and time
        px_arr = np.array([d["x"] for d in detections], dtype=float)
        py_arr = np.array([d["y"] for d in detections], dtype=float)
        frames = np.array([d["frame_index"] for d in detections], dtype=float)
        t_arr = frames / self.fps
        t0 = t_arr[0]
        t_rel = t_arr - t0

        # Court coordinates for bounce detection and output
        court_pts = [self.pixel_to_court(d["x"], d["y"]) for d in detections]
        cx_arr = np.array([p[0] for p in court_pts], dtype=float)
        cy_arr = np.array([p[1] for p in court_pts], dtype=float)

        # --- Fit ---
        g_eff = 0.0
        use_parabolic = len(detections) >= 4

        def linear(t, c0, v):
            return c0 + v * t

        def parabolic(t, c0, v, g):
            return c0 + v * t + 0.5 * g * t * t

        cx0, vx, cy0, vy = 0.0, 0.0, 0.0, 0.0
        fit_confidence = 0.0

        try:
            # X (court coords): always linear
            popt_x, _ = curve_fit(linear, t_rel, cx_arr)
            cx0, vx = float(popt_x[0]), float(popt_x[1])

            if use_parabolic:
                # Y (pixel coords): parabolic
                popt_py, _ = curve_fit(parabolic, t_rel, py_arr, p0=[py_arr[0], 0.0, 1500.0])
                fitted_g = float(popt_py[2])

                if _G_EFF_MIN <= abs(fitted_g) <= _G_EFF_MAX:
                    g_eff = abs(fitted_g)
                    # Also fit court-Y as linear for speed estimate
                    popt_y, _ = curve_fit(linear, t_rel, cy_arr)
                    cy0, vy = float(popt_y[0]), float(popt_y[1])

                    # Confidence from pixel-space parabolic residuals
                    py_pred = parabolic(t_rel, *popt_py)
                    rmse_py = float(np.sqrt(np.mean((py_arr - py_pred) ** 2)))
                    cx_pred = linear(t_rel, cx0, vx)
                    rmse_cx = float(np.sqrt(np.mean((cx_arr - cx_pred) ** 2)))
                    rmse = (rmse_py / 360.0 + rmse_cx / 8.23) / 2.0  # normalize
                    fit_confidence = float(np.clip(1.0 - rmse * 5.0, 0.0, 1.0))
                else:
                    # g outside plausible range -> fall back to linear
                    use_parabolic = False

            if not use_parabolic:
                # Linear fit for Y (court coords)
                popt_y, _ = curve_fit(linear, t_rel, cy_arr)
                cy0, vy = float(popt_y[0]), float(popt_y[1])
                g_eff = 0.0

                cx_pred = linear(t_rel, cx0, vx)
                cy_pred = linear(t_rel, cy0, vy)
                rmse_x = float(np.sqrt(np.mean((cx_arr - cx_pred) ** 2)))
                rmse_y = float(np.sqrt(np.mean((cy_arr - cy_pred) ** 2)))
                rmse = (rmse_x + rmse_y) / 2.0
                fit_confidence = float(np.clip(1.0 - rmse / 2.0, 0.0, 1.0))

            count_scale = min(1.0, len(detections) / 10.0)
            fit_confidence *= count_scale

        except (RuntimeError, ValueError):
            if len(t_rel) >= 2 and t_rel[-1] > 0:
                vx = float((cx_arr[-1] - cx_arr[0]) / t_rel[-1])
                vy = float((cy_arr[-1] - cy_arr[0]) / t_rel[-1])
                cx0 = float(cx_arr[0])
                cy0 = float(cy_arr[0])
            fit_confidence = 0.1
            g_eff = 0.0

        # Build positions list
        positions = []
        for i, d in enumerate(detections):
            positions.append({
                "cx": float(cx_arr[i]),
                "cy": float(cy_arr[i]),
                "t": float(t_arr[i]),
                "frame_index": int(d["frame_index"]),
            })

        # Bounce detection
        raw_bounces = self.find_bounces(detections)
        bounces_with_inout = []
        for b in raw_bounces:
            in_out = self.classify_landing(b["cx"], b["cy"])
            bounces_with_inout.append({
                "frameIndex": b["frameIndex"],
                "cx": b["cx"],
                "cy": b["cy"],
                "inOut": in_out,
            })

        speed_ms = float(np.sqrt(vx ** 2 + vy ** 2))
        speed_kph = speed_ms * 3.6

        return Trajectory(
            start_frame=int(detections[0]["frame_index"]),
            end_frame=int(detections[-1]["frame_index"]),
            positions=positions,
            bounces=bounces_with_inout,
            speed_kph=speed_kph,
            confidence=fit_confidence,
            cx0=cx0,
            vx=vx,
            cy0=cy0,
            vy=vy,
            g_eff=g_eff,
        )
```

- [ ] **Step 4: Run all trajectory tests**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/test_trajectory.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: upgrade trajectory fitter to parabolic model with linear fallback"
```

---

### Task 7: Wire segmentation into bridge_server.py

**Files:**
- Modify: `ml/bridge_server.py:339-382` (`rpc_fit_trajectories` — full method)
- Test: integration (run pipeline on test clip)

- [ ] **Step 1: Update rpc_fit_trajectories to segment before fitting**

In `ml/bridge_server.py`, replace lines 339-382 (the full `rpc_fit_trajectories` method):

```python
    def rpc_fit_trajectories(self, params: dict) -> Any:
        """Fit ball trajectories from a sequence of ball position detections.

        Input params:
            ball_positions: list of {x, y, confidence, frameIndex}
            court: {homography: [[3x3 matrix]]}
            fps: float

        Returns list of trajectory dicts (see ml.trajectory.Trajectory.to_dict).
        """
        from ml.trajectory import TrajectoryFitter, segment_detections

        ball_positions = params.get("ball_positions", [])
        court = params.get("court", {})
        fps = float(params.get("fps", 30.0))

        homography_raw = court.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        homography = np.array(homography_raw, dtype=float)

        if homography.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got shape {homography.shape}")

        # Normalise detection key names
        detections = []
        for bp in ball_positions:
            detections.append({
                "x": float(bp.get("x", 0)),
                "y": float(bp.get("y", 0)),
                "confidence": float(bp.get("confidence", 0)),
                "frame_index": int(bp.get("frameIndex", bp.get("frame_index", 0))),
            })

        if not detections:
            return []

        # Segment into per-shot groups, then fit each
        segments = segment_detections(detections, fps)
        fitter = TrajectoryFitter(homography, fps)

        trajectories = []
        for seg in segments:
            traj = fitter.fit(seg)
            if traj is not None:
                trajectories.append(traj.to_dict())

        return trajectories
```

- [ ] **Step 2: Run all Python tests**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -m pytest ml/tests/ -v`

Expected: ALL PASS

- [ ] **Step 3: Run Go tests to verify nothing broke**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/... -count=1 2>&1 | grep -E "ok|FAIL"`

Expected: ALL ok

- [ ] **Step 4: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/bridge_server.py
git commit -m "feat: wire segment_detections into trajectory RPC for per-shot fitting"
```

---

### Task 8: Integration test on real video

**Files:** None (verification only)

- [ ] **Step 1: Run pipeline on test clip**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go build -o tagger.exe ./cmd/tagger/
./tagger.exe testdata/tennis_mid2_60s.mp4
```

Expected output should show:
- **Trajectories: 5-15** (was 1)
- **Shots: 8-15** (was 10 — should stay similar or improve)
- **Points: 1+** (should still recognize points)

- [ ] **Step 2: Verify bounce frameIndex values are non-zero**

Check the JSON output to confirm bounce `frameIndex` values are real frame numbers, not 0.

- [ ] **Step 3: Commit any fixes if needed, then final commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add -A
git commit -m "test: verify gap-aware trajectory segmentation on 60s test clip"
```
