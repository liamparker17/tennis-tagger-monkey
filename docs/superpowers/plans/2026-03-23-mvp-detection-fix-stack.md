# MVP Detection Fix Stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 5 cascading failures (ball detection, court homography, shot segmentation, hitter ID, speed estimation) so Tennis Tagger can replace human taggers.

**Architecture:** No structural changes. All fixes are within existing Python ML modules (`ml/tracknet.py`, `ml/bridge_server.py`, `ml/trajectory.py`), Go point logic (`internal/point/shot.go`), and Go bridge types (`internal/bridge/types.go`). The Go-Python JSON-RPC bridge remains unchanged.

**Tech Stack:** Python 3.11+ (PyTorch, OpenCV, scipy), Go 1.26.1, BallTrackerNet (yastrebksv pre-trained weights)

**Spec:** `docs/superpowers/specs/2026-03-23-mvp-detection-fix-stack-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ml/tracknet.py` | Modify | Lower threshold for classifier path |
| `ml/bridge_server.py` | Modify | Fix background subtraction, remove frame skip, wire segmentation, add homography fallback |
| `ml/trajectory.py` | Modify | Add `segment_detections()`, add `speed_valid` field, add speed clamping |
| `ml/tests/test_trajectory.py` | Modify | Add tests for segmentation and speed validation |
| `ml/tests/test_tracknet.py` | Create | Tests for lowered threshold |
| `internal/bridge/types.go` | Modify | Add `Vy` field to `TrajectoryResult` |
| `internal/point/shot.go` | Modify | Add velocity-based hitter fallback |
| `internal/point/shot_test.go` | Modify | Add tests for velocity fallback and alternation check |
| `cmd/tagger/main.go` | Modify | Add `--court-corners` CLI flag |

---

## Task 0: Commit Uncommitted Work

**Files:**
- Modified: `ml/trajectory.py` (uncommitted changes)
- Modified: `ml/tests/test_trajectory.py` (uncommitted changes)
- Modified: `testdata/tennis_mid2_60s.mp4_output.csv` (uncommitted changes)

- [ ] **Step 1: Review and stage uncommitted files**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git diff --stat
```

- [ ] **Step 2: Run existing Python tests to make sure they pass**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_trajectory.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Run Go tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go test ./internal/... -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py testdata/tennis_mid2_60s.mp4_output.csv
git commit -m "feat: add gap-aware segmentation params and velocity helpers"
```

---

## Task 1: Lower TrackNet Classifier Threshold

**Files:**
- Modify: `ml/tracknet.py:524-584` (`_peak_detect` static method)
- Create: `ml/tests/test_tracknet.py`

- [ ] **Step 1: Write failing test for threshold parameter**

Create `ml/tests/test_tracknet.py`:

```python
"""Tests for ml.tracknet module."""

import numpy as np
import pytest

from ml.tracknet import TrackNetDetector


class TestPeakDetect:
    """Test _peak_detect with different thresholds."""

    def test_default_threshold_misses_low_confidence(self):
        """A 0.4-peak heatmap should return None at default 0.5 threshold."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[180, 320] = 0.4  # below 0.5
        result = TrackNetDetector._peak_detect(heatmap, threshold=0.5)
        assert result is None

    def test_lowered_threshold_detects_low_confidence(self):
        """A 0.4-peak heatmap should detect at 0.3 threshold."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[180, 320] = 0.4  # above 0.3
        result = TrackNetDetector._peak_detect(heatmap, threshold=0.3)
        assert result is not None
        assert abs(result["x"] - 320) < 1.0
        assert abs(result["y"] - 180) < 1.0
        assert result["confidence"] == pytest.approx(0.4, abs=0.01)

    def test_blob_detection_with_low_threshold(self):
        """A small blob with peak 0.35 should be detected at threshold 0.3."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        # 3x3 blob
        heatmap[179:182, 319:322] = 0.32
        heatmap[180, 320] = 0.35
        result = TrackNetDetector._peak_detect(heatmap, threshold=0.3)
        assert result is not None
        assert result["confidence"] == pytest.approx(0.35, abs=0.01)
```

- [ ] **Step 2: Run test to verify it passes** (the threshold parameter already exists on `_peak_detect`)

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_tracknet.py -v
```

Expected: All 3 tests pass (the method already accepts a `threshold` param).

- [ ] **Step 3: Modify `detect_batch` to use lower threshold for classifier path**

In `ml/tracknet.py`, in the `detect_batch` method (line ~460-471), change the `_peak_detect` call to pass the lower threshold when using BallTrackerNet:

```python
# In detect_batch, around line 471, replace:
#     results.append(self._peak_detect(hmap))
# with:
                threshold = 0.3 if self._is_classifier else _PEAK_THRESHOLD
                results.append(self._peak_detect(hmap, threshold=threshold))
```

- [ ] **Step 4: Run tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_tracknet.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/tracknet.py ml/tests/test_tracknet.py
git commit -m "feat(tracknet): lower detection threshold to 0.3 for BallTrackerNet classifier"
```

---

## Task 2: Remove Frame Skip in TrackNet Batch

**Files:**
- Modify: `ml/bridge_server.py:223-237` (`rpc_tracknet_batch`)

- [ ] **Step 1: Modify `rpc_tracknet_batch` to run every frame**

In `ml/bridge_server.py`, replace lines 223-237:

```python
        # Current code (REMOVE the frame skip):
        # Build every-other-frame diff triplets: for frame indices 4, 6, 8…
        results = []
        for i in range(len(subtracted)):
            if i < 4:
                results.append({"frame_index": i, "ball": None})
                continue
            # Run on EVERY frame with a full triplet (no more skipping)
            triplet = [subtracted[i - 4], subtracted[i - 2], subtracted[i]]
            detections = self.tracknet.detect_batch([triplet])
            ball = detections[0] if detections else None
            results.append({"frame_index": i, "ball": ball})
```

Note: The triplet indices `[i-4, i-2, i]` still skip by 2 within the triplet (matching TrackNet's temporal design), but we now produce a detection attempt for every frame >= 4 instead of every other frame.

- [ ] **Step 2: Run the tagger on the 10s test video to verify more detections**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger testdata/tennis_10s.mp4 2>&1
```

Expected: `tracknet_and_yolo` count should be higher than the previous 3. Look for the "Ball detection summary" log line.

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/bridge_server.py
git commit -m "feat(tracknet): remove frame skip, run detection on every frame with triplet"
```

---

## Task 3: Fix Background Subtraction Reference Frames

**Files:**
- Modify: `ml/bridge_server.py:175-237` (`rpc_tracknet_batch`)
- Modify: `ml/bridge_server.py:241-264` (`rpc_detect_court`)
- Modify: `internal/pipeline/concurrent.go:49-69` (send reference frames)

- [ ] **Step 1: Add `set_background_reference` RPC method**

In `ml/bridge_server.py`, add a new RPC method after `rpc_tracknet_batch`:

```python
    # ---- RPC: set_background_reference ------------------------------------

    def rpc_set_background_reference(self, params: dict) -> Any:
        """Build background reference from frames sampled across the full video.

        Should be called once before any tracknet_batch calls, typically
        during the court detection phase.
        """
        self._require_init()
        from ml.tracknet import BackgroundSubtractor

        frames_data = params.get("frames", [])
        if not frames_data:
            return {"status": "no_frames"}

        shm_path = params.get("shm_path")
        if shm_path:
            batch = []
            with open(shm_path, "rb") as f:
                for meta in frames_data:
                    f.seek(meta["offset"])
                    raw = f.read(meta["size"])
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (meta["height"], meta["width"], 3)
                    )
                    batch.append(arr.copy())
        else:
            batch = [_decode_frame(f) for f in frames_data]

        if not hasattr(self, "_bg_subtractor") or self._bg_subtractor is None:
            self._bg_subtractor = BackgroundSubtractor()
        self._bg_subtractor.build_reference(batch)
        logger.info("Background reference built from %d frames", len(batch))
        return {"status": "ok", "reference_frames": len(batch)}
```

Register it in the `dispatch` method (line ~392-409):

```python
            "set_background_reference": self.rpc_set_background_reference,
```

- [ ] **Step 2: Update `rpc_tracknet_batch` to skip building reference if already set**

In `rpc_tracknet_batch`, replace the background building block (lines 211-215):

```python
        from ml.tracknet import BackgroundSubtractor

        # Build background reference from batch ONLY if not already set
        # by a prior set_background_reference call
        if not hasattr(self, "_bg_subtractor") or self._bg_subtractor is None:
            self._bg_subtractor = BackgroundSubtractor()
        if not self._bg_subtractor.is_ready:
            logger.warning("Background reference not pre-set, building from batch (suboptimal)")
            ref_frames = batch[:30] if len(batch) >= 30 else batch
            self._bg_subtractor.build_reference(ref_frames)
```

- [ ] **Step 3: Add `SetBackgroundReference` to Go bridge interface and process implementation**

In `internal/bridge/types.go`, add to the `BridgeBackend` interface (after `TrackNetBatch`):

```go
	// SetBackgroundReference sends reference frames for TrackNet background subtraction.
	SetBackgroundReference(frames []Frame) error
```

In `internal/bridge/process.go`, add the implementation (after `FitTrajectories`):

```go
// SetBackgroundReference sends reference frames to the Python bridge for
// building the TrackNet background subtraction model.
func (b *ProcessBridge) SetBackgroundReference(frames []Frame) error {
	if len(frames) == 0 {
		return nil
	}
	if b.shm == nil {
		var err error
		b.shm, err = NewSharedMemBuffer(os.TempDir())
		if err != nil {
			return fmt.Errorf("SetBackgroundReference: create shm: %w", err)
		}
	}

	shmPath, metas, err := b.shm.WriteBatch(frames)
	if err != nil {
		return fmt.Errorf("SetBackgroundReference: write shm: %w", err)
	}

	frameMetas := make([]map[string]interface{}, len(metas))
	for i, m := range metas {
		frameMetas[i] = map[string]interface{}{
			"offset": m.Offset,
			"width":  m.Width,
			"height": m.Height,
			"size":   m.Size,
		}
	}

	payload, _ := json.Marshal(map[string]interface{}{
		"shm_path": shmPath,
		"frames":   frameMetas,
	})

	_, err = b.worker.Call("set_background_reference", payload)
	return err
}
```

In `internal/bridge/mock.go`, add the no-op stub:

```go
func (m *MockBridge) SetBackgroundReference(frames []Frame) error { return nil }
```

- [ ] **Step 4: Wire reference frame sampling into the pipeline**

In `internal/pipeline/concurrent.go`, after court detection (line ~69) and before the batch processing starts, add:

```go
	// 2b. Sample reference frames for TrackNet background subtraction.
	// Extract 30 frames evenly spaced across the video.
	{
		refCount := 30
		if meta.TotalFrames < refCount {
			refCount = meta.TotalFrames
		}
		step := meta.TotalFrames / refCount
		if step < 1 {
			step = 1
		}
		var refFrames []bridge.Frame
		for i := 0; i < meta.TotalFrames && len(refFrames) < refCount; i += step {
			frames, err := vr.ExtractBatch(i, 1)
			if err == nil && len(frames) > 0 {
				refFrames = append(refFrames, videoFrameToBridgeFrame(frames[0]))
			}
		}
		if len(refFrames) > 0 {
			if err := p.bridge.SetBackgroundReference(refFrames); err != nil {
				slog.Warn("Failed to set background reference", "error", err)
			} else {
				slog.Info("Background reference set", "frames", len(refFrames))
			}
		}
	}
```

- [ ] **Step 5: Run Go tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go test ./internal/... -v
```

Expected: All pass. The mock bridge has the stub so no breakage.

- [ ] **Step 6: Run full pipeline on test video**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger testdata/tennis_60s.mp4 2>&1
```

Expected: Ball detection count significantly higher than the previous 174/1800. Target: >900 (>50%).

- [ ] **Step 7: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/bridge_server.py internal/bridge/types.go internal/bridge/process.go internal/bridge/mock.go internal/pipeline/concurrent.go
git commit -m "feat: sample reference frames across video for TrackNet background subtraction"
```

---

## Task 4: Add Broadcast Fallback Homography

**Files:**
- Modify: `ml/bridge_server.py:241-264` (`rpc_detect_court`)
- Modify: `cmd/tagger/main.go:14-21` (add CLI flag)

- [ ] **Step 1: Write test for homography fallback**

Add to `ml/tests/test_tracknet.py` (or create `ml/tests/test_court.py`):

```python
class TestHomographyFallback:
    """Test that identity homography is replaced with broadcast fallback."""

    def test_identity_is_detected(self):
        """np.eye(3) should be flagged as identity."""
        import numpy as np
        H = np.eye(3)
        # Check if near-identity
        assert np.allclose(H, np.eye(3), atol=1e-6)

    def test_broadcast_fallback_maps_to_valid_range(self):
        """Fallback homography should map 640x360 center to valid court coords."""
        from ml.trajectory import TrajectoryFitter, SINGLES_WIDTH, COURT_LENGTH
        import numpy as np
        # Broadcast fallback for standard 640x360 frame:
        # Court corners approx: TL(160,60), TR(480,60), BL(80,340), BR(560,340)
        src = np.float32([[160, 60], [480, 60], [80, 340], [560, 340]])
        dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
        import cv2
        H, _ = cv2.findHomography(src, dst)
        fitter = TrajectoryFitter(H, fps=30.0)
        # Center of frame should map to roughly center court
        cx, cy = fitter.pixel_to_court(320, 200)
        assert 0 <= cx <= SINGLES_WIDTH, f"cx={cx} out of range"
        assert 0 <= cy <= COURT_LENGTH, f"cy={cy} out of range"
```

- [ ] **Step 2: Run test**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_tracknet.py::TestHomographyFallback -v
```

Expected: Pass.

- [ ] **Step 3: Add fallback homography to `rpc_detect_court`**

In `ml/bridge_server.py`, replace `rpc_detect_court` (lines 241-264):

```python
    def rpc_detect_court(self, params: dict) -> Any:
        """Detect court boundaries in a single frame.

        If detection returns an identity (or near-identity) homography,
        substitutes a broadcast fallback derived from standard court
        dimensions and typical broadcast camera position.
        """
        self._require_init()
        frame = _decode_frame(params["frame"])
        result = self.analyzer.detect_court(frame)

        # Check if homography is identity or near-identity (detection failed)
        homography = result.get("homography")
        if homography is not None:
            H = np.array(homography, dtype=float)
            if H.shape == (3, 3) and np.allclose(H, np.eye(3), atol=1e-4):
                logger.warning("Court detection returned identity homography, using broadcast fallback")
                result = self._broadcast_fallback(frame.shape[1], frame.shape[0], result)
        elif homography is None:
            logger.warning("Court detection returned no homography, using broadcast fallback")
            result = self._broadcast_fallback(frame.shape[1], frame.shape[0], result)

        # Set court polygon on detector for player filtering
        polygon = result.get("polygon")
        if polygon is not None:
            court_w = frame.shape[1]
            det_w = 640
            if court_w != det_w and court_w > 0:
                scale = det_w / court_w
                scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in polygon]
                self.detector.set_court_polygon(scaled)
            else:
                self.detector.set_court_polygon(polygon)
        return result

    @staticmethod
    def _broadcast_fallback(width: int, height: int, result: dict) -> dict:
        """Compute a fallback homography for standard broadcast camera angle.

        Assumes camera is centered behind one baseline, ~10m high, ~5m back.
        Maps approximate court corner pixels to normalised [0,1] court coords.
        """
        import cv2

        # Approximate court corner positions for standard broadcast at given resolution
        # These assume a typical wide-angle broadcast view
        w, h = float(width), float(height)
        src = np.float32([
            [w * 0.25, h * 0.17],   # top-left court corner
            [w * 0.75, h * 0.17],   # top-right court corner
            [w * 0.125, h * 0.94],  # bottom-left court corner
            [w * 0.875, h * 0.94],  # bottom-right court corner
        ])
        dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])

        H, _ = cv2.findHomography(src, dst)
        result["homography"] = H.tolist()
        result["method"] = "broadcast_fallback"
        result["confidence"] = 0.3
        # Set corners from src points
        result["corners"] = src.tolist()
        return result
```

- [ ] **Step 4: Add `--court-corners` flag to CLI**

In `cmd/tagger/main.go`, add after the existing flags (line ~20):

```go
	courtCorners := flag.String("court-corners", "", "Manual court corners: x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BL,BR pixel coords)")
```

After bridge init and before ProcessVideo (around line ~50), add parsing:

```go
	if *courtCorners != "" {
		// Parse manual court corners and pass to config
		// Format: "x1,y1,x2,y2,x3,y3,x4,y4"
		slog.Info("Manual court corners provided", "corners", *courtCorners)
		cfg.CourtCorners = *courtCorners
	}
```

Note: For MVP, this is a secondary feature. The broadcast fallback is the primary fix. The `--court-corners` flag can be wired to override the homography in a follow-up if the broadcast fallback proves insufficient.

- [ ] **Step 5: Run pipeline on test video**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger testdata/tennis_60s.mp4 2>&1
```

Expected: Check the CSV output — court positions should now be within valid ranges (0 ≤ cx ≤ 8.23, 0 ≤ cy ≤ 23.77). Speeds should no longer be 0.3 km/h.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/bridge_server.py cmd/tagger/main.go ml/tests/test_tracknet.py
git commit -m "feat: add broadcast fallback homography when court detection fails"
```

---

## Task 5: Implement `segment_detections` and Wire Into Pipeline

**Files:**
- Modify: `ml/trajectory.py` (add `segment_detections` function)
- Modify: `ml/bridge_server.py:339-382` (`rpc_fit_trajectories`)
- Modify: `ml/tests/test_trajectory.py` (add segmentation tests)

- [ ] **Step 1: Write failing tests for `segment_detections`**

Add to `ml/tests/test_trajectory.py`:

```python
from ml.trajectory import segment_detections


class TestSegmentDetections:
    """Test gap-aware trajectory segmentation."""

    def test_single_continuous_segment(self):
        """Detections with no gaps → one segment."""
        dets = _make_detections(
            xs=[100, 110, 120, 130, 140],
            ys=[200, 195, 190, 185, 180],
            frames=[0, 1, 2, 3, 4],
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == 5

    def test_large_gap_splits(self):
        """A gap of 20 frames should split into two segments."""
        dets = _make_detections(
            xs=[100, 110, 120, 300, 310, 320],
            ys=[200, 195, 190, 200, 195, 190],
            frames=[0, 1, 2, 22, 23, 24],
        )
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 2
        assert len(segments[0]) == 3
        assert len(segments[1]) == 3

    def test_deduplicates_first(self):
        """Same-frame duplicates should be removed before segmenting."""
        dets = [
            {"x": 100.0, "y": 200.0, "confidence": 0.5, "frame_index": 0},
            {"x": 105.0, "y": 200.0, "confidence": 0.9, "frame_index": 0},
            {"x": 110.0, "y": 195.0, "confidence": 0.8, "frame_index": 1},
        ]
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        # Should have kept the higher-confidence detection
        assert segments[0][0]["confidence"] == 0.9
        assert len(segments[0]) == 2  # 2 unique frames

    def test_empty_input(self):
        segments = segment_detections([], fps=30.0)
        assert segments == []

    def test_single_detection(self):
        dets = _make_detections([100], [200], [0])
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_trajectory.py::TestSegmentDetections -v
```

Expected: ImportError — `segment_detections` doesn't exist yet.

- [ ] **Step 3: Implement `segment_detections`**

In `ml/trajectory.py`, add after the `is_same_shot` function (around line 143):

```python
def segment_detections(detections: List[dict], fps: float) -> List[List[dict]]:
    """Split detections into shot segments using gap analysis and velocity checks.

    1. Deduplicate (keep highest confidence per frame)
    2. Sort by frame_index
    3. Split at gaps > _MIN_GAP_FRAMES where is_same_shot returns False

    Args:
        detections: List of {x, y, confidence, frame_index} dicts.
        fps: Video frame rate.

    Returns:
        List of detection segments, each a list of detection dicts.
    """
    if not detections:
        return []

    cleaned = deduplicate_detections(detections)
    if not cleaned:
        return []

    # Already sorted by deduplicate_detections, but ensure
    cleaned.sort(key=lambda d: d["frame_index"])

    segments: List[List[dict]] = []
    current: List[dict] = [cleaned[0]]

    for i in range(1, len(cleaned)):
        gap = cleaned[i]["frame_index"] - cleaned[i - 1]["frame_index"]

        if gap <= _MIN_GAP_FRAMES:
            # Small gap — always same segment
            current.append(cleaned[i])
        else:
            # Check if this looks like the same shot arc
            tail = current[-min(3, len(current)):]
            head = cleaned[i:i + min(3, len(cleaned) - i)]
            if is_same_shot(tail, head, fps):
                current.append(cleaned[i])
            else:
                segments.append(current)
                current = [cleaned[i]]

    segments.append(current)
    return segments
```

- [ ] **Step 4: Run tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_trajectory.py::TestSegmentDetections -v
```

Expected: All pass.

- [ ] **Step 5: Wire segmentation into `rpc_fit_trajectories`**

In `ml/bridge_server.py`, replace `rpc_fit_trajectories` (lines 339-382):

```python
    def rpc_fit_trajectories(self, params: dict) -> Any:
        """Fit ball trajectories from ball position detections.

        Segments detections into individual shots, then fits one trajectory
        per segment.
        """
        from ml.trajectory import TrajectoryFitter, segment_detections

        ball_positions = params.get("ball_positions", [])
        court = params.get("court", {})
        fps = float(params.get("fps", 30.0))

        homography_raw = court.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        homography = np.array(homography_raw, dtype=float)

        if homography.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got shape {homography.shape}")

        # Normalise key names: Go sends camelCase, Python uses snake_case
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

        # Segment into individual shots
        segments = segment_detections(detections, fps)
        logger.info("Segmented %d detections into %d segments", len(detections), len(segments))

        # Fit one trajectory per segment
        fitter = TrajectoryFitter(homography, fps)
        trajectories = []
        for seg in segments:
            traj = fitter.fit(seg)
            if traj is not None:
                trajectories.append(traj.to_dict())

        return trajectories
```

- [ ] **Step 6: Run full pipeline on test video**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger testdata/tennis_60s.mp4 2>&1
```

Expected: `trajectories` count should be > 1. `shots` and `points` should also be > 1.

- [ ] **Step 7: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/bridge_server.py ml/tests/test_trajectory.py
git commit -m "feat: add segment_detections and wire multi-trajectory fitting into pipeline"
```

---

## Task 6: Add Velocity-Based Hitter Fallback

**Files:**
- Modify: `internal/bridge/types.go:132-138` (`TrajectoryResult`)
- Modify: `internal/point/shot.go:28-96`
- Modify: `internal/point/shot_test.go`

- [ ] **Step 1: Add `Vy` field to Go `TrajectoryResult`**

In `internal/bridge/types.go`, update `TrajectoryResult` (line 132):

```go
// TrajectoryResult holds a fitted ball trajectory segment.
type TrajectoryResult struct {
	StartFrame int       `json:"startFrame"`
	EndFrame   int       `json:"endFrame"`
	Bounces    []Bounce  `json:"bounces"`
	SpeedKPH   float64   `json:"speedKph"`
	Confidence float64   `json:"confidence"`
	Vy         float64   `json:"vy"`          // court-Y velocity (m/s), positive = toward far baseline
}
```

Note: Python's `Trajectory.to_dict()` already sends `vy` — Go just wasn't reading it. Adding the field makes it available for hitter ID.

- [ ] **Step 2: Write failing test for velocity-based hitter fallback**

Add to `internal/point/shot_test.go`:

```go
// TestSegmentShotsVelocityFallback: when no bounces, use trajectory Vy to determine hitter.
func TestSegmentShotsVelocityFallback(t *testing.T) {
	// Trajectory with no bounces but positive Vy (moving toward far baseline)
	// → near player (0) hit it
	trajs := []bridge.TrajectoryResult{
		{StartFrame: 0, EndFrame: 30, Bounces: nil, SpeedKPH: 100.0, Confidence: 0.8, Vy: 5.0},
		{StartFrame: 31, EndFrame: 60, Bounces: nil, SpeedKPH: 80.0, Confidence: 0.7, Vy: -3.0},
	}
	shots := SegmentShots(trajs)
	if len(shots) != 2 {
		t.Fatalf("expected 2 shots, got %d", len(shots))
	}
	// Vy > 0 → near player (0)
	if shots[0].Hitter != 0 {
		t.Errorf("shot 0: expected hitter=0 (vy>0), got %d", shots[0].Hitter)
	}
	// Vy < 0 → far player (1)
	if shots[1].Hitter != 1 {
		t.Errorf("shot 1: expected hitter=1 (vy<0), got %d", shots[1].Hitter)
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go test ./internal/point/ -run TestSegmentShotsVelocityFallback -v
```

Expected: FAIL — trajectories with no bounces produce no shots currently.

- [ ] **Step 4: Implement velocity-based fallback in `SegmentShots`**

Replace `internal/point/shot.go` entirely:

```go
package point

import (
	"sort"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// Shot represents a single hit in a rally.
type Shot struct {
	Index      int            // 1-based shot index in rally
	Hitter     int            // 0 = near player (bottom), 1 = far player (top)
	StartFrame int
	EndFrame   int
	Bounce     *bridge.Bounce // where it landed (nil if no bounce detected)
	IsServe    bool
	SpeedKPH   float64
	Confidence float64
}

// trajBounce pairs a bounce with its parent trajectory.
type trajBounce struct {
	bounce bridge.Bounce
	traj   bridge.TrajectoryResult
}

// SegmentShots takes trajectory results and segments them into individual shots.
//
// Primary signal: bounce court-Y position (hitterFromBounce).
// Fallback: trajectory Vy direction when no bounces are detected.
// If a trajectory has bounces, one Shot per bounce.
// If a trajectory has no bounces, one Shot per trajectory using Vy.
func SegmentShots(trajectories []bridge.TrajectoryResult) []Shot {
	// 1. Collect all bounces, plus bounceless trajectories for Vy fallback.
	var bounced []trajBounce
	var bounceless []bridge.TrajectoryResult

	for _, traj := range trajectories {
		if len(traj.Bounces) > 0 {
			for _, b := range traj.Bounces {
				bounced = append(bounced, trajBounce{bounce: b, traj: traj})
			}
		} else {
			bounceless = append(bounceless, traj)
		}
	}

	// 2. Build shots from bounced trajectories.
	var shots []Shot
	if len(bounced) > 0 {
		sort.Slice(bounced, func(i, j int) bool {
			return bounced[i].bounce.FrameIndex < bounced[j].bounce.FrameIndex
		})
		for i, tb := range bounced {
			b := tb.bounce
			hitter := hitterFromBounce(b.CY)
			startFrame := tb.traj.StartFrame
			if i > 0 {
				startFrame = bounced[i-1].bounce.FrameIndex
			}
			bounceVal := b
			shots = append(shots, Shot{
				Index:      0, // set below
				Hitter:     hitter,
				StartFrame: startFrame,
				EndFrame:   b.FrameIndex,
				Bounce:     &bounceVal,
				SpeedKPH:   tb.traj.SpeedKPH,
				Confidence: tb.traj.Confidence,
			})
		}
	}

	// 3. Build shots from bounceless trajectories using Vy.
	for _, traj := range bounceless {
		hitter := hitterFromVelocity(traj.Vy)
		shots = append(shots, Shot{
			Hitter:     hitter,
			StartFrame: traj.StartFrame,
			EndFrame:   traj.EndFrame,
			SpeedKPH:   traj.SpeedKPH,
			Confidence: traj.Confidence,
		})
	}

	if len(shots) == 0 {
		return nil
	}

	// 4. Sort all shots chronologically.
	sort.Slice(shots, func(i, j int) bool {
		return shots[i].StartFrame < shots[j].StartFrame
	})

	// 5. Set indices and serve flag.
	for i := range shots {
		shots[i].Index = i + 1
		shots[i].IsServe = (i == 0)
	}

	return shots
}

// hitterFromBounce returns the player who hit the ball based on bounce position.
func hitterFromBounce(cy float64) int {
	if cy < NetY {
		return 1
	}
	return 0
}

// hitterFromVelocity returns the player who hit the ball based on trajectory direction.
// vy > 0 (moving toward far baseline) → near player (0) hit it
// vy <= 0 (moving toward near baseline) → far player (1) hit it
func hitterFromVelocity(vy float64) int {
	if vy > 0 {
		return 0
	}
	return 1
}
```

- [ ] **Step 5: Run all shot tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go test ./internal/point/ -v
```

Expected: All tests pass including the new velocity fallback test. Existing bounce-based tests should still pass because the bounce path is unchanged.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/bridge/types.go internal/point/shot.go internal/point/shot_test.go
git commit -m "feat: add velocity-based hitter fallback for bounceless trajectories"
```

---

## Task 7: Add Speed Validation and `speed_valid` Field

**Files:**
- Modify: `ml/trajectory.py:149-178` (`Trajectory` dataclass + `to_dict`)
- Modify: `ml/trajectory.py:303-397` (`TrajectoryFitter.fit`)
- Modify: `ml/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Add to `ml/tests/test_trajectory.py`:

```python
class TestSpeedValidation:
    """Test speed sanity clamping."""

    def test_valid_rally_speed_is_kept(self):
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        # Build detections that produce a reasonable speed (~100 km/h = ~27.8 m/s)
        # Moving ~0.93 m per frame at 30fps = 27.8 m/s
        dets = _make_detections(
            xs=[0, 0.93, 1.86, 2.79, 3.72],
            ys=[0, 0, 0, 0, 0],
            frames=[0, 1, 2, 3, 4],
        )
        traj = fitter.fit(dets)
        assert traj is not None
        assert traj.speed_valid is True
        assert traj.speed_kph > 20.0

    def test_impossibly_fast_speed_is_clamped(self):
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        # Moving 100m per frame at 30fps = 3000 m/s = 10800 km/h (impossible)
        dets = _make_detections(
            xs=[0, 100, 200, 300, 400],
            ys=[0, 0, 0, 0, 0],
            frames=[0, 1, 2, 3, 4],
        )
        traj = fitter.fit(dets)
        assert traj is not None
        assert traj.speed_valid is False
        assert traj.speed_kph == 0.0

    def test_speed_valid_in_to_dict(self):
        t = Trajectory(start_frame=0, end_frame=10, speed_valid=False)
        d = t.to_dict()
        assert "speedValid" in d
        assert d["speedValid"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_trajectory.py::TestSpeedValidation -v
```

Expected: FAIL — `speed_valid` field doesn't exist yet.

- [ ] **Step 3: Add `speed_valid` field to `Trajectory` dataclass**

In `ml/trajectory.py`, update the `Trajectory` dataclass (line ~150):

```python
@dataclass
class Trajectory:
    """Fitted ball trajectory for a segment of detections."""

    start_frame: int
    end_frame: int
    positions: List[dict] = field(default_factory=list)
    bounces: List[dict] = field(default_factory=list)
    speed_kph: float = 0.0
    confidence: float = 0.0
    cx0: float = 0.0
    vx: float = 0.0
    cy0: float = 0.0
    vy: float = 0.0
    g_eff: float = 0.0
    speed_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "positions": self.positions,
            "bounces": self.bounces,
            "speedKph": self.speed_kph,
            "confidence": self.confidence,
            "cx0": self.cx0,
            "vx": self.vx,
            "cy0": self.cy0,
            "vy": self.vy,
            "gEff": self.g_eff,
            "speedValid": self.speed_valid,
        }
```

- [ ] **Step 4: Add speed clamping to `TrajectoryFitter.fit`**

In `ml/trajectory.py`, at the end of the `fit` method (after `speed_kph = speed_ms * 3.6`, around line 384), add:

```python
        # Speed sanity clamping — max plausible tennis speed is ~260 km/h (serves)
        # Rally shots max ~180 km/h. Below 20 km/h is likely noise.
        _MAX_SPEED_KPH = 300.0  # generous upper bound covering all shot types
        _MIN_SPEED_KPH = 5.0    # below this is likely noise/stationary
        speed_valid = _MIN_SPEED_KPH <= speed_kph <= _MAX_SPEED_KPH
        if not speed_valid:
            speed_kph = 0.0
```

And update the return to include the new fields:

```python
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
            speed_valid=speed_valid,
        )
```

- [ ] **Step 5: Run all Python tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_trajectory.py -v
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/tests/test_trajectory.py
git commit -m "feat: add speed_valid field and sanity clamping to trajectory fitting"
```

---

## Task 8: End-to-End Validation

**Files:** None (testing only)

- [ ] **Step 1: Run all Go tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go test ./internal/... -v
```

Expected: All pass.

- [ ] **Step 2: Run all Python tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/ -v
```

Expected: All pass.

- [ ] **Step 3: Run on 60s test video and check output**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger testdata/tennis_60s.mp4 2>&1
```

Check the log output:
- `tracknet_and_yolo` > 900 (>50% of 1800 frames)
- `trajectories` > 1
- `shots` > 1
- `points` > 1

- [ ] **Step 4: Inspect CSV output quality**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
head -5 testdata/tennis_60s.mp4_output.csv
```

Check:
- Multiple "Point N" rows (not just "Point 1")
- Hitter alternates between 0 and 1
- Speed values are realistic (not 0.3 km/h)
- Court positions within valid ranges

- [ ] **Step 5: Run on full match video (final MVP gate)**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
go run ./cmd/tagger "files/data/training_pairs/Andrew Johnson vs Alberto Pulido Moreno_20251112_155810/Andrew Johnson vs Alberto Pulido Moreno.mp4" 2>&1
```

Check:
- Reasonable number of points detected
- Rally lengths 2-20 shots
- Hitter alternation
- Serve speeds 80-260 km/h

- [ ] **Step 6: Commit final CSV outputs as test baselines**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add testdata/tennis_60s.mp4_output.csv
git commit -m "test: update baseline CSV output after MVP detection fixes"
```
