"""Tests for ml.trajectory module."""

import numpy as np
import pytest

from ml.trajectory import Trajectory, TrajectoryFitter, deduplicate_detections, is_same_shot, segment_detections


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
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        detections = _make_detections(
            xs=[1, 2, 3, 4, 5, 6, 7],
            ys=[1, 2, 3, 4, 3, 2, 1],
            frames=[0, 1, 2, 3, 4, 5, 6],
        )
        traj = fitter.fit(detections)
        assert traj is not None
        d = traj.to_dict()
        assert len(d["bounces"]) > 0, "Expected at least one bounce to test key names"
        for bounce in d["bounces"]:
            assert "frameIndex" in bounce, f"Missing camelCase 'frameIndex', got keys: {list(bounce.keys())}"
            assert "inOut" in bounce, f"Missing camelCase 'inOut', got keys: {list(bounce.keys())}"
            assert "frame_index" not in bounce, "Found snake_case 'frame_index' — should be camelCase"
            assert "in_out" not in bounce, "Found snake_case 'in_out' — should be camelCase"


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
        assert segments[0][0]["confidence"] == 0.9
        assert len(segments[0]) == 2

    def test_empty_input(self):
        segments = segment_detections([], fps=30.0)
        assert segments == []

    def test_single_detection(self):
        dets = _make_detections([100], [200], [0])
        segments = segment_detections(dets, fps=30.0)
        assert len(segments) == 1
        assert len(segments[0]) == 1


class TestSpeedValidation:
    """Test speed sanity clamping."""

    def test_valid_rally_speed_is_kept(self):
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        # Identity H maps pixels to normalised [0,1] then scales by court dims.
        # Use small increments so court-plane speed stays in valid range.
        # 0.1 pixel/frame → ~0.82m/frame at 8.23m court width → 24.7 m/s → 89 km/h
        dets = _make_detections(
            xs=[0.5, 0.6, 0.7, 0.8, 0.9],
            ys=[0.5, 0.5, 0.5, 0.5, 0.5],
            frames=[0, 1, 2, 3, 4],
        )
        traj = fitter.fit(dets)
        assert traj is not None
        assert traj.speed_valid is True
        assert traj.speed_kph > 20.0

    def test_impossibly_fast_speed_is_clamped(self):
        fitter = TrajectoryFitter(_identity_homography(), fps=30.0)
        # Moving 100m per frame at 30fps = 10800 km/h (impossible)
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
