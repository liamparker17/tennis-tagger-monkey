"""Tests for ml.trajectory module."""

import numpy as np
import pytest

from ml.trajectory import TrajectoryFitter, deduplicate_detections, is_same_shot


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
