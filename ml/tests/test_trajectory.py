"""Tests for ml.trajectory module."""

import numpy as np
import pytest

from ml.trajectory import TrajectoryFitter, deduplicate_detections


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
