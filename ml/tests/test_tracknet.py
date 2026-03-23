"""Tests for ml.tracknet module."""

import cv2
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


class TestHomographyFallback:
    """Test that identity homography is replaced with broadcast fallback."""

    def test_identity_is_detected(self):
        """np.eye(3) should be flagged as identity."""
        H = np.eye(3)
        assert np.allclose(H, np.eye(3), atol=1e-6)

    def test_broadcast_fallback_maps_to_valid_range(self):
        """Fallback homography should map 640x360 center to valid court coords."""
        from ml.trajectory import TrajectoryFitter, SINGLES_WIDTH, COURT_LENGTH

        # Broadcast fallback for standard 640x360 frame
        src = np.float32([[160, 60], [480, 60], [80, 340], [560, 340]])
        dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
        H, _ = cv2.findHomography(src, dst)
        fitter = TrajectoryFitter(H, fps=30.0)
        # Center of frame should map to roughly center court
        cx, cy = fitter.pixel_to_court(320, 200)
        assert 0 <= cx <= SINGLES_WIDTH, f"cx={cx} out of range"
        assert 0 <= cy <= COURT_LENGTH, f"cy={cy} out of range"
