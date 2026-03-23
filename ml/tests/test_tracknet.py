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
