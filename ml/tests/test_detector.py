import pytest
import numpy as np


def test_invalid_backend():
    from ml.detector import Detector

    with pytest.raises(ValueError, match="Unknown backend"):
        Detector("fake.pt", device="cpu", backend="invalid")


def test_default_backend_is_yolo():
    from ml.detector import Detector

    det = Detector("nonexistent.pt", device="cpu")
    assert det.backend == "yolo"


def test_rtdetr_backend():
    from ml.detector import Detector

    det = Detector("nonexistent.pt", device="cpu", backend="rtdetr")
    assert det.backend == "rtdetr"


def test_fallback_returns_results():
    from ml.detector import Detector

    det = Detector("nonexistent.pt", device="cpu")
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    results = det.detect_batch(frames)
    assert len(results) == 1
    assert "players" in results[0]
    assert "ball" in results[0]
