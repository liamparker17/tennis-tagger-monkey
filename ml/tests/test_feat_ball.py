import pytest
from pathlib import Path
from ml.feature_extractor.ball import BallDetector
from ml.feature_extractor.frames import iter_frames

WEIGHTS = Path("models/yolov8s.pt")
SAMPLE = Path("testdata/sample_a.mp4")


@pytest.mark.skipif(not WEIGHTS.exists() or not SAMPLE.exists(), reason="needs model + sample")
def test_ball_detect_returns_3tuple_floats():
    det = BallDetector(WEIGHTS)
    frame = next(iter_frames(SAMPLE))
    result = det.detect(frame)
    assert isinstance(result, tuple) and len(result) == 3
    assert all(isinstance(v, float) for v in result)
