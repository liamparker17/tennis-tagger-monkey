import pytest, numpy as np
from pathlib import Path
from ml.feature_extractor.pose import PoseExtractor

POSE = Path("models/yolov8s-pose.pt")
SAMPLE = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not POSE.exists() or not SAMPLE.exists(), reason="needs model + sample")
def test_pose_shape():
    ext = PoseExtractor(POSE)
    frame = np.zeros((720, 1280, 3), np.uint8)
    px, conf = ext.extract(frame)
    assert px.shape == (2, 17, 2) and conf.shape == (2, 17)
