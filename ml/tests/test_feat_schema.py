import numpy as np
from pathlib import Path
from ml.feature_extractor.schema import FeatureSet, save_npz, load_npz, SCHEMA_VERSION


def test_roundtrip(tmp_path: Path):
    fs = FeatureSet(
        schema=SCHEMA_VERSION, fps=30,
        pose_px=np.zeros((10, 2, 17, 2), np.float32),
        pose_conf=np.zeros((10, 2, 17), np.float32),
        pose_court=np.zeros((10, 2, 17, 2), np.float32),
        pose_valid=np.zeros((10, 2), np.bool_),
        court_h=np.eye(3, dtype=np.float32),
        ball=np.zeros((10, 3), np.float32),
        audio_mel=np.zeros((64, 200), np.float32),
        clip_meta=np.zeros((4,), np.float32),
    )
    p = tmp_path / "c.npz"
    save_npz(p, fs)
    fs2 = load_npz(p)
    assert fs2.schema == SCHEMA_VERSION
    assert fs2.pose_px.shape == (10, 2, 17, 2)
