import numpy as np
from ml.point_model.features import build_feature_tensor, FEATURE_DIM

def test_feature_dim():
    T = 30
    feats = build_feature_tensor(
        pose_px=np.zeros((T, 2, 17, 2), np.float32),
        pose_court=np.zeros((T, 2, 17, 2), np.float32),
        pose_conf=np.zeros((T, 2, 17), np.float32),
        pose_valid=np.zeros((T, 2), np.bool_),
        ball=np.zeros((T, 3), np.float32),
        audio_mel=np.zeros((64, T*2), np.float32),
        clip_meta=np.array([1.0, 1280, 720, 30.0], np.float32),
    )
    assert feats.shape == (T, FEATURE_DIM)
