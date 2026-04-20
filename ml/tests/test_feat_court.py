import numpy as np
from ml.feature_extractor.court import project_to_court

def test_project_to_court_identity_preserves_points():
    H = np.eye(3, dtype=np.float32)
    px = np.array([[[1.0, 2.0], [3.0, 4.0]]], np.float32)  # (1, 2, 2)
    out = project_to_court(px, H)
    np.testing.assert_allclose(out, px, atol=1e-5)

def test_project_to_court_shape_preserved():
    H = np.eye(3, dtype=np.float32)
    px = np.zeros((5, 2, 17, 2), np.float32)
    out = project_to_court(px, H)
    assert out.shape == px.shape
