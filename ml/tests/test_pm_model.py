import torch
from ml.point_model.model import PointModel, PointModelConfig
from ml.point_model.features import FEATURE_DIM

def test_forward_shapes():
    cfg = PointModelConfig()
    m = PointModel(cfg)
    x = torch.zeros(2, 64, FEATURE_DIM)
    mask = torch.ones(2, 64)
    out = m(x, mask)
    assert out["contact_logits"].shape == (2, 64)
    assert out["bounce_logits"].shape == (2, 64)
    assert out["hitter_per_frame_logits"].shape == (2, 64, 2)
    from ml.point_model.labels import MAX_SHOTS
    assert out["stroke_logits"].shape == (2, MAX_SHOTS, 9)
    assert out["inout_logits"].shape == (2, MAX_SHOTS)
    assert out["outcome_logits"].shape == (2, 5)
