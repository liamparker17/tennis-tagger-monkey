import torch
from ml.point_model.model import PointModel, PointModelConfig

def test_forward_shapes():
    cfg = PointModelConfig()
    m = PointModel(cfg)
    x = torch.zeros(2, 64, 243)
    mask = torch.ones(2, 64)
    out = m(x, mask)
    assert out["contact_logits"].shape == (2, 64)
    assert out["bounce_logits"].shape == (2, 64)
    assert out["hitter_per_frame_logits"].shape == (2, 64, 2)
    assert out["stroke_logits"].shape == (2, 4, 9)
    assert out["inout_logits"].shape == (2, 4)
    assert out["outcome_logits"].shape == (2, 5)
