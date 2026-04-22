import torch
from ml.point_model.inference import run_inference, PointPrediction
from ml.point_model.model import PointModel, PointModelConfig
from ml.point_model.features import FEATURE_DIM

def test_inference_contract():
    m = PointModel(PointModelConfig()).eval()
    feats = torch.zeros(1, 64, FEATURE_DIM); mask = torch.ones(1, 64)
    pred = run_inference(m, feats, mask)
    assert isinstance(pred, list) and isinstance(pred[0], PointPrediction)
    from ml.point_model.labels import MAX_SHOTS
    assert len(pred[0].strokes) <= MAX_SHOTS
    assert pred[0].outcome in {"Ace","DoubleFault","Winner","ForcedError","UnforcedError"}
