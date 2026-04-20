import numpy as np, torch
from ml.point_model.fusion import fuse, FusedPointPrediction, detect_audio_impulses


def test_serve_prior_overrides_weak_model():
    stroke = torch.zeros(4, 9)
    stroke[0, 0] = np.log(0.40)
    stroke[0, 2] = np.log(0.30)
    contact = torch.full((60,), -3.0); contact[10] = 2.0; contact[40] = 2.0
    hitter = torch.zeros(60, 2); hitter[10, 0] = 2.0; hitter[40, 1] = 2.0
    inout = torch.zeros(4); outcome = torch.zeros(5)
    audio_impulse = np.zeros(60, np.bool_); audio_impulse[10] = True; audio_impulse[40] = True
    pose_court = np.zeros((60, 2, 17, 2), np.float32)

    out = fuse(contact, hitter, stroke, inout, outcome,
               audio_impulse=audio_impulse, pose_court=pose_court, fps=30)
    assert isinstance(out, FusedPointPrediction)
    assert out.strokes[0].stroke == "Serve"
    assert out.strokes[0].hitter == 0
    assert out.strokes[1].hitter == 1


def test_ace_requires_single_shot():
    stroke = torch.zeros(4, 9); stroke[0, 2] = 5.0
    contact = torch.full((60,), -3.0); contact[10] = 5.0; contact[40] = 5.0
    hitter = torch.zeros(60, 2); hitter[10, 0] = 5.0; hitter[40, 1] = 5.0
    outcome = torch.full((5,), -3.0); outcome[0] = 4.0
    inout = torch.zeros(4)
    out = fuse(contact, hitter, stroke, inout, outcome,
               audio_impulse=np.zeros(60, np.bool_),
               pose_court=np.zeros((60, 2, 17, 2), np.float32), fps=30)
    assert out.outcome != "Ace"


def test_impulse_resamples_to_T():
    mel = np.zeros((64, 200), np.float32); mel[40:60, 100] = 5.0
    out = detect_audio_impulses(mel, T=60)
    assert out.shape == (60,)
    assert out.any()
