import numpy as np
from ml.point_model.labels import build_targets, Targets

def test_weak_only():
    t = build_targets(
        T=60, fps=30, stroke_count=3, stroke_types=["Serve","Forehand","Backhand"],
        winner_or_error="Winner", point_won_by="P1", server="P1",
        strong_contact_frames=None,
    )
    assert isinstance(t, Targets)
    assert t.contact_frames.shape == (60,)
    assert t.contact_frames.sum() == 3
    assert t.stroke_idx.tolist()[:3] == [2, 0, 1]
    assert t.outcome_idx == 2
    assert t.contact_strong is False

def test_strong_overrides_weak():
    t = build_targets(
        T=60, fps=30, stroke_count=2, stroke_types=["Serve","Forehand"],
        winner_or_error="Ace", point_won_by="P1", server="P1",
        strong_contact_frames=[(5, 0), (45, 1)],
    )
    assert t.contact_strong is True
    assert int(t.contact_frames[5]) == 1 and int(t.contact_frames[45]) == 1
    assert int(t.hitter_per_frame[5]) == 0 and int(t.hitter_per_frame[45]) == 1
