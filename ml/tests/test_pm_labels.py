import numpy as np
from ml.point_model.labels import build_targets, Targets

def test_weak_only():
    t = build_targets(
        T=60, fps=30, stroke_count_lo=3, stroke_count_hi=3,
        stroke_types=["Serve","Forehand","Backhand"],
        outcome="Winner", point_won_by="P1",
        server="P1", player_a="P1", player_b="P2",
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
        T=60, fps=30, stroke_count_lo=2, stroke_count_hi=2,
        stroke_types=["Serve","Forehand"],
        outcome="Ace", point_won_by="P1",
        server="P1", player_a="P1", player_b="P2",
        strong_contact_frames=[(5, 0), (45, 1)],
    )
    assert t.contact_strong is True
    assert int(t.contact_frames[5]) == 1 and int(t.contact_frames[45]) == 1
    assert int(t.hitter_per_frame[5]) == 0 and int(t.hitter_per_frame[45]) == 1

def test_server_name_resolves_to_slot():
    t = build_targets(
        T=30, fps=30, stroke_count_lo=1, stroke_count_hi=1,
        stroke_types=["Serve"],
        outcome="Ace", point_won_by="Bob",
        server="Bob", player_a="Alice", player_b="Bob",
        strong_contact_frames=None,
    )
    assert t.server_is_p1 is False

def test_semantic_stroke_mapping():
    t = build_targets(
        T=60, fps=30, stroke_count_lo=3, stroke_count_hi=3,
        stroke_types=["1st Serve Made", "Backhand Return Made", "Forehand Volley"],
        outcome="Winner", point_won_by="P1",
        server="P1", player_a="P1", player_b="P2",
        strong_contact_frames=None,
    )
    # Serve=2, Backhand=1, Volley=3
    assert t.stroke_idx.tolist()[:3] == [2, 1, 3]
