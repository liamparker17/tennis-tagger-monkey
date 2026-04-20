import json, numpy as np
from pathlib import Path
from ml.point_model.dataset import ClipDataset

def _make_match(root: Path, name: str, features_root: Path):
    d = root / name; d.mkdir(parents=True)
    labels = {"schema": 1, "match": name, "video": "x", "csv": "y", "points": [{
        "index": 0, "start_ms": 0, "duration_ms": 2000,
        "server": "P1", "returner": "P2", "surface": "", "speed_kmh": None,
        "hands": ["",""], "stroke_types": ["Serve","Forehand"], "last_shot_stroke":"Forehand",
        "stroke_count": 2, "point_won_by":"P1", "winner_or_error":"Winner", "score_state":"",
        "contact_xy": [], "placement_xy": [], "serve_bounce_xy": None, "ad_bounce_xy": None,
        "clip": "p_0001.mp4", "clip_start_s": 0.0, "clip_duration_s": 4.0,
    }]}
    (d / "labels.json").write_text(json.dumps(labels))
    feat_dir = features_root / name; feat_dir.mkdir(parents=True)
    T = 30
    np.savez_compressed(feat_dir / "p_0001.npz",
        schema=np.int32(1), fps=np.int32(30),
        pose_px=np.zeros((T,2,17,2), np.float32),
        pose_conf=np.zeros((T,2,17), np.float32),
        pose_court=np.zeros((T,2,17,2), np.float32),
        pose_valid=np.zeros((T,2), np.bool_),
        court_h=np.eye(3, dtype=np.float32),
        ball=np.zeros((T,3), np.float32),
        audio_mel=np.zeros((64, T), np.float32),
        clip_meta=np.array([1.0, 1280, 720, 30.0], np.float32),
    )

def test_dataset(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    features = tmp_path / "features"; features.mkdir()
    _make_match(clips, "m1", features)
    ds = ClipDataset(clips_root=clips, features_root=features,
                     match_filter=None, T_max=64)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["features"].shape == (64, 243)
    assert sample["mask"].shape == (64,)
    assert sample["targets"].outcome_idx == 2
