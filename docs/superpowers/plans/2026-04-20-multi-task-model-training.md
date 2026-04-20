# Multi-Task Model + Training — Plan 3

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Train one transformer over per-clip feature stacks (Plan 2A) + Dartfish multi-labels (Plan 1), with optional contact-frame supervision (Plan 2B), and emit a checkpoint that Plan 4 (Go pipeline) can call via inference adapter.

**Architecture:** Shared encoder → 6 heads:
1. **Contact event** — per-frame logits, hitter index (0/1) target. Strong supervision when present, masked out otherwise.
2. **Bounce event** — per-frame logits. Weak target derived from Dartfish `serve_bounce_xy` / `placement_xy` timestamps where available.
3. **Stroke type** — per-shot classification into a fixed vocab.
4. **In/out** — binary per shot.
5. **Outcome** — point-level: Ace / DoubleFault / Winner / ForcedError / UnforcedError.
6. **Hitter** — per-shot near/far slot.

**Tech stack:** PyTorch 2+, numpy. CUDA optional. Holdout = 10 matches by deterministic hash.

---

## File layout

```
ml/point_model/
  __init__.py
  vocab.py        # stroke + outcome vocabularies
  features.py     # FeatureSet → flat per-frame feature tensor (243-dim)
  labels.py       # PointLabels → per-frame & per-shot targets
  dataset.py      # ClipDataset, collate
  splits.py       # deterministic train/val by match-name hash
  model.py        # PointModel (encoder + 6 heads)
  losses.py       # masked CE, focal, regression
  metrics.py      # per-head accuracy, F1
  inference.py    # PointPrediction dataclass + run_inference
  train.py        # training loop
  eval.py         # holdout evaluation
  __main__.py     # CLI: train | eval | predict

ml/tests/
  test_pm_features.py
  test_pm_labels.py
  test_pm_dataset.py
  test_pm_model.py
  test_pm_losses.py
  test_pm_inference.py
```

Outputs:
```
files/models/point_model/<run>/best.pt
files/models/point_model/<run>/metrics.json
```

---

## Vocabularies

```python
STROKE_CLASSES = ["Forehand","Backhand","Serve","Volley","Slice","Smash","DropShot","Lob","Other"]
OUTCOME_CLASSES = ["Ace","DoubleFault","Winner","ForcedError","UnforcedError"]
```

`STROKE_PAD_INDEX = -100` (ignored by CE).

---

## Feature dimension (243)

Per frame, flatten:
- `pose_px` 2×17×2 = 68 → divided by (W, H) for normalization
- `pose_court` 2×17×2 = 68
- `pose_conf` 2×17 = 34 (used as gating, but also in the feature)
- `pose_valid` 2 = 2
- `ball` 3 = 3 (x normalized, y normalized, conf)
- `audio_mel` resampled to T frames, take 64-dim slice = 64
- `clip_meta` broadcast = 4

Total: 68+68+34+2+3+64+4 = **243**.

---

### Task 3.1 — Vocab module

**Files:** create `ml/point_model/vocab.py`.

```python
# ml/point_model/vocab.py
STROKE_CLASSES = ["Forehand","Backhand","Serve","Volley","Slice","Smash","DropShot","Lob","Other"]
OUTCOME_CLASSES = ["Ace","DoubleFault","Winner","ForcedError","UnforcedError"]
STROKE_TO_IDX = {s: i for i, s in enumerate(STROKE_CLASSES)}
OUTCOME_TO_IDX = {s: i for i, s in enumerate(OUTCOME_CLASSES)}
STROKE_PAD_INDEX = -100
NUM_STROKE = len(STROKE_CLASSES)
NUM_OUTCOME = len(OUTCOME_CLASSES)

def stroke_index(name: str) -> int:
    return STROKE_TO_IDX.get(name.strip(), STROKE_TO_IDX["Other"])

def outcome_index(label: str) -> int:
    return OUTCOME_TO_IDX.get(label.strip(), OUTCOME_TO_IDX["UnforcedError"])
```

Commit: `feat(plan3): vocabularies`.

---

### Task 3.2 — Feature builder

**Files:** create `ml/point_model/features.py`, `ml/tests/test_pm_features.py`.

- [ ] Test:

```python
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
    assert feats.shape == (T, FEATURE_DIM) == (T, 243)
```

- [ ] Implement:

```python
# ml/point_model/features.py
from __future__ import annotations
import numpy as np

FEATURE_DIM = 243

def _resample_mel_to_T(mel: np.ndarray, T: int) -> np.ndarray:
    n_mels, F = mel.shape
    if F == T: return mel.T.astype(np.float32)
    idx = np.linspace(0, F - 1, num=T).astype(np.int64) if F > 0 else np.zeros(T, np.int64)
    return mel[:, idx].T.astype(np.float32)

def build_feature_tensor(pose_px, pose_court, pose_conf, pose_valid,
                         ball, audio_mel, clip_meta) -> np.ndarray:
    T = pose_px.shape[0]
    W = max(float(clip_meta[1]), 1.0); H = max(float(clip_meta[2]), 1.0)
    px_n  = (pose_px / np.array([W, H], np.float32)).reshape(T, -1)        # 68
    crt_n = pose_court.reshape(T, -1)                                       # 68
    conf  = pose_conf.reshape(T, -1)                                        # 34
    valid = pose_valid.astype(np.float32).reshape(T, -1)                    # 2
    ball_n = np.stack([
        ball[:, 0] / W, ball[:, 1] / H, ball[:, 2]
    ], axis=1).astype(np.float32)                                           # 3
    mel = _resample_mel_to_T(audio_mel, T)                                  # 64
    meta = np.tile(clip_meta.reshape(1, -1), (T, 1)).astype(np.float32)     # 4
    out = np.concatenate([px_n, crt_n, conf, valid, ball_n, mel, meta], axis=1)
    assert out.shape[1] == FEATURE_DIM, out.shape
    return out
```

Commit: `feat(plan3): per-frame feature tensor builder (243-dim)`.

---

### Task 3.3 — Label builder (weak + strong supervision)

**Files:** create `ml/point_model/labels.py`, `ml/tests/test_pm_labels.py`.

Defines weak per-frame contact targets from `stroke_count` (uniformly spaced through clip), strong overrides from labeler JSON, per-shot stroke + hitter, point-level outcome.

- [ ] Test:

```python
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
    assert t.stroke_idx.tolist()[:3] == [2, 0, 1]   # Serve, Forehand, Backhand
    assert t.outcome_idx == 2                       # Winner
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
```

- [ ] Implement:

```python
# ml/point_model/labels.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .vocab import stroke_index, outcome_index, STROKE_PAD_INDEX

MAX_SHOTS = 4

@dataclass
class Targets:
    contact_frames: np.ndarray   # (T,) {0,1}
    hitter_per_frame: np.ndarray # (T,) {-1,0,1}; -1 = no contact
    contact_strong: bool         # True if from labeler, False if heuristic
    stroke_idx: np.ndarray       # (MAX_SHOTS,) padded with STROKE_PAD_INDEX
    hitter_per_shot: np.ndarray  # (MAX_SHOTS,) {-1,0,1}
    outcome_idx: int
    server_is_p1: bool

def _alternate_hitters(server_is_p1: bool, n: int) -> list[int]:
    near = 0 if server_is_p1 else 1
    return [near if (i % 2 == 0) else 1 - near for i in range(n)]

def build_targets(*, T: int, fps: int, stroke_count: int, stroke_types: list[str],
                  winner_or_error: str, point_won_by: str, server: str,
                  strong_contact_frames: Optional[list[tuple[int, int]]]) -> Targets:
    contact = np.zeros((T,), np.int64)
    hitter_pf = np.full((T,), -1, np.int64)
    server_is_p1 = (server.strip() == "P1")
    n = max(0, min(stroke_count, MAX_SHOTS, T))
    seq = _alternate_hitters(server_is_p1, n)

    if strong_contact_frames:
        strong = True
        for f, h in strong_contact_frames:
            if 0 <= f < T:
                contact[f] = 1; hitter_pf[f] = int(h)
    else:
        strong = False
        if n > 0:
            for i in range(n):
                f = int(round((i + 0.5) * (T / n)))
                f = max(0, min(T - 1, f))
                contact[f] = 1; hitter_pf[f] = seq[i]

    stroke_idx = np.full((MAX_SHOTS,), STROKE_PAD_INDEX, np.int64)
    hitter_ps = np.full((MAX_SHOTS,), -1, np.int64)
    for i, s in enumerate(stroke_types[:MAX_SHOTS]):
        stroke_idx[i] = stroke_index(s)
        if i < n: hitter_ps[i] = seq[i]

    return Targets(
        contact_frames=contact, hitter_per_frame=hitter_pf,
        contact_strong=strong, stroke_idx=stroke_idx,
        hitter_per_shot=hitter_ps, outcome_idx=outcome_index(winner_or_error),
        server_is_p1=server_is_p1,
    )
```

Commit: `feat(plan3): target builder (weak heuristic + strong override)`.

---

### Task 3.4 — Dataset

**Files:** create `ml/point_model/dataset.py`, `ml/tests/test_pm_dataset.py`.

Reads Plan 1's `labels.json` and Plan 2A's `<match>/p_NNNN.npz`, optionally `<match>/contact_labels.json`. Pads/truncates to fixed `T_MAX = 300` (~10 s @ 30 fps).

- [ ] Test:

```python
import json, numpy as np
from pathlib import Path
from ml.point_model.dataset import ClipDataset

def _make_match(root: Path, name: str):
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
    feat_dir = root.parent / "features" / name; feat_dir.mkdir(parents=True)
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
    (tmp_path / "features").mkdir()
    _make_match(clips, "m1")
    ds = ClipDataset(clips_root=clips, features_root=tmp_path / "features",
                     match_filter=None, T_max=64)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["features"].shape == (64, 243)
    assert sample["mask"].shape == (64,)
    assert sample["targets"].outcome_idx == 2
```

- [ ] Implement:

```python
# ml/point_model/dataset.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from .features import build_feature_tensor, FEATURE_DIM
from .labels import build_targets, Targets

T_MAX = 300

@dataclass
class Sample:
    features: np.ndarray  # (T_max, 243)
    mask: np.ndarray      # (T_max,) 1 where real
    targets: Targets
    meta: dict

class ClipDataset(Dataset):
    def __init__(self, clips_root: Path, features_root: Path,
                 match_filter: Optional[list[str]] = None, T_max: int = T_MAX):
        self.clips_root = clips_root; self.features_root = features_root
        self.T_max = T_max
        self.items: list[tuple[str, dict, Path, Optional[Path]]] = []
        for d in sorted(clips_root.iterdir()):
            if not d.is_dir(): continue
            if match_filter and d.name not in match_filter: continue
            labels_path = d / "labels.json"
            if not labels_path.exists(): continue
            doc = json.loads(labels_path.read_text())
            contact_path = d / "contact_labels.json"
            contact_doc = json.loads(contact_path.read_text()) if contact_path.exists() else None
            for p in doc["points"]:
                npz = features_root / d.name / (Path(p["clip"]).stem + ".npz")
                if not npz.exists(): continue
                self.items.append((d.name, p, npz, contact_path if contact_doc else None))
                # cache contact_doc to avoid re-reading
                p["_contact"] = (contact_doc.get("clips", {}).get(Path(p["clip"]).stem)
                                  if contact_doc else None)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        match, p, npz, _ = self.items[i]
        z = np.load(npz)
        feats = build_feature_tensor(
            pose_px=z["pose_px"], pose_court=z["pose_court"],
            pose_conf=z["pose_conf"], pose_valid=z["pose_valid"],
            ball=z["ball"], audio_mel=z["audio_mel"], clip_meta=z["clip_meta"],
        )
        T = feats.shape[0]
        Tm = self.T_max
        out = np.zeros((Tm, FEATURE_DIM), np.float32)
        mask = np.zeros((Tm,), np.float32)
        n = min(T, Tm)
        out[:n] = feats[:n]; mask[:n] = 1.0

        strong = p.get("_contact")
        strong_pairs = [(int(e["frame"]), int(e["hitter"])) for e in strong] if strong else None
        targets = build_targets(
            T=Tm, fps=int(z["fps"]),
            stroke_count=int(p["stroke_count"]),
            stroke_types=list(p["stroke_types"]),
            winner_or_error=p["winner_or_error"],
            point_won_by=p["point_won_by"], server=p["server"],
            strong_contact_frames=strong_pairs,
        )
        return {"features": out, "mask": mask, "targets": targets,
                "meta": {"match": match, "clip": p["clip"]}}
```

Commit: `feat(plan3): ClipDataset with weak/strong supervision`.

---

### Task 3.5 — Splits

**Files:** create `ml/point_model/splits.py`. Deterministic by SHA-1 of match name.

```python
# ml/point_model/splits.py
import hashlib

HOLDOUT_FRACTION = 0.10  # ≈ 10 of 100 matches

def is_holdout(match_name: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    h = int(hashlib.sha1(match_name.encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0 < fraction

def split_matches(names: list[str]) -> tuple[list[str], list[str]]:
    train = [n for n in names if not is_holdout(n)]
    val = [n for n in names if is_holdout(n)]
    return train, val
```

Commit: `feat(plan3): deterministic match splits`.

---

### Task 3.6 — Model

**Files:** create `ml/point_model/model.py`, `ml/tests/test_pm_model.py`.

- [ ] Test:

```python
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
```

- [ ] Implement:

```python
# ml/point_model/model.py
from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn
from .vocab import NUM_STROKE, NUM_OUTCOME
from .features import FEATURE_DIM
from .labels import MAX_SHOTS

@dataclass
class PointModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_T: int = 300
    n_shots: int = MAX_SHOTS

class PointModel(nn.Module):
    def __init__(self, cfg: PointModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(FEATURE_DIM, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_T, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.head_contact = nn.Linear(cfg.d_model, 1)
        self.head_bounce  = nn.Linear(cfg.d_model, 1)
        self.head_hitter  = nn.Linear(cfg.d_model, 2)

        self.shot_query = nn.Parameter(torch.randn(cfg.n_shots, cfg.d_model) * 0.02)
        self.shot_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                               dropout=cfg.dropout, batch_first=True)
        self.head_stroke = nn.Linear(cfg.d_model, NUM_STROKE)
        self.head_inout  = nn.Linear(cfg.d_model, 1)

        self.head_outcome = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, NUM_OUTCOME),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos[:, :T]
        key_padding = mask < 0.5  # True = pad
        h = self.encoder(h, src_key_padding_mask=key_padding)

        contact = self.head_contact(h).squeeze(-1)
        bounce = self.head_bounce(h).squeeze(-1)
        hit = self.head_hitter(h)

        q = self.shot_query.unsqueeze(0).expand(B, -1, -1)
        shot_h, _ = self.shot_attn(q, h, h, key_padding_mask=key_padding)
        stroke = self.head_stroke(shot_h)
        inout = self.head_inout(shot_h).squeeze(-1)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        outcome = self.head_outcome(pooled)

        return {
            "contact_logits": contact, "bounce_logits": bounce,
            "hitter_per_frame_logits": hit,
            "stroke_logits": stroke, "inout_logits": inout,
            "outcome_logits": outcome,
        }
```

Commit: `feat(plan3): PointModel transformer + 6 heads`.

---

### Task 3.7 — Losses

**Files:** create `ml/point_model/losses.py`, `ml/tests/test_pm_losses.py`.

- [ ] Test:

```python
import torch
from ml.point_model.losses import compute_losses

def test_losses_run():
    B, T = 2, 32
    logits = {
        "contact_logits": torch.zeros(B, T),
        "bounce_logits": torch.zeros(B, T),
        "hitter_per_frame_logits": torch.zeros(B, T, 2),
        "stroke_logits": torch.zeros(B, 4, 9),
        "inout_logits": torch.zeros(B, 4),
        "outcome_logits": torch.zeros(B, 5),
    }
    targets = {
        "contact": torch.zeros(B, T, dtype=torch.long),
        "contact_strong": torch.tensor([1.0, 0.0]),
        "hitter_per_frame": torch.full((B, T), -1, dtype=torch.long),
        "stroke": torch.full((B, 4), -100, dtype=torch.long),
        "outcome": torch.zeros(B, dtype=torch.long),
        "mask": torch.ones(B, T),
    }
    losses = compute_losses(logits, targets)
    assert "total" in losses and torch.is_tensor(losses["total"])
    losses["total"].backward
```

- [ ] Implement:

```python
# ml/point_model/losses.py
from __future__ import annotations
import torch, torch.nn.functional as F

W = {"contact": 1.0, "bounce": 0.5, "hitter": 0.5, "stroke": 1.0, "outcome": 1.0}

def _bce_masked(logits, target, mask, weight: float = 1.0):
    if mask.sum() == 0: return logits.sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return (raw * mask * weight).sum() / mask.sum().clamp_min(1.0)

def compute_losses(logits, t):
    mask = t["mask"]
    cs = t["contact_strong"].view(-1, 1)
    contact_mask = mask * cs                                  # only strong-supervised clips
    weak_mask = mask * (1.0 - cs) * 0.1                       # weak heuristic at 0.1×
    l_contact = _bce_masked(logits["contact_logits"], t["contact"], contact_mask) \
              + _bce_masked(logits["contact_logits"], t["contact"], weak_mask)

    l_bounce = torch.tensor(0.0, device=mask.device)          # placeholder until bounce targets land

    hit_mask = (t["hitter_per_frame"] >= 0).float() * mask
    hit_targets = t["hitter_per_frame"].clamp_min(0)
    raw_hit = F.cross_entropy(logits["hitter_per_frame_logits"].reshape(-1, 2),
                              hit_targets.reshape(-1), reduction="none")
    raw_hit = raw_hit.view_as(hit_mask)
    l_hitter = (raw_hit * hit_mask).sum() / hit_mask.sum().clamp_min(1.0)

    l_stroke = F.cross_entropy(
        logits["stroke_logits"].reshape(-1, logits["stroke_logits"].shape[-1]),
        t["stroke"].reshape(-1), ignore_index=-100,
    )

    l_outcome = F.cross_entropy(logits["outcome_logits"], t["outcome"])

    total = (W["contact"] * l_contact + W["bounce"] * l_bounce + W["hitter"] * l_hitter
             + W["stroke"] * l_stroke + W["outcome"] * l_outcome)
    return {"total": total, "contact": l_contact, "bounce": l_bounce,
            "hitter": l_hitter, "stroke": l_stroke, "outcome": l_outcome}
```

Commit: `feat(plan3): masked multi-task losses`.

---

### Task 3.8 — Metrics

**Files:** create `ml/point_model/metrics.py`.

```python
# ml/point_model/metrics.py
from __future__ import annotations
import torch
from .vocab import STROKE_CLASSES, OUTCOME_CLASSES

def stroke_accuracy(logits, targets) -> dict:
    valid = targets != -100
    if valid.sum() == 0: return {"acc": 0.0}
    pred = logits.argmax(dim=-1)
    return {"acc": float(((pred == targets) & valid).sum() / valid.sum())}

def outcome_accuracy(logits, targets) -> dict:
    pred = logits.argmax(dim=-1)
    return {"acc": float((pred == targets).float().mean())}

def contact_f1(logits, targets, mask, threshold: float = 0.5) -> dict:
    pred = (logits.sigmoid() > threshold).long()
    tp = ((pred == 1) & (targets == 1) & (mask > 0.5)).sum().float()
    fp = ((pred == 1) & (targets == 0) & (mask > 0.5)).sum().float()
    fn = ((pred == 0) & (targets == 1) & (mask > 0.5)).sum().float()
    p = tp / (tp + fp).clamp_min(1); r = tp / (tp + fn).clamp_min(1)
    f1 = 2 * p * r / (p + r).clamp_min(1)
    return {"p": float(p), "r": float(r), "f1": float(f1)}
```

Commit: `feat(plan3): per-head metrics`.

---

### Task 3.9 — Inference contract

**Files:** create `ml/point_model/inference.py`, `ml/tests/test_pm_inference.py`.

- [ ] Test:

```python
import torch
from ml.point_model.inference import run_inference, PointPrediction
from ml.point_model.model import PointModel, PointModelConfig

def test_inference_contract():
    m = PointModel(PointModelConfig()).eval()
    feats = torch.zeros(1, 64, 243); mask = torch.ones(1, 64)
    pred = run_inference(m, feats, mask)
    assert isinstance(pred, list) and isinstance(pred[0], PointPrediction)
    assert len(pred[0].strokes) <= 4
    assert pred[0].outcome in {"Ace","DoubleFault","Winner","ForcedError","UnforcedError"}
```

- [ ] Implement:

```python
# ml/point_model/inference.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
from .vocab import STROKE_CLASSES, OUTCOME_CLASSES

@dataclass
class StrokePrediction:
    index: int
    stroke: str
    hitter: int
    in_court: Optional[bool] = None

@dataclass
class PointPrediction:
    contact_frames: list[int]
    bounce_frames: list[int] = field(default_factory=list)
    strokes: list[StrokePrediction] = field(default_factory=list)
    outcome: str = "UnforcedError"
    raw: dict = field(default_factory=dict)

@torch.no_grad()
def run_inference(model, features: torch.Tensor, mask: torch.Tensor,
                  contact_threshold: float = 0.5) -> list[PointPrediction]:
    out = model(features, mask)
    B = features.shape[0]
    results: list[PointPrediction] = []
    contact = out["contact_logits"].sigmoid()
    hitter = out["hitter_per_frame_logits"].argmax(dim=-1)
    stroke = out["stroke_logits"].argmax(dim=-1)
    outcome = out["outcome_logits"].argmax(dim=-1)
    inout = out["inout_logits"].sigmoid()
    for b in range(B):
        cf = (contact[b] > contact_threshold).nonzero(as_tuple=True)[0].tolist()
        strokes = [StrokePrediction(
            index=i, stroke=STROKE_CLASSES[int(stroke[b, i])],
            hitter=int(hitter[b, cf[i]]) if i < len(cf) else 0,
            in_court=bool(inout[b, i] > 0.5),
        ) for i in range(stroke.shape[1])]
        results.append(PointPrediction(
            contact_frames=cf, strokes=strokes,
            outcome=OUTCOME_CLASSES[int(outcome[b])],
            raw={k: v[b].cpu() for k, v in out.items()},
        ))
    return results
```

Commit: `feat(plan3): PointPrediction inference adapter`.

---

### Task 3.10 — Training loop

**Files:** create `ml/point_model/train.py`.

```python
# ml/point_model/train.py
from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import ClipDataset
from .splits import split_matches
from .model import PointModel, PointModelConfig
from .losses import compute_losses
from .metrics import contact_f1, stroke_accuracy, outcome_accuracy

@dataclass
class TrainConfig:
    clips_root: Path
    features_root: Path
    out_dir: Path
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 2

def _to_targets(batch, device):
    Tg = batch["targets"]  # list of Targets dataclasses (or pre-collated)
    return {
        "contact": torch.from_numpy(np.stack([t.contact_frames for t in Tg])).to(device),
        "contact_strong": torch.tensor([1.0 if t.contact_strong else 0.0 for t in Tg], device=device),
        "hitter_per_frame": torch.from_numpy(np.stack([t.hitter_per_frame for t in Tg])).to(device),
        "stroke": torch.from_numpy(np.stack([t.stroke_idx for t in Tg])).to(device),
        "outcome": torch.tensor([t.outcome_idx for t in Tg], device=device),
        "mask": batch["mask"].to(device),
    }

def _collate(items):
    return {
        "features": torch.from_numpy(np.stack([x["features"] for x in items])),
        "mask":     torch.from_numpy(np.stack([x["mask"]     for x in items])),
        "targets":  [x["targets"] for x in items],
        "meta":     [x["meta"]    for x in items],
    }

def train(cfg: TrainConfig) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matches = sorted(d.name for d in cfg.clips_root.iterdir() if d.is_dir()
                     and (d / "labels.json").exists())
    train_m, val_m = split_matches(matches)
    train_ds = ClipDataset(cfg.clips_root, cfg.features_root, train_m)
    val_ds   = ClipDataset(cfg.clips_root, cfg.features_root, val_m)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=_collate)
    val_dl   = DataLoader(val_ds, cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=_collate)

    model = PointModel(PointModelConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); train_loss = 0.0; n = 0
        for batch in train_dl:
            x = batch["features"].to(device); m = batch["mask"].to(device)
            tg = _to_targets(batch, device)
            opt.zero_grad()
            out = model(x, m); loss = compute_losses(out, tg)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); train_loss += float(loss); n += 1
        train_loss /= max(n, 1)

        model.eval(); val_loss = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["features"].to(device); m = batch["mask"].to(device)
                tg = _to_targets(batch, device)
                out = model(x, m)
                val_loss += float(compute_losses(out, tg)["total"]); vn += 1
        val_loss /= max(vn, 1)

        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                        "secs": time.time() - t0})
        print(f"ep {ep} train={train_loss:.3f} val={val_loss:.3f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": PointModelConfig().__dict__,
                        "val_loss": val_loss}, cfg.out_dir / "best.pt")

    (cfg.out_dir / "metrics.json").write_text(json.dumps({"history": history,
                                                           "best_val": best}, indent=2))
    return {"best_val": best, "history": history}
```

Commit: `feat(plan3): training loop with deterministic split`.

---

### Task 3.11 — Eval

**Files:** create `ml/point_model/eval.py`.

```python
# ml/point_model/eval.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import ClipDataset
from .splits import split_matches
from .model import PointModel, PointModelConfig
from .metrics import contact_f1, stroke_accuracy, outcome_accuracy
from .train import _collate, _to_targets

def evaluate(ckpt_path: Path, clips_root: Path, features_root: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matches = sorted(d.name for d in clips_root.iterdir() if d.is_dir()
                     and (d / "labels.json").exists())
    _, val_m = split_matches(matches)
    ds = ClipDataset(clips_root, features_root, val_m)
    dl = DataLoader(ds, 8, shuffle=False, collate_fn=_collate)

    sd = torch.load(ckpt_path, map_location=device)
    model = PointModel(PointModelConfig(**sd["config"])).to(device)
    model.load_state_dict(sd["model"]); model.eval()

    cf = {"tp":0,"fp":0,"fn":0}
    stroke_correct = 0; stroke_total = 0
    out_correct = 0; out_total = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["features"].to(device); m = batch["mask"].to(device)
            tg = _to_targets(batch, device)
            o = model(x, m)
            f1 = contact_f1(o["contact_logits"], tg["contact"], m)
            # accumulate stroke / outcome accuracy
            sl = o["stroke_logits"].argmax(-1); st = tg["stroke"]
            valid = st != -100
            stroke_correct += int(((sl == st) & valid).sum()); stroke_total += int(valid.sum())
            ol = o["outcome_logits"].argmax(-1)
            out_correct += int((ol == tg["outcome"]).sum()); out_total += int(tg["outcome"].numel())
    return {
        "contact_f1": f1, "stroke_acc": stroke_correct / max(stroke_total, 1),
        "outcome_acc": out_correct / max(out_total, 1),
    }
```

Commit: `feat(plan3): holdout eval`.

---

### Task 3.12 — CLI + smoke

**Files:** create `ml/point_model/__main__.py`.

```python
# ml/point_model/__main__.py
import argparse, json, sys
from pathlib import Path
from .train import train, TrainConfig
from .eval import evaluate

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("point_model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    pt.add_argument("--features", type=Path, default=Path("files/data/features"))
    pt.add_argument("--out", type=Path, default=Path("files/models/point_model/run0"))
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=8)

    pe = sub.add_parser("eval")
    pe.add_argument("--ckpt", type=Path, required=True)
    pe.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    pe.add_argument("--features", type=Path, default=Path("files/data/features"))

    args = ap.parse_args(argv)
    if args.cmd == "train":
        r = train(TrainConfig(args.clips, args.features, args.out,
                              epochs=args.epochs, batch_size=args.batch_size))
        print(json.dumps(r["best_val"]))
    elif args.cmd == "eval":
        r = evaluate(args.ckpt, args.clips, args.features)
        print(json.dumps(r, indent=2))
    return 0

if __name__ == "__main__": sys.exit(main())
```

- [ ] Smoke: with one labeled match in `files/data/clips/_smoke/` and matching `files/data/features/_smoke/`, run:
  ```bash
  python -m ml.point_model train --clips files/data/clips --features files/data/features --epochs 1
  ```
  Expect a `best.pt` and `metrics.json` in the out dir.

- [ ] Commit: `feat(plan3): CLI for train/eval + smoke`.

---

## Done when

- `pytest ml/tests/test_pm_*.py -v` is green.
- `python -m ml.point_model train --epochs 20` produces a checkpoint and decreasing val loss across at least the first few epochs.
- `python -m ml.point_model eval --ckpt files/models/point_model/run0/best.pt` prints `{contact_f1, stroke_acc, outcome_acc}` for the deterministic 10% holdout.
- `PointPrediction` is the wire contract Plan 4 (Go pipeline) consumes.
