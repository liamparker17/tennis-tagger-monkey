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

    f1 = {"p": 0.0, "r": 0.0, "f1": 0.0}
    stroke_correct = 0; stroke_total = 0
    out_correct = 0; out_total = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["features"].to(device); m = batch["mask"].to(device)
            tg = _to_targets(batch, device)
            o = model(x, m)
            f1 = contact_f1(o["contact_logits"], tg["contact"], m)
            sl = o["stroke_logits"].argmax(-1); st = tg["stroke"]
            valid = st != -100
            stroke_correct += int(((sl == st) & valid).sum()); stroke_total += int(valid.sum())
            ol = o["outcome_logits"].argmax(-1)
            out_correct += int((ol == tg["outcome"]).sum()); out_total += int(tg["outcome"].numel())
    return {
        "contact_f1": f1, "stroke_acc": stroke_correct / max(stroke_total, 1),
        "outcome_acc": out_correct / max(out_total, 1),
    }
