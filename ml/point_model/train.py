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
    Tg = batch["targets"]
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
