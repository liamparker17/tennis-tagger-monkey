from __future__ import annotations
import json, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# pythonw.exe (used by the GUI wrapper) drops stdout, so route status lines
# through stderr and flush every print. Makes training progress visible in
# the pipeline log instead of disappearing into the void.
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
from .dataset import ClipDataset
from .splits import split_matches, split_clips
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
    resume: Path | None = None  # if set, load model+opt state and continue from next epoch

def _to_targets(batch, device):
    Tg = batch["targets"]
    return {
        "contact": torch.from_numpy(np.stack([t.contact_frames for t in Tg])).to(device),
        "contact_strong": torch.tensor([1.0 if t.contact_strong else 0.0 for t in Tg], device=device),
        "bounce": torch.from_numpy(np.stack([t.bounce_frames for t in Tg])).to(device),
        "bounce_strong": torch.tensor([1.0 if t.bounce_strong else 0.0 for t in Tg], device=device),
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
    # Skip underscore-prefixed dirs (e.g. _smoke, _shared) — they are test
    # fixtures or cross-match scratch spaces, not real matches. Including
    # them in val produced a 5-orders-of-magnitude train/val gap because the
    # fixture labels are placeholders, not real supervision.
    matches = sorted(d.name for d in cfg.clips_root.iterdir() if d.is_dir()
                     and not d.name.startswith("_")
                     and (d / "labels.json").exists())
    if not matches:
        _log(f"no real matches found under {cfg.clips_root} — nothing to train on")
        return {"best_val": float("inf"), "history": []}
    train_m, val_m = split_matches(matches)
    _log(f"matches: {len(matches)} total, {len(train_m)} train, {len(val_m)} val")

    # Single-match fallback: if match-level split left val empty, partition
    # clips inside the one training match deterministically (hash on
    # match/stem) so we still get a real val signal instead of val=0.0.
    train_clip_filter: set[str] | None = None
    val_clip_filter: set[str] | None = None
    if not val_m and len(train_m) == 1:
        only = train_m[0]
        feat_dir = cfg.features_root / only
        stems = sorted(p.stem for p in feat_dir.glob("*.npz")) if feat_dir.is_dir() else []
        if stems:
            tr, va = split_clips(only, stems)
            train_clip_filter, val_clip_filter = tr, va
            val_m = [only]
            _log(f"single-match fallback: holding out {len(va)}/{len(stems)} "
                 f"clips from {only} as val (~{len(va)/len(stems):.0%})")
        else:
            _log("WARN: no val matches — val_loss will be 0.0 and uninformative. "
                 "Add a second match or use clip-level splitting to get a real val signal.")
    elif not val_m:
        _log("WARN: no val matches — val_loss will be 0.0 and uninformative. "
             "Add a second match or use clip-level splitting to get a real val signal.")

    train_ds = ClipDataset(cfg.clips_root, cfg.features_root, train_m,
                           clip_filter=train_clip_filter)
    val_ds   = ClipDataset(cfg.clips_root, cfg.features_root, val_m,
                           clip_filter=val_clip_filter)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=_collate)
    val_dl   = DataLoader(val_ds, cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=_collate)

    model_cfg = PointModelConfig()
    model = PointModel(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    start_epoch = 0

    if cfg.resume is not None and cfg.resume.exists():
        ck = torch.load(cfg.resume, map_location=device)
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as e:
            # Architecture changed since the checkpoint was saved (different
            # seq length, feature dim, or head count). Rather than crashing
            # the pipeline, archive the stale checkpoint and start fresh.
            stale = cfg.resume.with_suffix(cfg.resume.suffix + ".stale_arch")
            cfg.resume.rename(stale)
            _log(f"checkpoint arch mismatch — archived {cfg.resume.name} -> {stale.name}")
            _log(f"  ({str(e).splitlines()[0]})")
            _log("starting fresh.")
        else:
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
            start_epoch = int(ck.get("epoch", -1)) + 1
            best = float(ck.get("best_val", best))
            history = list(ck.get("history", []))
            _log(f"resuming from {cfg.resume} @ epoch {start_epoch}, best_val={best:.3f}")

    if start_epoch >= cfg.epochs:
        _log(f"checkpoint already at epoch {start_epoch}/{cfg.epochs}. "
             f"nothing to do — reset the run or raise --epochs.")
        return {"best_val": best, "history": history}

    for ep in range(start_epoch, cfg.epochs):
        model.train(); t0 = time.time(); train_loss = 0.0; n = 0; n_skipped = 0
        for batch in train_dl:
            x = batch["features"].to(device); m = batch["mask"].to(device)
            tg = _to_targets(batch, device)
            opt.zero_grad()
            out = model(x, m); loss = compute_losses(out, tg)["total"]
            # Skip the step when loss is non-finite (NaN/Inf). Stepping with a
            # non-finite gradient corrupts every parameter and silently kills
            # the run. Earlier 20-epoch runs hit this at epoch 16.
            if not torch.isfinite(loss):
                n_skipped += 1; continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); train_loss += float(loss); n += 1
        train_loss /= max(n, 1)
        if n_skipped:
            _log(f"  WARN: skipped {n_skipped} non-finite batches this epoch")

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
        _log(f"ep {ep} train={train_loss:.3f} val={val_loss:.3f}")
        # Only overwrite best.pt when we have a real val signal. With no val
        # matches, val_loss is 0.0 every epoch and epoch 0 weights would
        # win forever.
        if len(val_ds) > 0 and val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "config": model_cfg.__dict__,
                        "val_loss": val_loss, "epoch": ep}, cfg.out_dir / "best.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": model_cfg.__dict__,
                    "epoch": ep, "val_loss": val_loss,
                    "best_val": best, "history": history},
                   cfg.out_dir / "last.pt")

    (cfg.out_dir / "metrics.json").write_text(json.dumps({"history": history,
                                                           "best_val": best}, indent=2))
    return {"best_val": best, "history": history}
