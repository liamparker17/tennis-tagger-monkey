"""Row-level eval: predict per-clip Dartfish-meaningful fields and diff
against labels.json. Produces a single summary JSON so successive training
runs yield a number that means "X% closer to a human tagger".

Fields evaluated (the ones the trunk model actually predicts; placement XY,
score state, and game/set are out of scope until those modules wire up):
  - stroke[i] type for i in 0..MAX_SHOTS-1   (stroke head, per-slot accuracy)
  - outcome                                  (outcome head, 5 classes)
  - point_won_by                             (derived: outcome + final hitter)
  - n_shots                                  (derived: NMS over contact head)
  - contact_f1                               (per-frame, mirrors eval.py)

Run: python -m ml.point_model.eval_dartfish --ckpt files/models/point_model/current/best.pt
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ClipDataset
from .splits import split_matches, split_clips
from .model import PointModel, PointModelConfig
from .train import _collate, _to_targets
from .vocab import STROKE_CLASSES, OUTCOME_CLASSES, stroke_index, outcome_index
from .labels import MAX_SHOTS
from .fusion import _nms_1d, CONTACT_NMS_DIST, CONTACT_THRESHOLD


# Outcome → which side won the point (relative to server).
#   server-side wins:    Ace, Winner-by-server, opponent-error
#   returner-side wins:  DoubleFault, Winner-by-returner, server-error
# We only know "Winner / ForcedError / UnforcedError" — not who hit/erred —
# so we resolve it from the final-hitter prediction below.
ACE_IDX = OUTCOME_CLASSES.index("Ace")
DF_IDX = OUTCOME_CLASSES.index("DoubleFault")
WINNER_IDX = OUTCOME_CLASSES.index("Winner")
FE_IDX = OUTCOME_CLASSES.index("ForcedError")
UE_IDX = OUTCOME_CLASSES.index("UnforcedError")


def _winner_side_from(outcome_idx: int, final_hitter_is_server: bool) -> int:
    """Returns 0 if server-side wins the point, 1 if returner-side wins."""
    if outcome_idx == ACE_IDX:
        return 0
    if outcome_idx == DF_IDX:
        return 1
    if outcome_idx == WINNER_IDX:
        # Whoever hit the last (winning) shot.
        return 0 if final_hitter_is_server else 1
    # Errors: whoever hit the last (errant) shot loses.
    return 1 if final_hitter_is_server else 0


def _truth_winner_side(point: dict) -> Optional[int]:
    """0 = server-side won, 1 = returner-side won. None if unresolvable."""
    won_by = (point.get("point_won_by") or "").strip().lower()
    server = (point.get("server") or "").strip().lower()
    a = (point.get("player_a") or "").strip().lower()
    b = (point.get("player_b") or "").strip().lower()
    if not won_by:
        return None
    if won_by == server:
        return 0
    # Server wasn't the winner → returner-side won. Sanity check: the
    # other player should be the winner.
    other = b if server == a else a
    if other and won_by == other:
        return 1
    return None


def _truth_n_shots(point: dict) -> int:
    """Best estimate of true shot count: prefer len(stroke_types) if it falls
    inside the human-tagged stroke_count bucket, else clamp."""
    n = len(point.get("stroke_types") or [])
    lo = int(point.get("stroke_count_lo", 0))
    hi = int(point.get("stroke_count_hi", 0))
    if lo > 0 and hi >= lo and lo <= n <= hi:
        return n
    if lo > 0 and hi >= lo:
        return (lo + hi) // 2
    return max(n, 1)


@dataclass
class _SlotTally:
    correct: int = 0
    total: int = 0
    per_class_correct: Counter = None
    per_class_total: Counter = None
    confusion: dict = None  # truth_label -> Counter(pred_label)

    def __post_init__(self):
        if self.per_class_correct is None: self.per_class_correct = Counter()
        if self.per_class_total is None: self.per_class_total = Counter()
        if self.confusion is None: self.confusion = {}

    def add(self, truth: str, pred: str) -> None:
        self.total += 1
        self.per_class_total[truth] += 1
        if truth == pred:
            self.correct += 1
            self.per_class_correct[truth] += 1
        self.confusion.setdefault(truth, Counter())[pred] += 1

    def summary(self) -> dict:
        return {
            "accuracy": self.correct / max(self.total, 1),
            "n": self.total,
            "per_class_accuracy": {
                k: self.per_class_correct[k] / self.per_class_total[k]
                for k in self.per_class_total
                if self.per_class_total[k] > 0
            },
            "per_class_n": dict(self.per_class_total),
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
        }


def _resolve_val(matches: list[str], features_root: Path
                 ) -> tuple[list[str], dict[str, set[str]]]:
    """Mirror train.py's split logic so eval looks at the same val set the
    training run measured. Returns (val_match_names, clip_filter_per_match).
    """
    train_m, val_m = split_matches(matches)
    clip_filters: dict[str, set[str]] = {}
    if not val_m and len(train_m) == 1:
        only = train_m[0]
        feat_dir = features_root / only
        stems = sorted(p.stem for p in feat_dir.glob("*.npz")) if feat_dir.is_dir() else []
        if stems:
            _, va = split_clips(only, stems)
            clip_filters[only] = va
            val_m = [only]
    return val_m, clip_filters


@torch.no_grad()
def evaluate(ckpt_path: Path, clips_root: Path, features_root: Path,
             out_path: Optional[Path] = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matches = sorted(d.name for d in clips_root.iterdir() if d.is_dir()
                     and not d.name.startswith("_")
                     and (d / "labels.json").exists())
    val_m, clip_filters = _resolve_val(matches, features_root)
    if not val_m:
        return {"error": "no val matches available"}

    sd = torch.load(ckpt_path, map_location=device)
    model = PointModel(PointModelConfig(**sd["config"])).to(device)
    model.load_state_dict(sd["model"]); model.eval()

    stroke_tally = [_SlotTally() for _ in range(MAX_SHOTS)]
    stroke_overall = _SlotTally()
    outcome_tally = _SlotTally()
    won_by_tally = _SlotTally()
    n_shots_abs_err = []
    n_shots_within1 = 0
    n_shots_total = 0

    contact_tp = contact_fp = contact_fn = 0

    per_clip_dump: list[dict] = []

    for match in val_m:
        cf = clip_filters.get(match)
        ds = ClipDataset(clips_root, features_root, [match], clip_filter=cf)
        dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=_collate)
        # Map (match, clip) -> raw point dict, so we can recover ground truth
        # for fields that aren't packaged in Targets.
        labels_doc = json.loads((clips_root / match / "labels.json").read_text())
        clip_to_pt = {p["clip"]: p for p in labels_doc["points"]}

        for batch in dl:
            x = batch["features"].to(device); m = batch["mask"].to(device)
            tg = _to_targets(batch, device)
            o = model(x, m)

            # --- per-frame contact F1 (gated to mask) ---
            cp = (torch.sigmoid(o["contact_logits"]) > 0.5).long()
            mm = (m > 0.5).long()
            contact_tp += int(((cp == 1) & (tg["contact"] == 1) & (mm == 1)).sum())
            contact_fp += int(((cp == 1) & (tg["contact"] == 0) & (mm == 1)).sum())
            contact_fn += int(((cp == 0) & (tg["contact"] == 1) & (mm == 1)).sum())

            # --- per-clip predictions ---
            stroke_pred = o["stroke_logits"].argmax(-1).cpu().numpy()  # [B, MAX_SHOTS]
            outcome_pred = o["outcome_logits"].argmax(-1).cpu().numpy()  # [B]
            hitter_pred_pf = o["hitter_per_frame_logits"].argmax(-1).cpu().numpy()  # [B, T]
            contact_p = torch.sigmoid(o["contact_logits"]).cpu().numpy()  # [B, T]

            for b, meta in enumerate(batch["meta"]):
                truth = clip_to_pt.get(meta["clip"])
                if truth is None: continue

                # Predicted contact frames via NMS — same as inference does
                pred_contacts = _nms_1d(contact_p[b], CONTACT_THRESHOLD, CONTACT_NMS_DIST)
                pred_n_shots = len(pred_contacts)

                # ---- stroke per slot ----
                truth_strokes = truth.get("stroke_types") or []
                truth_n = len(truth_strokes)
                for i in range(MAX_SHOTS):
                    if i >= truth_n:
                        continue  # no truth for this slot — skip
                    t_label = STROKE_CLASSES[stroke_index(truth_strokes[i])]
                    p_label = STROKE_CLASSES[int(stroke_pred[b, i])]
                    stroke_tally[i].add(t_label, p_label)
                    stroke_overall.add(t_label, p_label)

                # ---- outcome ----
                t_outcome = OUTCOME_CLASSES[outcome_index(truth.get("outcome", "UnforcedError"))]
                p_outcome = OUTCOME_CLASSES[int(outcome_pred[b])]
                outcome_tally.add(t_outcome, p_outcome)

                # ---- point won by (server vs returner side) ----
                truth_side = _truth_winner_side(truth)
                if truth_side is not None:
                    server_idx = 0 if (truth.get("player_a", "").strip().lower()
                                       == truth.get("server", "").strip().lower()) else 1
                    if pred_contacts:
                        last_f = pred_contacts[-1]
                        last_hitter = int(hitter_pred_pf[b, last_f]) if 0 <= last_f < hitter_pred_pf.shape[1] else server_idx
                        final_is_server = (last_hitter == server_idx)
                    else:
                        final_is_server = True
                    pred_side = _winner_side_from(int(outcome_pred[b]), final_is_server)
                    won_by_tally.add(
                        "server" if truth_side == 0 else "returner",
                        "server" if pred_side == 0 else "returner",
                    )

                # ---- n_shots ----
                truth_ns = _truth_n_shots(truth)
                err = abs(pred_n_shots - truth_ns)
                n_shots_abs_err.append(err)
                if err <= 1: n_shots_within1 += 1
                n_shots_total += 1

                per_clip_dump.append({
                    "match": meta["match"], "clip": meta["clip"],
                    "truth": {"outcome": t_outcome,
                              "strokes": [STROKE_CLASSES[stroke_index(s)]
                                          for s in truth_strokes[:MAX_SHOTS]],
                              "n_shots": truth_ns,
                              "won_by_side": truth_side},
                    "pred": {"outcome": p_outcome,
                             "strokes": [STROKE_CLASSES[int(stroke_pred[b, i])]
                                         for i in range(min(MAX_SHOTS, truth_n))],
                             "n_shots": pred_n_shots},
                })

    # --- aggregate ---
    p = contact_tp / max(contact_tp + contact_fp, 1)
    r = contact_tp / max(contact_tp + contact_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    summary = {
        "ckpt": str(ckpt_path),
        "val_matches": val_m,
        "n_clips_evaluated": n_shots_total,
        "stroke_overall": stroke_overall.summary(),
        "stroke_per_slot": [t.summary() for t in stroke_tally],
        "outcome": outcome_tally.summary(),
        "won_by_side": won_by_tally.summary(),
        "n_shots": {
            "mae": float(np.mean(n_shots_abs_err)) if n_shots_abs_err else 0.0,
            "within_1_accuracy": n_shots_within1 / max(n_shots_total, 1),
            "n": n_shots_total,
        },
        "contact_f1": {"p": p, "r": r, "f1": f1,
                       "tp": contact_tp, "fp": contact_fp, "fn": contact_fn},
        "headline_accuracy": (
            stroke_overall.summary()["accuracy"] * 0.4
            + outcome_tally.summary()["accuracy"] * 0.3
            + won_by_tally.summary()["accuracy"] * 0.3
        ),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "per_clip": per_clip_dump},
                                      indent=2))
    return summary


def _print_human(s: dict) -> None:
    if "error" in s:
        print(s["error"]); return
    print(f"ckpt: {s['ckpt']}")
    print(f"val matches: {s['val_matches']}")
    print(f"clips evaluated: {s['n_clips_evaluated']}")
    print()
    print(f"HEADLINE accuracy (weighted stroke/outcome/won-by): {s['headline_accuracy']:.1%}")
    print()
    so = s["stroke_overall"]
    print(f"stroke overall: {so['accuracy']:.1%} ({so['n']} slots)")
    for cls, acc in sorted(so["per_class_accuracy"].items()):
        n_cls = so["per_class_n"].get(cls, 0)
        print(f"  {cls:10s} {acc:.1%}  n={n_cls}")
    print()
    print("stroke per-slot accuracy:")
    for i, t in enumerate(s["stroke_per_slot"]):
        slot_name = ["serve", "return", "srv+1", "ret+1", "last"][i]
        print(f"  slot {i} ({slot_name:6s}): {t['accuracy']:.1%}  n={t['n']}")
    print()
    o = s["outcome"]
    print(f"outcome: {o['accuracy']:.1%} ({o['n']} points)")
    for cls, acc in sorted(o["per_class_accuracy"].items()):
        print(f"  {cls:14s} {acc:.1%}  n={o['per_class_n'].get(cls, 0)}")
    print()
    w = s["won_by_side"]
    print(f"point won by (server/returner): {w['accuracy']:.1%}  n={w['n']}")
    print()
    n = s["n_shots"]
    print(f"n_shots: MAE={n['mae']:.2f}  within-1={n['within_1_accuracy']:.1%}  n={n['n']}")
    print()
    cf = s["contact_f1"]
    print(f"contact frame F1: p={cf['p']:.2f} r={cf['r']:.2f} f1={cf['f1']:.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("eval_dartfish")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--features", type=Path, default=Path("files/data/features"))
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to write the full per-clip dump as JSON")
    args = ap.parse_args(argv)
    s = evaluate(args.ckpt, args.clips, args.features, args.out)
    _print_human(s)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
