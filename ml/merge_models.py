"""Average two PyTorch checkpoints into one.

Used by `tagger model merge` to combine a local fine-tuned point model with
a friend's. Averages floating-point tensors element-wise; integer buffers
(e.g. BatchNorm counters) take the local copy.

Usage:
    python merge_models.py --out merged.pt local.pt friend.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _state_dict(ck):
    if isinstance(ck, dict):
        for key in ("model", "state_dict", "model_state_dict", "weights"):
            if key in ck and isinstance(ck[key], dict):
                return ck, key, ck[key]
        if all(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck, None, ck
    raise SystemExit("Couldn't find a state_dict in the checkpoint.")


def main() -> int:
    p = argparse.ArgumentParser(description="Average two PyTorch checkpoints.")
    p.add_argument("inputs", nargs=2, help="local.pt friend.pt")
    p.add_argument("--out", required=True, help="Output .pt path")
    args = p.parse_args()

    cks = [torch.load(x, map_location="cpu", weights_only=False) for x in args.inputs]
    sds = [_state_dict(ck) for ck in cks]
    sd_a, sd_b = sds[0][2], sds[1][2]

    if set(sd_a.keys()) != set(sd_b.keys()):
        raise SystemExit(
            "Models have different layers — they must be the same architecture. "
            "Both you and your friend need to start from the same base model."
        )
    for k in sd_a:
        if sd_a[k].shape != sd_b[k].shape:
            raise SystemExit(f"Layer {k!r} has different shapes — incompatible architectures.")

    merged = {}
    for k, ta in sd_a.items():
        tb = sd_b[k]
        if ta.dtype.is_floating_point:
            merged[k] = ((ta.float() + tb.float()) / 2).to(ta.dtype)
        else:
            merged[k] = ta.clone()

    ck_out, sd_key, _ = sds[0]
    if sd_key is None:
        out = merged
    else:
        out = dict(ck_out)
        out[sd_key] = merged
        if "val_loss" in out:
            out["val_loss"] = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Averaged {len(merged)} layers -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
