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
