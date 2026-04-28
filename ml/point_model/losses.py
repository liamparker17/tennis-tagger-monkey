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
    contact_mask = mask * cs
    weak_mask = mask * (1.0 - cs) * 0.1
    l_contact = _bce_masked(logits["contact_logits"], t["contact"], contact_mask) \
              + _bce_masked(logits["contact_logits"], t["contact"], weak_mask)

    # Bounce supervision is sparse — typically only the last bounce per point
    # from ball_anchors. Train only on points where bounce_strong=1; mask out
    # the rest so we don't teach the head "no bounces ever" on unlabeled points.
    bs = t["bounce_strong"].view(-1, 1)
    bounce_mask = mask * bs
    l_bounce = _bce_masked(logits["bounce_logits"], t["bounce"], bounce_mask)

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
