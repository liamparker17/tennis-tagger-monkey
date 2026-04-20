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
