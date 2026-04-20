from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from .vocab import STROKE_CLASSES, OUTCOME_CLASSES

CONTACT_NMS_DIST = 12
CONTACT_THRESHOLD = 0.3
SERVE_PRIOR = 4.0
ALTERNATE_PRIOR = 3.0
VOLLEY_NEAR_NET_Y = 0.4
GATE_PENALTY = 3.0
AUDIO_BONUS = 1.5
LOW_CONF_OUTCOME = 0.5
LOW_CONF_STROKE = 0.4

SERVE_IDX   = STROKE_CLASSES.index("Serve")
VOLLEY_IDX  = STROKE_CLASSES.index("Volley")
SMASH_IDX   = STROKE_CLASSES.index("Smash")
ACE_IDX     = OUTCOME_CLASSES.index("Ace")
DF_IDX      = OUTCOME_CLASSES.index("DoubleFault")

@dataclass
class FusedStroke:
    index: int
    stroke: str
    hitter: int
    in_court: bool
    prob: float
    contact_frame: int

@dataclass
class FusedPointPrediction:
    contact_frames: list[int]
    bounce_frames: list[int]
    strokes: list[FusedStroke]
    outcome: str
    outcome_prob: float
    low_confidence: bool = False
    raw: dict = field(default_factory=dict)

def _nms_1d(probs: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    cand = np.where(probs >= threshold)[0]
    if cand.size == 0: return []
    cand = sorted(cand, key=lambda i: -probs[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - c) >= min_dist for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)

def _arms_above_head(pose_court: np.ndarray, frame: int, hitter: int) -> bool:
    if frame >= pose_court.shape[0]: return False
    kp = pose_court[frame, hitter]
    nose_y = kp[0, 1]
    wrist_y = min(kp[9, 1], kp[10, 1])
    return wrist_y < nose_y

def _hitter_court_y(pose_court: np.ndarray, frame: int, hitter: int) -> float:
    if frame >= pose_court.shape[0]: return 0.5
    return float(pose_court[frame, hitter, :, 1].mean())

def detect_audio_impulses(audio_mel: np.ndarray, T: int,
                          hi_band: tuple[int, int] = (40, 60),
                          z_threshold: float = 1.5) -> np.ndarray:
    if audio_mel.size == 0: return np.zeros(T, np.bool_)
    n_mels, F_len = audio_mel.shape
    hi = min(hi_band[1], n_mels); lo = min(hi_band[0], hi)
    band = audio_mel[lo:hi].mean(axis=0) if hi > lo else np.zeros(F_len, np.float32)
    if band.size < 5: return np.zeros(T, np.bool_)
    mu = band.mean(); sd = band.std() + 1e-6
    impulse_F = (band - mu) / sd > z_threshold
    # max-pool across resample bins so short impulses survive F→T subsampling
    edges = np.linspace(0, F_len, T + 1).astype(np.int64)
    out = np.zeros(T, np.bool_)
    for i in range(T):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[i] = bool(impulse_F[lo:hi].any())
    return out

def fuse(contact_logits: torch.Tensor, hitter_logits: torch.Tensor,
         stroke_logits: torch.Tensor, inout_logits: torch.Tensor,
         outcome_logits: torch.Tensor, *,
         audio_impulse: np.ndarray, pose_court: np.ndarray,
         fps: int) -> FusedPointPrediction:
    contact_p = torch.sigmoid(contact_logits).cpu().numpy()
    bonus = np.zeros_like(contact_p)
    impulse_idx = np.where(audio_impulse)[0]
    for ti in impulse_idx:
        lo, hi = max(0, ti - 2), min(len(contact_p), ti + 3)
        bonus[lo:hi] += AUDIO_BONUS
    # log-odds + audio bonus → back to probability
    clp = np.clip(contact_p, 1e-6, 1 - 1e-6)
    logit = np.log(clp / (1 - clp)) + bonus
    contact_p_post = 1 / (1 + np.exp(-logit))
    contacts = _nms_1d(contact_p_post, CONTACT_THRESHOLD, CONTACT_NMS_DIST)
    n_shots = len(contacts)

    hitter_p = F.softmax(hitter_logits, dim=-1).cpu().numpy()

    n_slots = stroke_logits.shape[0]
    stroke_lp = F.log_softmax(stroke_logits, dim=-1).cpu().numpy().copy()
    fused_strokes: list[FusedStroke] = []
    prev_hitter = -1
    for i in range(min(max(n_shots, 1), n_slots)):
        if i >= n_shots:
            break
        f = contacts[i]
        if i == 0:
            stroke_lp[i, SERVE_IDX] += SERVE_PRIOR

        hp = np.log(hitter_p[f] + 1e-6).copy()
        if prev_hitter >= 0:
            hp[1 - prev_hitter] += ALTERNATE_PRIOR
        this_hitter = int(np.argmax(hp))

        y = _hitter_court_y(pose_court, f, this_hitter)
        if y > VOLLEY_NEAR_NET_Y:
            stroke_lp[i, VOLLEY_IDX] -= GATE_PENALTY
        if not _arms_above_head(pose_court, f, this_hitter):
            stroke_lp[i, SMASH_IDX] -= GATE_PENALTY

        probs = np.exp(stroke_lp[i] - stroke_lp[i].max())
        probs /= probs.sum()
        cls = int(np.argmax(probs))
        fused_strokes.append(FusedStroke(
            index=i, stroke=STROKE_CLASSES[cls], hitter=this_hitter,
            in_court=bool(torch.sigmoid(inout_logits[i]) > 0.5),
            prob=float(probs[cls]), contact_frame=f,
        ))
        prev_hitter = this_hitter

    outcome_lp = F.log_softmax(outcome_logits, dim=-1).cpu().numpy().copy()
    if n_shots != 1:
        outcome_lp[ACE_IDX] = -1e9
    if n_shots not in (1, 2) and n_shots != 0:
        outcome_lp[DF_IDX] = -1e9
    op = np.exp(outcome_lp - outcome_lp.max()); op /= op.sum()
    out_idx = int(np.argmax(op)); outcome_name = OUTCOME_CLASSES[out_idx]
    outcome_prob = float(op[out_idx])

    low = (outcome_prob < LOW_CONF_OUTCOME
           or any(s.prob < LOW_CONF_STROKE for s in fused_strokes))

    return FusedPointPrediction(
        contact_frames=contacts, bounce_frames=[],
        strokes=fused_strokes, outcome=outcome_name,
        outcome_prob=outcome_prob, low_confidence=low,
    )
