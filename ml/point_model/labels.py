from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .vocab import stroke_index, outcome_index, STROKE_PAD_INDEX

MAX_SHOTS = 4

@dataclass
class Targets:
    contact_frames: np.ndarray
    hitter_per_frame: np.ndarray
    contact_strong: bool
    stroke_idx: np.ndarray
    hitter_per_shot: np.ndarray
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
