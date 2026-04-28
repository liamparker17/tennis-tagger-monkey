from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .vocab import stroke_index, outcome_index, STROKE_PAD_INDEX

MAX_SHOTS = 5  # serve, return, srv+1, ret+1, last_shot

@dataclass
class Targets:
    contact_frames: np.ndarray
    hitter_per_frame: np.ndarray
    contact_strong: bool
    bounce_frames: np.ndarray
    bounce_strong: bool
    stroke_idx: np.ndarray
    hitter_per_shot: np.ndarray
    outcome_idx: int
    server_is_p1: bool


def _alternate_hitters(server_is_p1: bool, n: int) -> list[int]:
    """Returns a list of length n alternating between 0 and 1, where the
    value at each index identifies which HUMAN hit that shot (0 = player_a
    = P1, 1 = player_b = P2). Server hits first and the two players
    alternate after that. Human identity, not court position — this must
    agree with the identity axis used by the pose features (idx 0 of
    pose_px/pose_conf is player_a, idx 1 is player_b)."""
    first = 0 if server_is_p1 else 1
    return [first if (i % 2 == 0) else 1 - first for i in range(n)]


def _server_is_p1(server: str, player_a: str, player_b: str) -> bool:
    """P1 = player_a by project convention. Compare by name; fall back to
    the literal 'P1' sentinel if names are missing."""
    s = server.strip().lower()
    a = player_a.strip().lower()
    b = player_b.strip().lower()
    if a and s == a: return True
    if b and s == b: return False
    return s == "p1"


def _estimate_stroke_count(lo: int, hi: int, stroke_types: list[str]) -> int:
    """Dartfish reports a stroke-count bucket like '1 to 4'. Use the number
    of actual tagged strokes when it falls inside the bucket, else use the
    midpoint. Clamp to [1, MAX_SHOTS]."""
    n_tagged = len(stroke_types)
    if lo > 0 and hi >= lo and lo <= n_tagged <= hi:
        n = n_tagged
    elif lo > 0 and hi >= lo:
        n = (lo + hi) // 2
    else:
        n = n_tagged
    return max(1, min(n, MAX_SHOTS))


def build_targets(*, T: int, fps: int,
                  stroke_count_lo: int, stroke_count_hi: int,
                  stroke_types: list[str],
                  outcome: str, point_won_by: str,
                  server: str, player_a: str, player_b: str,
                  strong_contact_frames: Optional[list[tuple[int, int]]],
                  bounce_frames: Optional[list[int]] = None) -> Targets:
    contact = np.zeros((T,), np.int64)
    hitter_pf = np.full((T,), -1, np.int64)
    server_is_p1 = _server_is_p1(server, player_a, player_b)
    n = _estimate_stroke_count(stroke_count_lo, stroke_count_hi, stroke_types)
    n = min(n, T)
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

    bounce = np.zeros((T,), np.int64)
    bounce_is_strong = False
    if bounce_frames:
        bounce_is_strong = True
        for f in bounce_frames:
            if 0 <= f < T:
                bounce[f] = 1

    stroke_idx = np.full((MAX_SHOTS,), STROKE_PAD_INDEX, np.int64)
    hitter_ps = np.full((MAX_SHOTS,), -1, np.int64)
    for i, s in enumerate(stroke_types[:MAX_SHOTS]):
        stroke_idx[i] = stroke_index(s)
        if i < n: hitter_ps[i] = seq[i]

    return Targets(
        contact_frames=contact, hitter_per_frame=hitter_pf,
        contact_strong=strong,
        bounce_frames=bounce, bounce_strong=bounce_is_strong,
        stroke_idx=stroke_idx,
        hitter_per_shot=hitter_ps, outcome_idx=outcome_index(outcome),
        server_is_p1=server_is_p1,
    )
