"""Per-match player color identity model.

Same projection math as BallColorModel, but repurposed: the goal isn't
"where is the player in the frame" (YOLO pose already solves that) but
"which detected box is player_a (P1) and which is player_b (P2)?" —
human identity, not court position, so it survives side swaps.

Two color anchors per player (shirt + shorts / primary+secondary) make it
robust when one body region is occluded or one clothing color matches the
other player's.

Schema produced by preflight step 5:

  setup.json:
    player_colors: {
      a: [[L1,a1,b1], [L2,a2,b2]],   # player_a = P1 (the name typed in
      b: [[L1,a1,b1], [L2,a2,b2]],   # 'NEAR player' at preflight time)
    }

At runtime, each YOLO person box gets scored against all 4 anchors using
the α-projection (same as ball color). Per-player score = max over that
player's two anchors. Best assignment of 2 boxes → {a, b} is chosen by
total score. If the winning assignment margin is small we return None,
and the caller falls back to sort-by-y (position-based — wrong after a
side swap, but better than inventing an identity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


_RESIDUAL_SIGMA = 8.0        # a bit looser than ball — clothing varies per region
_TORSO_CROP_FRAC = 0.35      # sample fraction of box height from top (torso)
_LEG_CROP_FRAC = 0.35        # same from bottom (legs)
_MIN_MARGIN = 0.05           # if winner - loser < this, skip color assignment


@dataclass
class PlayerColorModel:
    a_anchors: np.ndarray      # (K, 3) Lab — player_a (P1) anchors, K in {1, 2}
    b_anchors: np.ndarray      # (K, 3) Lab — player_b (P2) anchors
    bg_lab: np.ndarray         # (3,) generic background (court median)
    residual_sigma: float = _RESIDUAL_SIGMA

    def to_dict(self) -> dict:
        return {
            "a_anchors": self.a_anchors.tolist(),
            "b_anchors": self.b_anchors.tolist(),
            "bg_lab": self.bg_lab.tolist(),
            "residual_sigma": float(self.residual_sigma),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerColorModel":
        # Back-compat: legacy caches saved under near/far keys are still
        # loadable (near = player_a, far = player_b by preflight convention).
        a = d.get("a_anchors", d.get("near_anchors"))
        b = d.get("b_anchors", d.get("far_anchors"))
        return cls(
            a_anchors=np.asarray(a, dtype=np.float32),
            b_anchors=np.asarray(b, dtype=np.float32),
            bg_lab=np.asarray(d["bg_lab"], dtype=np.float32),
            residual_sigma=float(d.get("residual_sigma", _RESIDUAL_SIGMA)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlayerColorModel":
        return cls.from_dict(json.loads(path.read_text()))

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def _score_sample(self, pixels_lab: np.ndarray, anchor: np.ndarray) -> float:
        v = (anchor - self.bg_lab).astype(np.float32)
        v_sq = float(v @ v) + 1e-6
        diff = pixels_lab - self.bg_lab
        alpha = (diff @ v) / v_sq
        resid = diff - alpha[..., None] * v
        r2 = np.einsum("...k,...k->...", resid, resid)
        gauss = np.exp(-r2 / (2.0 * self.residual_sigma * self.residual_sigma))
        return float((np.clip(alpha, 0, 1) * gauss).mean())

    def score_box(self, frame_bgr: np.ndarray, x1: float, y1: float,
                  x2: float, y2: float) -> tuple[float, float]:
        """Returns (a_score, b_score) for a single YOLO person box — max
        over that player's anchors, mean over torso + leg samples."""
        h, w = frame_bgr.shape[:2]
        xi1 = max(0, int(round(x1))); yi1 = max(0, int(round(y1)))
        xi2 = min(w, int(round(x2))); yi2 = min(h, int(round(y2)))
        if xi2 <= xi1 or yi2 <= yi1: return (0.0, 0.0)
        bh = yi2 - yi1
        torso_end = yi1 + int(bh * _TORSO_CROP_FRAC)
        leg_start = yi2 - int(bh * _LEG_CROP_FRAC)
        crops_bgr = []
        if torso_end > yi1:
            crops_bgr.append(frame_bgr[yi1:torso_end, xi1:xi2])
        if yi2 > leg_start:
            crops_bgr.append(frame_bgr[leg_start:yi2, xi1:xi2])
        if not crops_bgr: return (0.0, 0.0)
        samples_lab = [
            cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
            for c in crops_bgr if c.size
        ]
        if not samples_lab: return (0.0, 0.0)
        lab = np.concatenate(samples_lab, axis=0)
        a_score = max(self._score_sample(lab, anc) for anc in self.a_anchors)
        b_score = max(self._score_sample(lab, anc) for anc in self.b_anchors)
        return (a_score, b_score)

    def assign(self, frame_bgr: np.ndarray,
               boxes_xyxy: np.ndarray) -> list[int] | None:
        """Given a (K,4) array of person boxes, return a length-K list
        mapping each box to player identity (0 = player_a/P1,
        1 = player_b/P2, -1 = unassigned). Returns None when the winning
        assignment's margin is below the noise floor — caller should fall
        back to sort-by-y position."""
        if boxes_xyxy is None or len(boxes_xyxy) == 0: return None
        scores = [self.score_box(frame_bgr, *b) for b in boxes_xyxy[:2]]
        if len(scores) == 1:
            as_, bs = scores[0]
            return [0 if as_ > bs else 1]
        # Two-box case: pick the assignment (box0→A, box1→B) vs
        # (box0→B, box1→A) with the higher total score.
        (a0, b0), (a1, b1) = scores[0], scores[1]
        sa = a0 + b1   # box0=A, box1=B
        sb = b0 + a1   # box0=B, box1=A
        if abs(sa - sb) < _MIN_MARGIN: return None
        return [0, 1] if sa > sb else [1, 0]

    def pick_best(self, frame_bgr: np.ndarray,
                  boxes_xyxy: np.ndarray) -> tuple[int | None, int | None]:
        """Given K >= 1 person boxes (no need to pre-trim to 2), return
        `(idx_a, idx_b)` — the indices of the boxes that best match
        player_a and player_b respectively, or None for either when no
        candidate clears the noise floor. Unlike `assign`, this is how we
        USE the color signature proactively: when YOLO-pose returns 3+
        detections (stands/umpire/etc.), we pick the two that actually
        match the players, not just the two highest-confidence boxes.
        When YOLO only returns 1 candidate, we still tag it as whichever
        player it matches best (a or b)."""
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return None, None
        scores = [self.score_box(frame_bgr, *b) for b in boxes_xyxy]
        a_scores = [s[0] for s in scores]
        b_scores = [s[1] for s in scores]
        idx_a = int(np.argmax(a_scores)) if a_scores else None
        idx_b = int(np.argmax(b_scores)) if b_scores else None
        # Guard against the same box winning both identities (can happen
        # when one player is off-screen): keep the stronger assignment,
        # leave the other slot empty.
        if idx_a is not None and idx_a == idx_b:
            if a_scores[idx_a] >= b_scores[idx_b]:
                idx_b = None
            else:
                idx_a = None
        # Enforce a minimum score so a random background box isn't
        # tagged when the player really is out of frame.
        if idx_a is not None and a_scores[idx_a] < _MIN_MARGIN:
            idx_a = None
        if idx_b is not None and b_scores[idx_b] < _MIN_MARGIN:
            idx_b = None
        return idx_a, idx_b


def calibrate_from_setup(video_path: Path, reference_frame_idx: int,
                         player_colors: dict) -> PlayerColorModel | None:
    """Build a model from the Lab values clicked in preflight. `player_colors`
    is the dict saved in setup.json with keys `a` (player_a/P1, = the name
    typed in "NEAR player" at preflight time) and `b` (player_b/P2, = FAR
    player at preflight). Legacy near/far keys are accepted for back-compat.
    The reference frame is used to sample a court-background median for
    the bg_lab term."""
    if not player_colors: return None
    a_raw = player_colors.get("a") or player_colors.get("near") or []
    b_raw = player_colors.get("b") or player_colors.get("far") or []
    if not a_raw or not b_raw: return None
    a_anchors = np.asarray(a_raw, dtype=np.float32).reshape(-1, 3)
    b_anchors = np.asarray(b_raw, dtype=np.float32).reshape(-1, 3)
    bg_lab = np.array([128.0, 128.0, 128.0], dtype=np.float32)  # neutral fallback
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(reference_frame_idx)))
                ok, frame = cap.read()
                if ok and frame is not None:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                    bg_lab = np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)
            finally:
                cap.release()
    return PlayerColorModel(a_anchors=a_anchors, b_anchors=b_anchors, bg_lab=bg_lab)
