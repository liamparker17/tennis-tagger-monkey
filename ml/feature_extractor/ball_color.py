"""Per-match ball color model.

Calibrates from preflight ball_labels: for each labelled bbox, samples the
pixels inside as "ball" and the annular ring around it as "background".
Runtime, any candidate pixel is scored by how well it lies on the line in
CIELab space between the local background and the ball color — the α in
  C(p) = α * ball_color + (1 - α) * bg(p)
captures motion blur, so even smeared ball edges still score high as long
as they're a blend of ball + background rather than a different color.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_RING_MARGIN = 3    # pixels of annular ring sampled as background around each bbox
_RING_THICKNESS = 4
_RESIDUAL_SIGMA = 6.0   # Lab-space stddev for allowable off-line residual


@dataclass
class BallColorModel:
    ball_lab: np.ndarray       # (3,) L,a,b
    bg_lab: np.ndarray         # (3,) fallback background mean (used when local bg unavailable)
    residual_sigma: float = _RESIDUAL_SIGMA
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "ball_lab": self.ball_lab.tolist(),
            "bg_lab": self.bg_lab.tolist(),
            "residual_sigma": float(self.residual_sigma),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BallColorModel":
        return cls(
            ball_lab=np.asarray(d["ball_lab"], dtype=np.float32),
            bg_lab=np.asarray(d["bg_lab"], dtype=np.float32),
            residual_sigma=float(d.get("residual_sigma", _RESIDUAL_SIGMA)),
            n_samples=int(d.get("n_samples", 0)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "BallColorModel":
        return cls.from_dict(json.loads(path.read_text()))

    # ------------------------------------------------------------------
    # Runtime scoring
    # ------------------------------------------------------------------
    def score_lab(self, pixels_lab: np.ndarray, bg_lab: np.ndarray | None = None) -> np.ndarray:
        """Score a batch of Lab pixels. `pixels_lab` is (..., 3), `bg_lab` is
        (3,) or broadcastable. Returns ball-likelihood in [0, 1] of same
        leading shape as `pixels_lab`.

        score = clip(α, 0, 1) * exp(-||residual||² / (2 σ²))
        where C = α * ball + (1-α) * bg, projected in Lab.
        """
        bg = self.bg_lab if bg_lab is None else np.asarray(bg_lab, dtype=np.float32)
        v = (self.ball_lab - bg).astype(np.float32)
        v_sq = float(v @ v) + 1e-6
        diff = pixels_lab.astype(np.float32) - bg
        # α_est along the ball↔bg line
        alpha = (diff @ v) / v_sq
        # perpendicular residual
        # residual = diff - alpha[..., None] * v
        residual = diff - alpha[..., None] * v
        r2 = np.einsum("...k,...k->...", residual, residual)
        gauss = np.exp(-r2 / (2.0 * self.residual_sigma * self.residual_sigma))
        return np.clip(alpha, 0.0, 1.0) * gauss

    def score_bbox(self, frame_bgr: np.ndarray, x1: float, y1: float,
                   x2: float, y2: float) -> float:
        """Mean ball-likelihood over pixels inside the bbox, using the ring
        around the bbox as the local background estimate."""
        h, w = frame_bgr.shape[:2]
        xi1 = max(0, int(round(x1))); yi1 = max(0, int(round(y1)))
        xi2 = min(w, int(round(x2))); yi2 = min(h, int(round(y2)))
        if xi2 <= xi1 or yi2 <= yi1: return 0.0
        crop = frame_bgr[yi1:yi2, xi1:xi2]
        if crop.size == 0: return 0.0
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Local bg: annular ring, fall back to the stored bg if the ring is out of frame.
        rx1 = max(0, xi1 - _RING_MARGIN - _RING_THICKNESS)
        ry1 = max(0, yi1 - _RING_MARGIN - _RING_THICKNESS)
        rx2 = min(w, xi2 + _RING_MARGIN + _RING_THICKNESS)
        ry2 = min(h, yi2 + _RING_MARGIN + _RING_THICKNESS)
        local_bg = None
        if rx2 > rx1 and ry2 > ry1:
            ring = frame_bgr[ry1:ry2, rx1:rx2].copy()
            # mask out the inner bbox + margin
            ix1 = max(0, xi1 - _RING_MARGIN - rx1)
            iy1 = max(0, yi1 - _RING_MARGIN - ry1)
            ix2 = min(ring.shape[1], xi2 + _RING_MARGIN - rx1)
            iy2 = min(ring.shape[0], yi2 + _RING_MARGIN - ry1)
            mask = np.ones(ring.shape[:2], dtype=bool)
            mask[iy1:iy2, ix1:ix2] = False
            if mask.any():
                ring_lab = cv2.cvtColor(ring, cv2.COLOR_BGR2LAB).astype(np.float32)
                local_bg = np.median(ring_lab[mask], axis=0)
        scores = self.score_lab(crop_lab, bg_lab=local_bg)
        return float(scores.mean())

    def scan_frame(self, frame_bgr: np.ndarray, stride: int = 4) -> tuple[float, float, float]:
        """Recovery path: downsample the whole frame, find the pixel with the
        strongest ball-likelihood. Returns (x, y, score) in original-frame
        coordinates. Uses the stored bg_lab as background (no local estimate
        across the full frame)."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (w // stride, h // stride),
                           interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        scores = self.score_lab(lab, bg_lab=self.bg_lab)
        idx = int(np.argmax(scores))
        sy, sx = divmod(idx, scores.shape[1])
        return (float(sx * stride + stride / 2),
                float(sy * stride + stride / 2),
                float(scores[sy, sx]))


def calibrate_from_setup(video_path: Path, ball_labels: list[dict]) -> BallColorModel | None:
    """Walk the preflight ball_labels, seek to each frame in the source video,
    sample pixels inside the bbox as ball and the annular ring as background.
    Returns None if nothing usable was gathered (no labels or video missing)."""
    if not ball_labels or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    ball_px: list[np.ndarray] = []
    bg_px: list[np.ndarray] = []
    try:
        for entry in ball_labels:
            f = int(entry.get("frame", -1))
            bb = entry.get("bbox")
            if f < 0 or not bb: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok or frame is None: continue
            h, w = frame.shape[:2]
            x1 = max(0, int(round(bb["x1"]))); y1 = max(0, int(round(bb["y1"])))
            x2 = min(w, int(round(bb["x2"]))); y2 = min(h, int(round(bb["y2"])))
            if x2 <= x1 or y2 <= y1: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                ball_px.append(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                               .astype(np.float32).reshape(-1, 3))
            # Ring around bbox for background samples.
            rx1 = max(0, x1 - _RING_MARGIN - _RING_THICKNESS)
            ry1 = max(0, y1 - _RING_MARGIN - _RING_THICKNESS)
            rx2 = min(w, x2 + _RING_MARGIN + _RING_THICKNESS)
            ry2 = min(h, y2 + _RING_MARGIN + _RING_THICKNESS)
            if rx2 > rx1 and ry2 > ry1:
                ring = frame[ry1:ry2, rx1:rx2].copy()
                ix1 = max(0, x1 - _RING_MARGIN - rx1)
                iy1 = max(0, y1 - _RING_MARGIN - ry1)
                ix2 = min(ring.shape[1], x2 + _RING_MARGIN - rx1)
                iy2 = min(ring.shape[0], y2 + _RING_MARGIN - ry1)
                mask = np.ones(ring.shape[:2], dtype=bool)
                mask[iy1:iy2, ix1:ix2] = False
                if mask.any():
                    ring_lab = cv2.cvtColor(ring, cv2.COLOR_BGR2LAB).astype(np.float32)
                    bg_px.append(ring_lab[mask])
    finally:
        cap.release()
    if not ball_px: return None
    ball_arr = np.concatenate(ball_px, axis=0)
    bg_arr = np.concatenate(bg_px, axis=0) if bg_px else ball_arr  # degenerate fallback
    # Robust location: median along each Lab channel.
    ball_lab = np.median(ball_arr, axis=0).astype(np.float32)
    bg_lab = np.median(bg_arr, axis=0).astype(np.float32)
    # Residual sigma: how tight the ball cluster is perpendicular to the
    # ball↔bg line. Clamp to avoid pathological shrinkage on tiny samples.
    v = (ball_lab - bg_lab).astype(np.float32)
    v_sq = float(v @ v) + 1e-6
    diff = ball_arr - bg_lab
    alpha = (diff @ v) / v_sq
    resid = diff - alpha[:, None] * v
    r2 = np.einsum("ij,ij->i", resid, resid)
    sigma = float(np.sqrt(np.median(r2) / 2.0)) if len(r2) else _RESIDUAL_SIGMA
    sigma = float(np.clip(sigma, 3.0, 20.0))
    return BallColorModel(ball_lab=ball_lab, bg_lab=bg_lab,
                          residual_sigma=sigma, n_samples=int(len(ball_arr)))
