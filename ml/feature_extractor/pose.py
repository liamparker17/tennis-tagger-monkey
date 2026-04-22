from __future__ import annotations
from pathlib import Path
import numpy as np

from .player_color import PlayerColorModel


# Dropped well below ultralytics' default (0.25) because broadcast framing
# makes the far player small and low-confidence. The color model filters
# out stands/umpire/ballkids that slip through at this threshold.
_POSE_CONF_THRESHOLD = 0.10


class PoseExtractor:
    def __init__(self, weights: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.color_model: PlayerColorModel | None = None
        self.stats = {"frames": 0, "color_used": 0, "y_fallback": 0}

    def set_color_model(self, model: PlayerColorModel | None) -> None:
        self.color_model = model

    def reset_stats(self) -> None:
        self.stats = {"frames": 0, "color_used": 0, "y_fallback": 0}

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.stats["frames"] += 1
        r = self.model.predict(frame_bgr, verbose=False,
                               conf=_POSE_CONF_THRESHOLD)[0]
        px = np.zeros((2, 17, 2), np.float32)
        conf = np.zeros((2, 17), np.float32)
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return px, conf
        kp_xy = r.keypoints.xy.cpu().numpy()
        kp_c = r.keypoints.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()

        # Proactive color-driven selection: score EVERY candidate against
        # both players' color signatures from preflight and pick the best
        # match per identity. This uses the preflight labels as they were
        # meant to be used — actively finding the far player even when
        # they're low-confidence or outnumbered by stands/umpire — rather
        # than just disambiguating two already-picked boxes.
        if self.color_model is not None:
            idx_a, idx_b = self.color_model.pick_best(frame_bgr, xyxy)
            if idx_a is not None:
                px[0] = kp_xy[idx_a]; conf[0] = kp_c[idx_a]
            if idx_b is not None:
                px[1] = kp_xy[idx_b]; conf[1] = kp_c[idx_b]
            if idx_a is not None or idx_b is not None:
                self.stats["color_used"] += 1
                return px, conf

        # Fallback: no color model at all. Take top 2 by detection conf
        # and assume preflight orientation (near = bottom of frame =
        # player_a). Wrong after any side swap, but better than nothing.
        det_c = r.boxes.conf.cpu().numpy()
        order = np.argsort(-det_c)[:2]
        sel_xy = kp_xy[order]; sel_c = kp_c[order]
        ys = sel_xy[..., 1].mean(axis=1)
        if len(ys) == 2 and ys[0] < ys[1]:
            sel_xy = sel_xy[::-1]; sel_c = sel_c[::-1]
        for i in range(len(sel_xy)):
            px[i] = sel_xy[i]; conf[i] = sel_c[i]
        self.stats["y_fallback"] += 1
        return px, conf
