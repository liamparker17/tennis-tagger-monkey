from __future__ import annotations
from pathlib import Path
import numpy as np

class PoseExtractor:
    def __init__(self, weights: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(frame_bgr, verbose=False)[0]
        px = np.zeros((2, 17, 2), np.float32)
        conf = np.zeros((2, 17), np.float32)
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return px, conf
        kp_xy = r.keypoints.xy.cpu().numpy()
        kp_c  = r.keypoints.conf.cpu().numpy()
        det_c = r.boxes.conf.cpu().numpy()
        order = np.argsort(-det_c)[:2]
        sel_xy = kp_xy[order]; sel_c = kp_c[order]
        ys = sel_xy[..., 1].mean(axis=1)
        if len(ys) == 2 and ys[0] < ys[1]:
            sel_xy = sel_xy[::-1]; sel_c = sel_c[::-1]
        for i in range(len(sel_xy)):
            px[i] = sel_xy[i]; conf[i] = sel_c[i]
        return px, conf
