from __future__ import annotations
from pathlib import Path
import numpy as np


class BallDetector:
    def __init__(self, weights: Path, ball_class_id: int = 32):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.cls_id = ball_class_id

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        r = self.model.predict(frame_bgr, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (0.0, 0.0, 0.0)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        mask = cls == self.cls_id
        if not mask.any():
            return (0.0, 0.0, 0.0)
        i = int(np.argmax(np.where(mask, conf, -1)))
        x = (xyxy[i, 0] + xyxy[i, 2]) / 2
        y = (xyxy[i, 1] + xyxy[i, 3]) / 2
        return (float(x), float(y), float(conf[i]))
