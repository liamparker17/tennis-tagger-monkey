from __future__ import annotations
from pathlib import Path
import numpy as np

from .ball_color import BallColorModel


# Fusion weights for YOLO + color. Tuned empirically; kept here rather than
# in BallColorModel so the model is purely descriptive and the detection
# policy lives with the detector.
_W_YOLO = 0.6
_W_COLOR = 0.4
# Fused score floor for accepting a YOLO box. Kept loose because the
# fine-tuned detector's confidence is small-dataset-calibrated (mean ~0.18)
# and the ball color sigma is tight (~5–8) so both channels run low even
# on real balls. Too-high a floor throws away most real detections.
_RECOVERY_FLOOR = 0.10
# A strong single-channel hit is accepted even if its fused score is
# under the floor — lets YOLO-confident or color-confident balls through
# when the other channel happens to be weak on that frame.
_YOLO_ACCEPT = 0.30
_COLOR_ACCEPT = 0.30
# Recovery-path outputs get this conf — low enough that downstream code can
# weight them below YOLO+color agreement.
_RECOVERY_CONF = 0.2
# YOLO confidence threshold — lowered from ultralytics' default (0.25) so
# weak ball detections still make it to the color-scoring stage. Without
# this, a ball with strong color match but low YOLO conf is discarded
# before we get a chance to rescue it.
_YOLO_CONF_THRESHOLD = 0.05


class BallDetector:
    def __init__(self, weights: Path, ball_class_id: int = 32):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.cls_id = ball_class_id
        self.color_model: BallColorModel | None = None
        # Cheap instrumentation so feature extraction can report how often
        # the color recovery path actually fires.
        self.stats = {"frames": 0, "yolo_hit": 0, "color_rescued": 0, "miss": 0}

    def set_color_model(self, model: BallColorModel | None) -> None:
        """Called per-match; None clears the model so the detector falls back
        to YOLO-only behaviour for matches without a color calibration."""
        self.color_model = model

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        self.stats["frames"] += 1
        r = self.model.predict(frame_bgr, verbose=False,
                               conf=_YOLO_CONF_THRESHOLD)[0]
        boxes_exist = r.boxes is not None and len(r.boxes) > 0
        best = None  # (fused_score, cx, cy)
        if boxes_exist:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            mask = cls == self.cls_id
            if mask.any():
                idxs = np.where(mask)[0]
                for i in idxs:
                    x1, y1, x2, y2 = xyxy[i]
                    yolo_c = float(conf[i])
                    if self.color_model is not None:
                        color_s = self.color_model.score_bbox(
                            frame_bgr, x1, y1, x2, y2)
                        fused = _W_YOLO * yolo_c + _W_COLOR * color_s
                    else:
                        fused = yolo_c; color_s = 0.0
                    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                    # Track the single best candidate plus its channel scores
                    # so we can apply both the fused floor and the single-
                    # channel disjunctive accept after the loop.
                    if best is None or fused > best[0]:
                        best = (fused, float(cx), float(cy), yolo_c, color_s)

        if best is not None and (best[0] >= _RECOVERY_FLOOR or
                                 best[3] >= _YOLO_ACCEPT or
                                 best[4] >= _COLOR_ACCEPT):
            self.stats["yolo_hit"] += 1
            return (best[1], best[2], best[0])

        # Recovery: no YOLO ball above floor. If we have a color model, scan
        # the frame; only trust the result if its score is clearly positive.
        if self.color_model is not None:
            cx, cy, score = self.color_model.scan_frame(frame_bgr, stride=4)
            if score >= _RECOVERY_FLOOR:
                self.stats["color_rescued"] += 1
                return (cx, cy, _RECOVERY_CONF * score)

        # Fall through: either YOLO had a weak hit and no color model, or
        # nothing at all. Return the YOLO best if any, otherwise zeros.
        if best is not None:
            self.stats["yolo_hit"] += 1
            return (best[1], best[2], best[0])
        self.stats["miss"] += 1
        return (0.0, 0.0, 0.0)

    # Keep a minimal public fusion helper so other parts of the pipeline
    # (visualisers, diagnostics) can reproduce the accept criterion without
    # duplicating the constants.
    @staticmethod
    def _accept(fused: float, yolo_c: float, color_s: float) -> bool:
        return (fused >= _RECOVERY_FLOOR
                or yolo_c >= _YOLO_ACCEPT
                or color_s >= _COLOR_ACCEPT)

    def reset_stats(self) -> None:
        self.stats = {"frames": 0, "yolo_hit": 0, "color_rescued": 0, "miss": 0}
