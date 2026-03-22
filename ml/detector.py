"""
Object Detector — Single YOLO pass for players + ball.

Transplanted from files/detection/unified_detector.py.
Stripped to clean batch API for Go bridge consumption.
"""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

logger = logging.getLogger(__name__)

# COCO class IDs
CLASS_PERSON = 0
CLASS_SPORTS_BALL = 32

# Ball size filter (pixels) — tennis balls are small
_BALL_MIN_PX = 5
_BALL_MAX_PX = 100

# Default confidence thresholds
_PLAYER_CONF = 0.5
_BALL_CONF = 0.2


class Detector:
    """Single YOLO detector for players + ball.

    Runs one inference pass and splits results by class,
    replacing the old separate PlayerDetector / BallDetector pair.
    """

    BACKENDS = ("yolo", "rtdetr")

    def __init__(self, model_path: str, device: str = "auto", backend: str = "yolo"):
        """Load YOLO model and warm up.

        Args:
            model_path: Path to YOLO weights (.pt or .onnx).
            device: ``"auto"`` picks CUDA when available, else CPU.
                    Pass ``"cpu"`` to force CPU.
            backend: ``"yolo"`` or ``"rtdetr"``.
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}, expected one of {self.BACKENDS}")

        self.backend = backend
        self.model: Optional[object] = None
        self.device = "cpu"
        self.ball_history: deque = deque(maxlen=30)

        if not _YOLO_OK:
            logger.warning("ultralytics not installed — detector disabled")
            return

        # Resolve device
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "0"
                    name = torch.cuda.get_device_name(0)
                    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info("GPU: %s (%.1f GB)", name, mem)
                else:
                    logger.info("No CUDA GPU — using CPU")
            except ImportError:
                pass
        elif device != "cpu":
            self.device = device

        try:
            self.model = YOLO(model_path)
            if self.device != "cpu":
                self.model.to("cuda")
            logger.info("Loaded model %s on device=%s", model_path, self.device)

            # Warmup with a blank frame
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, conf=0.1, device=self.device)
            logger.info("Warmup complete")
        except Exception:
            logger.exception("Failed to load YOLO model")
            self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_batch(self, frames: np.ndarray) -> list[dict]:
        """Detect players and ball in a batch of BGR frames.

        Uses a two-pass approach:
        1. Full-frame pass — catches near player + ball
        2. Top-half crop pass — effectively 2x zoom on far court, catches far player

        Args:
            frames: ``list[np.ndarray]`` or a 4-D array ``(N, H, W, 3)``.

        Returns:
            One dict per frame::

                {
                    "players": [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...],
                    "ball": {"bbox": [x1,y1,x2,y2], "confidence": float} | None
                }
        """
        if self.model is None or len(frames) == 0:
            return self._fallback(frames)

        predict_kwargs = dict(
            conf=_BALL_CONF,
            classes=[CLASS_PERSON, CLASS_SPORTS_BALL],
            verbose=False,
            device=self.device,
            imgsz=640,
            max_det=20,
        )
        if self.backend == "yolo":
            predict_kwargs["half"] = self.device != "cpu"
            predict_kwargs["augment"] = False

        try:
            # Pass 1: full frames
            full_results = self.model.predict(frames, **predict_kwargs)

            # Pass 2: top-half crops (far court zoom)
            crops = []
            crop_heights = []
            for f in frames:
                h = f.shape[0]
                half_h = h // 2
                crops.append(f[:half_h, :, :])
                crop_heights.append(half_h)

            crop_results = self.model.predict(crops, **predict_kwargs)
        except Exception:
            logger.warning("Detection failed, using fallback", exc_info=True)
            return self._fallback(frames)

        output: list[dict] = []
        for idx, (full_res, crop_res) in enumerate(zip(full_results, crop_results)):
            players: list[dict] = []
            ball: Optional[dict] = None

            # Extract from full-frame pass
            players, ball = self._extract_detections(full_res)

            # Extract from crop pass and remap coordinates
            # (crop is top half, so y coordinates are already correct)
            crop_players, crop_ball = self._extract_detections(crop_res)
            # No coordinate adjustment needed — crop is the top half starting at y=0

            # Merge crop players that don't overlap with existing detections
            for cp in crop_players:
                if not self._overlaps_any(cp, players, iou_thresh=0.3):
                    players.append(cp)

            # Use crop ball if no ball found in full frame (crop may see it better)
            if ball is None and crop_ball is not None:
                ball = crop_ball

            # Keep only the 2 largest players by bbox area
            if len(players) > 2:
                players.sort(
                    key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]),
                    reverse=True,
                )
                players = players[:2]

            if ball is not None:
                cx = (ball["bbox"][0] + ball["bbox"][2]) / 2
                cy = (ball["bbox"][1] + ball["bbox"][3]) / 2
                self.ball_history.append((cx, cy))

            output.append({"players": players, "ball": ball})

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_detections(results) -> tuple[list[dict], Optional[dict]]:
        """Parse YOLO results into players list and best ball."""
        players: list[dict] = []
        ball: Optional[dict] = None

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return players, ball

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])
            cls = int(classes[i])

            if cls == CLASS_PERSON and conf >= _PLAYER_CONF:
                players.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                })
            elif cls == CLASS_SPORTS_BALL and conf >= _BALL_CONF:
                w = x2 - x1
                h = y2 - y1
                if _BALL_MIN_PX < w < _BALL_MAX_PX and _BALL_MIN_PX < h < _BALL_MAX_PX:
                    if ball is None or conf > ball["confidence"]:
                        ball = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                        }

        return players, ball

    @staticmethod
    def _overlaps_any(det: dict, existing: list[dict], iou_thresh: float = 0.3) -> bool:
        """Check if det overlaps with any existing detection above IoU threshold."""
        bx = det["bbox"]
        for e in existing:
            ex = e["bbox"]
            # Compute IoU
            ix1 = max(bx[0], ex[0])
            iy1 = max(bx[1], ex[1])
            ix2 = min(bx[2], ex[2])
            iy2 = min(bx[3], ex[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_a = (bx[2] - bx[0]) * (bx[3] - bx[1])
            area_b = (ex[2] - ex[0]) * (ex[3] - ex[1])
            iou = inter / (area_a + area_b - inter)
            if iou >= iou_thresh:
                return True
        return False

    # ------------------------------------------------------------------
    # Fallback (no YOLO)
    # ------------------------------------------------------------------

    def _fallback(self, frames) -> list[dict]:
        """Classical CV ball detection when YOLO is unavailable."""
        out: list[dict] = []
        for frame in frames:
            ball = self._classical_ball(frame)
            out.append({"players": [], "ball": ball})
        return out

    @staticmethod
    def _classical_ball(frame: np.ndarray) -> Optional[dict]:
        """HSV colour + contour detection for tennis ball."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = float(w) / h if h > 0 else 0
                if 0.7 < ar < 1.3:
                    return {
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": 0.5,
                    }
        return None
