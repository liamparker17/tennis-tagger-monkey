"""
Unified Object Detector Module

PERFORMANCE OPTIMIZATION: Single YOLO inference for both players AND ball.

Previously, PlayerDetector and BallDetector each loaded YOLOv8 and ran separate
inference passes. This unified detector runs the model ONCE with both classes,
effectively DOUBLING processing speed.

GPU Memory: ~2GB for YOLOv8x (was ~4GB with two separate models)
Speed: 2x faster than running separate detectors
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class UnifiedDetector:
    """
    Single YOLO detector for all object classes (players + ball).

    Replaces separate PlayerDetector and BallDetector for 2x speed improvement.
    """

    # COCO class IDs
    CLASS_PERSON = 0
    CLASS_SPORTS_BALL = 32

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('UnifiedDetector')

        # Get detection configs
        player_config = config.get('detection', {}).get('player_detector', {})
        ball_config = config.get('detection', {}).get('ball_detector', {})

        # Use the more capable model (usually they're the same anyway)
        self.model_path = player_config.get('model', 'yolov8x.pt')
        self.player_confidence = player_config.get('confidence', 0.5)
        self.ball_confidence = ball_config.get('confidence', 0.3)

        # Use lower confidence for unified detection, filter later
        self.min_confidence = min(self.player_confidence, self.ball_confidence)

        # Ball tracking history (for smoothing)
        self.ball_history = deque(maxlen=30)

        # Track ID management for custom tracking fallback
        self.next_player_id = 1
        self.next_ball_id = 1
        self.player_tracks = {}
        self.ball_tracks = {}

        self.model = None
        self.device = 'cpu'

        if YOLO_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = '0'
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    self.logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
                else:
                    self.logger.warning("No GPU detected, using CPU (will be slow)")

                # Load model ONCE
                self.model = YOLO(self.model_path)

                if self.device == '0':
                    self.model.to('cuda')

                self.logger.info(f"Unified detector loaded: {self.model_path}")
                self.logger.info(f"  Classes: Person (conf>{self.player_confidence}), Ball (conf>{self.ball_confidence})")

                # Warmup
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, verbose=False, conf=0.1, device=self.device)
                self.logger.info("Warmup complete")

            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            self.logger.warning("Ultralytics not available")

    def detect_all_batch(
        self,
        frames: List[np.ndarray],
        use_tracking: bool = True
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Detect players AND ball in a single inference pass.

        Args:
            frames: List of BGR frames
            use_tracking: Use native ByteTrack for tracking

        Returns:
            Tuple of (player_detections_batch, ball_detections_batch)
            Each is a list of detection lists, one per frame
        """
        if self.model is None or len(frames) == 0:
            return self._fallback_detect_batch(frames)

        # SINGLE inference for both classes [person, sports_ball]
        try:
            if use_tracking:
                results_batch = self.model.track(
                    frames,
                    conf=self.min_confidence,
                    classes=[self.CLASS_PERSON, self.CLASS_SPORTS_BALL],
                    verbose=False,
                    device=self.device,
                    half=True,  # FP16 for 2x speed
                    tracker='bytetrack.yaml',
                    persist=True,
                    imgsz=640,  # Consistent size
                    max_det=20  # Max detections per frame
                )
            else:
                results_batch = self.model.predict(
                    frames,
                    conf=self.min_confidence,
                    classes=[self.CLASS_PERSON, self.CLASS_SPORTS_BALL],
                    verbose=False,
                    device=self.device,
                    half=True,
                    augment=False,
                    imgsz=640,
                    max_det=20
                )
        except Exception as e:
            self.logger.warning(f"Detection error: {e}, using fallback")
            return self._fallback_detect_batch(frames)

        # Split results by class
        players_batch = []
        balls_batch = []

        for results in results_batch:
            players = []
            balls = []

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                players_batch.append([])
                balls_batch.append([])
                continue

            # Batch transfer to CPU (more efficient than per-box)
            xyxy_all = boxes.xyxy.cpu().numpy()
            conf_all = boxes.conf.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(int)

            # Get track IDs if available
            ids_all = None
            if hasattr(boxes, 'id') and boxes.id is not None:
                ids_all = boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy_all[i]
                conf = float(conf_all[i])
                cls = int(cls_all[i])
                track_id = int(ids_all[i]) if ids_all is not None else -1

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                }

                if track_id >= 0:
                    detection['track_id'] = track_id

                if cls == self.CLASS_PERSON and conf >= self.player_confidence:
                    detection['class'] = 'player'
                    players.append(detection)
                elif cls == self.CLASS_SPORTS_BALL and conf >= self.ball_confidence:
                    # Filter ball by size (tennis balls are small)
                    width = x2 - x1
                    height = y2 - y1
                    if 5 < width < 100 and 5 < height < 100:
                        detection['class'] = 'ball'
                        balls.append(detection)

            players_batch.append(players)
            balls_batch.append(balls[:1])  # Only best ball detection

        return players_batch, balls_batch

    def _fallback_detect_batch(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Fallback classical detection when YOLO unavailable"""
        players_batch = []
        balls_batch = []

        for frame in frames:
            players_batch.append([])  # Would need HOG detector
            balls_batch.append(self._classical_ball_detect(frame))

        return players_batch, balls_batch

    def _classical_ball_detect(self, frame: np.ndarray) -> List[Dict]:
        """Classical CV ball detection using color and shape"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tennis ball color range (yellow-green)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    detections.append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': 0.5,
                        'center': [float(x + w/2), float(y + h/2)],
                        'class': 'ball'
                    })

        return detections[:1]


# Singleton for reuse across main.py
_unified_detector_instance = None

def get_unified_detector(config: dict) -> UnifiedDetector:
    """Get or create singleton unified detector"""
    global _unified_detector_instance
    if _unified_detector_instance is None:
        _unified_detector_instance = UnifiedDetector(config)
    return _unified_detector_instance
