"""
Ball Detection Module
Detects and tracks tennis ball in video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class BallDetector:
    """Detect and track tennis ball"""
    
    def __init__(self, config: dict):
        self.config = config
        self.detection_config = config.get('detection', {}).get('ball_detector', {})
        self.logger = logging.getLogger('BallDetector')
        
        self.model_path = self.detection_config.get('model', 'yolov8x.pt')
        self.confidence = self.detection_config.get('confidence', 0.3)
        
        # Ball tracking history
        self.ball_history = deque(maxlen=30)
        
        if YOLO_AVAILABLE:
            try:
                # Check if GPU is available and set device
                import torch
                if torch.cuda.is_available():
                    self.device = '0'  # GPU device
                    self.logger.info(f"Using GPU for ball detection: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = 'cpu'
                    self.logger.warning("GPU not available, using CPU (will be slow)")

                self.model = YOLO(self.model_path)
                # Force model to GPU
                if self.device == '0':
                    self.model.to('cuda')
                self.logger.info(f"Loaded ball detector: {self.model_path} on device: {self.device}")

                # Warmup: Run dummy inference to initialise CUDA kernels
                import numpy as np
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy_frame, verbose=False, conf=0.3, device=self.device)
                self.logger.info("Ball detector warmup complete")

            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect ball in frame"""
        if self.model is None:
            return self._classical_ball_detect(frame)
        
        # Use YOLO or specialized ball detector
        # For tennis ball, we might need a custom-trained model
        # For now, using generic object detection
        detections = []
        
        # Try detecting with sports ball class (class 32 in COCO)
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[32],  # Sports ball
            verbose=False,
            device=self.device
        )
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Filter by size (tennis balls are small)
                width = x2 - x1
                height = y2 - y1
                
                if 5 < width < 100 and 5 < height < 100:
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
        
        return detections[:1]  # Return only best detection

    def detect_batch(self, frames: List[np.ndarray], use_tracking: bool = False) -> List[List[Dict]]:
        """
        Detect ball in a batch of frames (GPU-optimised)

        Args:
            frames: List of input frames (BGR format)
            use_tracking: Use native Ultralytics tracking for ball

        Returns:
            List of detection lists, one per frame
        """
        if self.model is None or len(frames) == 0:
            return [self._classical_ball_detect(f) for f in frames]

        # Use native tracking if requested (better temporal consistency)
        if use_tracking:
            results_batch = self.model.track(
                frames,
                conf=self.confidence,
                classes=[32],  # Sports ball
                verbose=False,
                device=self.device,
                half=True,  # FP16 for 2x speed on GPU (RTX 2050 supports this)
                tracker='bytetrack.yaml',  # ByteTrack works well for single object
                persist=True
            )
        else:
            # Run batch prediction for sports ball with optimisations
            results_batch = self.model.predict(
                frames,
                conf=self.confidence,
                classes=[32],  # Sports ball
                verbose=False,
                device=self.device,
                half=True,  # FP16 for 2x speed on GPU (RTX 2050 supports this)
                augment=False,  # Disable test-time augmentation for speed
                max_det=10  # Limit detections for speed
            )

        # Parse results for each frame
        all_detections = []
        for results in results_batch:
            frame_detections = []
            boxes = results.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # Filter by size (tennis balls are small)
                width = x2 - x1
                height = y2 - y1

                if 5 < width < 100 and 5 < height < 100:
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    }

                    # Add track ID if using native tracking
                    if use_tracking and hasattr(box, 'id') and box.id is not None:
                        detection['track_id'] = int(box.id[0])

                    frame_detections.append(detection)

            all_detections.append(frame_detections[:1])  # Best detection only

        return all_detections

    def _classical_ball_detect(self, frame: np.ndarray) -> List[Dict]:
        """Classical CV ball detection using color and shape"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tennis ball color range (yellow-green)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:  # Size filter
                x, y, w, h = cv2.boundingRect(contour)
                
                # Aspect ratio check (ball should be roughly circular)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    detections.append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': 0.5,
                        'center': [float(x + w/2), float(y + h/2)]
                    })
        
        return detections[:1]
    
    def track(self, detections: List[Dict]) -> List[Dict]:
        """Track ball using Kalman filter or similar"""
        if not detections:
            return []
        
        # Simple tracking with history smoothing
        current_ball = detections[0]
        self.ball_history.append(current_ball['center'])
        
        # Smooth position using recent history
        if len(self.ball_history) > 3:
            smoothed_x = np.mean([pos[0] for pos in self.ball_history])
            smoothed_y = np.mean([pos[1] for pos in self.ball_history])
            
            current_ball['smoothed_center'] = [smoothed_x, smoothed_y]
        
        return [current_ball]
