"""
Player Detection Module
Detects and tracks tennis players in video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available")


class PlayerDetector:
    """Detect and track tennis players"""
    
    def __init__(self, config: dict):
        self.config = config
        self.detection_config = config.get('detection', {}).get('player_detector', {})
        self.tracking_config = config.get('tracking', {})
        self.logger = logging.getLogger('PlayerDetector')
        
        # Load model
        self.model_path = self.detection_config.get('model', 'yolov8x.pt')
        self.confidence = self.detection_config.get('confidence', 0.5)
        self.iou_threshold = self.detection_config.get('iou_threshold', 0.45)
        self.max_detections = self.detection_config.get('max_detections', 4)
        
        if YOLO_AVAILABLE:
            try:
                # Check if GPU is available and set device
                import torch
                if torch.cuda.is_available():
                    self.device = '0'  # GPU device
                    self.logger.info(f"Using GPU for detection: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = 'cpu'
                    self.logger.warning("GPU not available, using CPU (will be slow)")

                self.model = YOLO(self.model_path)
                # Force model to GPU
                if self.device == '0':
                    self.model.to('cuda')
                self.logger.info(f"Loaded YOLO model: {self.model_path} on device: {self.device}")

                # Warmup: Run dummy inference to initialise CUDA kernels
                import numpy as np
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy_frame, verbose=False, conf=0.5, device=self.device)
                self.logger.info("Model warmup complete")

            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            self.model = None
            self.logger.warning("YOLO not available, using fallback detector")
        
        # Initialize tracking state
        self.tracks = {}
        self.next_track_id = 0
        self.max_age = self.tracking_config.get('max_age', 30)
        self.min_hits = self.tracking_config.get('min_hits', 3)
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections, each with bbox, confidence, class
        """
        if self.model is None:
            return self._fallback_detect(frame)
        
        # Run YOLO detection
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[0],  # Person class
            verbose=False,
            device=self.device
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': cls,
                    'class_name': 'person'
                })
        
        # Limit to max detections
        detections = sorted(detections, key=lambda x: x['confidence'], 
                          reverse=True)[:self.max_detections]
        
        return detections

    def detect_batch(self, frames: List[np.ndarray], use_tracking: bool = False) -> List[List[Dict]]:
        """
        Detect players in a batch of frames (GPU-optimised)

        Args:
            frames: List of input frames (BGR format)
            use_tracking: Use native Ultralytics tracking (ByteTrack/BoTSORT)

        Returns:
            List of detection lists, one per frame
        """
        if self.model is None or len(frames) == 0:
            return [self._fallback_detect(f) for f in frames]

        # Use native Ultralytics tracking if requested (much faster!)
        if use_tracking:
            # Track method is optimized and includes temporal information
            tracker_type = self.tracking_config.get('method', 'bytetrack')
            results_batch = self.model.track(
                frames,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=[0],  # Person class
                verbose=False,
                device=self.device,
                half=True,  # FP16 for 2x speed on GPU (RTX 2050 supports this)
                tracker=f'{tracker_type}.yaml',  # bytetrack.yaml or botsort.yaml
                persist=True  # Persist tracks across batches
            )
        else:
            # Run batch prediction with optimisations
            results_batch = self.model.predict(
                frames,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=[0],  # Person class
                verbose=False,
                device=self.device,
                half=True,  # FP16 for 2x speed on GPU (RTX 2050 supports this)
                augment=False,  # Disable test-time augmentation for speed
                max_det=self.max_detections  # Limit early to save processing
            )

        # Parse results for each frame
        all_detections = []
        for results in results_batch:
            frame_detections = []
            boxes = results.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': 0,
                    'class_name': 'person'
                }

                # Add track ID if using native tracking
                if use_tracking and hasattr(box, 'id') and box.id is not None:
                    detection['track_id'] = int(box.id[0])

                frame_detections.append(detection)

            # Limit to max detections
            frame_detections = sorted(frame_detections, key=lambda x: x['confidence'],
                                    reverse=True)[:self.max_detections]
            all_detections.append(frame_detections)

        return all_detections

    def _fallback_detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Fallback detection using classical CV methods
        """
        # Simple background subtraction or motion detection
        # This is a placeholder - real implementation would use HOG, etc.
        return []
    
    def track(self, detections: List[Dict]) -> List[Dict]:
        """
        Track players across frames using simple IOU tracking
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of tracks with IDs
        """
        # Simple IOU-based tracking
        current_tracks = []
        
        # Match detections to existing tracks
        matched = set()
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in matched:
                    continue
                
                iou = self._compute_iou(track_data['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Update track if match found
            if best_iou > self.tracking_config.get('iou_threshold', 0.3):
                self.tracks[track_id]['bbox'] = detections[best_det_idx]['bbox']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                matched.add(best_det_idx)
                
                if self.tracks[track_id]['hits'] >= self.min_hits:
                    current_tracks.append({
                        'id': track_id,
                        'bbox': self.tracks[track_id]['bbox'],
                        'confidence': detections[best_det_idx]['confidence']
                    })
            else:
                # Increment age if no match
                self.tracks[track_id]['age'] += 1
        
        # Remove old tracks
        self.tracks = {
            tid: tdata for tid, tdata in self.tracks.items()
            if tdata['age'] < self.max_age
        }
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched:
                self.tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'age': 0,
                    'hits': 1
                }
                self.next_track_id += 1
        
        return current_tracks
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Compute intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Compute union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
