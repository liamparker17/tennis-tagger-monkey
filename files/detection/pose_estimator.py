"""
Pose Estimation Module
Estimates player body keypoints for stroke analysis

OPTIMIZED VERSION:
- Uses YOLOv8-pose on GPU (much faster than MediaPipe CPU)
- Supports batch processing
- Falls back to MediaPipe if YOLO not available
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

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseEstimator:
    """Estimate player poses with GPU acceleration"""

    def __init__(self, config: dict):
        self.config = config
        self.pose_config = config.get('detection', {}).get('pose_estimator', {})
        self.logger = logging.getLogger('PoseEstimator')
        self.enabled = self.pose_config.get('enabled', True)
        self.use_yolo = self.pose_config.get('model', 'yolov8-pose') == 'yolov8-pose'

        # Try to use YOLO-pose first (GPU-accelerated)
        if YOLO_AVAILABLE and self.enabled and self.use_yolo:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = '0'
                    self.logger.info(f"Using GPU for pose estimation: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = 'cpu'
                    self.logger.warning("GPU not available for pose, using CPU")

                # Use YOLOv8n-pose (fastest pose model)
                self.model = YOLO('yolov8n-pose.pt')
                if self.device == '0':
                    self.model.to('cuda')
                self.logger.info(f"Loaded YOLOv8-pose model on device: {self.device}")

                # Warmup
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy_frame, verbose=False, device=self.device)
                self.logger.info("YOLOv8-pose warmup complete")

                self.pose = None  # Not using MediaPipe
                self.mode = 'yolo'

            except Exception as e:
                self.logger.warning(f"Failed to load YOLOv8-pose, falling back to MediaPipe: {e}")
                self.model = None
                self.mode = 'mediapipe'
                self._init_mediapipe()
        else:
            self.model = None
            self.mode = 'mediapipe'
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe pose (fallback)"""
        if MEDIAPIPE_AVAILABLE and self.enabled:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0=lite, 1=full, 2=heavy (using lite for speed)
                min_detection_confidence=self.pose_config.get('confidence', 0.5),
                min_tracking_confidence=0.5
            )
            self.logger.info("MediaPipe Pose initialized (lite mode)")
        else:
            self.pose = None
    
    def estimate(self, frame: np.ndarray,
                 player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses for detected players in single frame"""
        if not self.enabled:
            return []

        if self.mode == 'yolo' and self.model is not None:
            return self._estimate_yolo_single(frame, player_detections)
        elif self.mode == 'mediapipe' and self.pose is not None:
            return self._estimate_mediapipe(frame, player_detections)
        else:
            return []

    def estimate_batch(self, frames: List[np.ndarray],
                      player_detections_batch: List[List[Dict]]) -> List[List[Dict]]:
        """
        Estimate poses for batch of frames (GPU-optimized)

        Args:
            frames: List of frames
            player_detections_batch: List of player detections per frame

        Returns:
            List of pose lists, one per frame
        """
        if not self.enabled or len(frames) == 0:
            return [[] for _ in frames]

        if self.mode == 'yolo' and self.model is not None:
            # GPU-accelerated batch processing
            return self._estimate_yolo_batch(frames, player_detections_batch)
        elif self.mode == 'mediapipe':
            # MediaPipe doesn't support batching, process sequentially
            return [self._estimate_mediapipe(frame, detections)
                   for frame, detections in zip(frames, player_detections_batch)]
        else:
            return [[] for _ in frames]

    def _estimate_yolo_single(self, frame: np.ndarray,
                             player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses using YOLOv8-pose for single frame"""
        if len(player_detections) == 0:
            return []

        # Run YOLO-pose on full frame
        results = self.model.predict(
            frame,
            verbose=False,
            device=self.device,
            conf=0.5
        )[0]

        poses = []
        if results.keypoints is not None:
            keypoints_data = results.keypoints.data.cpu().numpy()

            # Match poses to player detections by bbox overlap
            for i, kp_data in enumerate(keypoints_data):
                # kp_data shape: (17, 3) - 17 keypoints with x, y, confidence
                keypoints = []
                for j in range(17):  # COCO 17 keypoints
                    keypoints.append({
                        'x': float(kp_data[j, 0]),
                        'y': float(kp_data[j, 1]),
                        'confidence': float(kp_data[j, 2]),
                        'z': 0.0,  # YOLO doesn't provide z
                        'visibility': float(kp_data[j, 2])  # Use confidence as visibility
                    })

                # Try to match to player detection
                bbox = results.boxes[i].xyxy[0].cpu().numpy() if i < len(results.boxes) else None

                poses.append({
                    'player_id': -1,  # Will be matched later
                    'keypoints': keypoints,
                    'bbox': [float(x) for x in bbox] if bbox is not None else None
                })

        return poses

    def _estimate_yolo_batch(self, frames: List[np.ndarray],
                            player_detections_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Estimate poses using YOLOv8-pose for batch of frames (GPU-optimized)"""
        # Run YOLO-pose on batch
        results_batch = self.model.predict(
            frames,
            verbose=False,
            device=self.device,
            conf=0.5,
            half=True,  # FP16 for 2x speed on GPU
            augment=False
        )

        all_poses = []
        for results in results_batch:
            frame_poses = []
            if results.keypoints is not None:
                keypoints_data = results.keypoints.data.cpu().numpy()

                for i, kp_data in enumerate(keypoints_data):
                    keypoints = []
                    for j in range(17):  # COCO 17 keypoints
                        keypoints.append({
                            'x': float(kp_data[j, 0]),
                            'y': float(kp_data[j, 1]),
                            'confidence': float(kp_data[j, 2]),
                            'z': 0.0,
                            'visibility': float(kp_data[j, 2])
                        })

                    bbox = results.boxes[i].xyxy[0].cpu().numpy() if i < len(results.boxes) else None

                    frame_poses.append({
                        'player_id': -1,
                        'keypoints': keypoints,
                        'bbox': [float(x) for x in bbox] if bbox is not None else None
                    })

            all_poses.append(frame_poses)

        return all_poses

    def _estimate_mediapipe(self, frame: np.ndarray,
                           player_detections: List[Dict]) -> List[Dict]:
        """Estimate poses using MediaPipe (CPU fallback)"""
        poses = []

        for player in player_detections:
            x1, y1, x2, y2 = player['bbox']

            # Crop player region
            player_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            if player_crop.size == 0:
                continue

            # Convert to RGB
            player_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)

            # Run pose estimation
            results = self.pose.process(player_rgb)

            if results.pose_landmarks:
                # Extract keypoints
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert back to original frame coordinates
                    kp_x = x1 + landmark.x * (x2 - x1)
                    kp_y = y1 + landmark.y * (y2 - y1)
                    keypoints.append({
                        'x': float(kp_x),
                        'y': float(kp_y),
                        'z': float(landmark.z),
                        'visibility': float(landmark.visibility),
                        'confidence': float(landmark.visibility)
                    })

                poses.append({
                    'player_id': player.get('id', -1),
                    'keypoints': keypoints,
                    'bbox': player['bbox']
                })

        return poses
    
    def __del__(self):
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
