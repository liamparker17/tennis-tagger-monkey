"""
Multi-Object Tracker Module

Implements DeepSORT-style tracking for consistent player and ball IDs
across frames. Maintains track continuity even during occlusions.
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class Track:
    """
    Represents a tracked object across multiple frames.
    """
    
    next_id = 0
    
    def __init__(self, bbox: List[float], class_type: str, confidence: float):
        """
        Initialize track.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            class_type: Object class (player/ball/racket)
            confidence: Detection confidence
        """
        self.track_id = Track.next_id
        Track.next_id += 1
        
        self.class_type = class_type
        self.bbox = bbox
        self.confidence = confidence
        
        # Tracking state
        self.age = 0  # Frames since creation
        self.hits = 1  # Number of detections
        self.time_since_update = 0
        
        # Kalman filter for smooth trajectory
        self.kf = self._init_kalman_filter(bbox)
        
        # Appearance feature (for re-identification)
        self.appearance_features = []
    
    def _init_kalman_filter(self, bbox: List[float]) -> KalmanFilter:
        """Initialize Kalman filter for bbox tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [x, y, w, h, vx, vy, vw] (position, size, velocities)
        # Measurement: [x, y, w, h]
        
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        kf.x = np.array([x, y, w, h, 0, 0, 0])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R *= 10
        
        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        # Covariance
        kf.P[4:, 4:] *= 1000
        kf.P *= 10
        
        return kf
    
    def predict(self):
        """Predict next state using Kalman filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: List[float], confidence: float):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            confidence: Detection confidence
        """
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        self.kf.update(np.array([x, y, w, h]))
        
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self) -> Dict:
        """Get current track state."""
        x, y, w, h = self.kf.x[:4]
        
        bbox = [
            x - w/2,  # x1
            y - h/2,  # y1
            x + w/2,  # x2
            y + h/2   # y2
        ]
        
        return {
            'track_id': self.track_id,
            'bbox': bbox,
            'class': self.class_type,
            'confidence': self.confidence,
            'age': self.age,
            'hits': self.hits
        }


class MultiObjectTracker:
    """
    DeepSORT-style multi-object tracker.
    
    Maintains consistent IDs for players and ball across frames,
    handling occlusions and re-appearances.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize tracker.
        
        Args:
            config: Tracking configuration
        """
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.use_appearance = config.get('use_appearance', False)
        
        self.tracks = []
    
    def update(
        self,
        detections: Dict[str, List],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: Detection dictionary from detector
            frame: Optional frame for appearance features
        
        Returns:
            List of active track states
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Convert detections to list
        det_list = []
        for det_type in ['players', 'ball', 'rackets']:
            for det in detections.get(det_type, []):
                det_list.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class': det_type[:-1] if det_type.endswith('s') else det_type
                })
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_tracks = self._associate(det_list)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            det = det_list[det_idx]
            self.tracks[track_idx].update(det['bbox'], det['confidence'])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = det_list[det_idx]
            new_track = Track(det['bbox'], det['class'], det['confidence'])
            self.tracks.append(new_track)
        
        # Remove old tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]
        
        # Return confirmed tracks
        active_tracks = [
            t.get_state()
            for t in self.tracks
            if t.hits >= self.min_hits or t.age < self.min_hits
        ]
        
        return active_tracks
    
    def _associate(
        self,
        detections: List[Dict]
    ) -> tuple:
        """
        Associate detections to tracks using IoU and appearance.
        
        Args:
            detections: List of detection dicts
        
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IoU cost matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t_idx, track in enumerate(self.tracks):
            track_bbox = track.get_state()['bbox']
            for d_idx, det in enumerate(detections):
                # Only match same class
                if track.class_type == det['class']:
                    iou = self._compute_iou(track_bbox, det['bbox'])
                    iou_matrix[t_idx, d_idx] = iou
        
        # Hungarian algorithm for optimal assignment
        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out low IoU matches
        matches = []
        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                matches.append((t_idx, d_idx))
        
        # Find unmatched detections and tracks
        matched_track_indices = [m[0] for m in matches]
        matched_det_indices = [m[1] for m in matches]
        
        unmatched_tracks = [
            i for i in range(len(self.tracks))
            if i not in matched_track_indices
        ]
        
        unmatched_dets = [
            i for i in range(len(detections))
            if i not in matched_det_indices
        ]
        
        return matches, unmatched_dets, unmatched_tracks
    
    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union of two bboxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        
        Returns:
            IoU value [0, 1]
        """
        x1_tl = max(bbox1[0], bbox2[0])
        y1_tl = max(bbox1[1], bbox2[1])
        x1_br = min(bbox1[2], bbox2[2])
        y1_br = min(bbox1[3], bbox2[3])
        
        if x1_br < x1_tl or y1_br < y1_tl:
            return 0.0
        
        intersection = (x1_br - x1_tl) * (y1_br - y1_tl)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        Track.next_id = 0
