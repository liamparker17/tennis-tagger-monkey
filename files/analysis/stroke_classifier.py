"""
Stroke Classifier Module

Classifies tennis strokes using 3D CNNs (X3D) that analyze
temporal sequences of frames to identify stroke types.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import cv2


class StrokeClassifier:
    """
    Tennis stroke classification using 3D convolutional networks.
    
    Classifies strokes into:
    - Forehand
    - Backhand
    - Forehand Volley
    - Backhand Volley
    - Serve
    - Smash
    - Drop Shot
    - Lob
    """
    
    STROKE_CLASSES = [
        "Forehand",
        "Backhand",
        "Forehand Volley",
        "Backhand Volley",
        "Serve",
        "Smash",
        "Drop Shot",
        "Lob"
    ]
    
    def __init__(
        self,
        model_name: str = "x3d_m",
        device: torch.device = torch.device('cpu'),
        confidence_threshold: float = 0.7
    ):
        """
        Initialize stroke classifier.
        
        Args:
            model_name: Model architecture (x3d_s, x3d_m, x3d_l)
            device: Torch device
            confidence_threshold: Minimum confidence for classification
        """
        self.device = device
        self.conf_threshold = confidence_threshold
        self.temporal_window = 16  # Frames per clip
        
        # Load pretrained model (would be custom-trained in production)
        print(f"   Loading {model_name} stroke classifier...")
        self.model = self._load_model(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"   Stroke classifier ready")
    
    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load or create stroke classification model.
        
        In production, this would load a fine-tuned model.
        For now, creates a simple 3D CNN architecture.
        """
        # Simple 3D CNN for demonstration
        # In production, use pre-trained X3D from pytorchvideo
        model = Simple3DCNN(
            num_classes=len(self.STROKE_CLASSES),
            temporal_depth=self.temporal_window
        )
        
        return model
    
    def classify_sequence(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict],
        fps: float
    ) -> List[Dict]:
        """
        Classify all strokes in a video sequence.
        
        Args:
            frames: List of video frames
            tracks: Tracking results from MultiObjectTracker
            fps: Video frame rate
        
        Returns:
            List of stroke event dictionaries
        """
        stroke_events = []
        
        # Detect potential stroke moments (ball trajectory changes)
        stroke_candidates = self._detect_stroke_candidates(tracks, fps)
        
        # Classify each candidate
        for candidate in stroke_candidates:
            start_frame = candidate['frame']
            player_id = candidate['player_id']
            
            # Extract temporal clip around stroke
            clip_frames = self._extract_clip(
                frames, tracks, start_frame, player_id
            )
            
            if clip_frames is None:
                continue
            
            # Classify stroke
            stroke_class, confidence = self._classify_clip(clip_frames)
            
            if confidence >= self.conf_threshold:
                stroke_events.append({
                    'timestamp': start_frame / fps,
                    'frame': start_frame,
                    'player_id': player_id,
                    'stroke_type': stroke_class,
                    'confidence': confidence,
                    'player_bbox': candidate.get('player_bbox'),
                    'ball_position': candidate.get('ball_position')
                })
        
        return stroke_events
    
    def _detect_stroke_candidates(
        self,
        tracks: List[Dict],
        fps: float
    ) -> List[Dict]:
        """
        Detect potential stroke moments from tracking data.
        
        Looks for:
        - Ball trajectory changes (direction/speed)
        - Ball-player proximity
        """
        candidates = []
        ball_positions = []
        
        # Extract ball trajectory
        for frame_idx, frame_tracks in enumerate(tracks):
            ball_track = None
            for track in frame_tracks.get('tracks', []):
                if track['class'] == 'ball':
                    ball_track = track
                    break
            
            if ball_track:
                bbox = ball_track['bbox']
                ball_pos = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ]
                ball_positions.append({
                    'frame': frame_idx,
                    'position': ball_pos,
                    'track_id': ball_track['track_id']
                })
        
        # Detect trajectory changes (potential strokes)
        for i in range(2, len(ball_positions) - 2):
            prev_pos = np.array(ball_positions[i-1]['position'])
            curr_pos = np.array(ball_positions[i]['position'])
            next_pos = np.array(ball_positions[i+1]['position'])
            
            # Compute velocity change
            vel_before = curr_pos - prev_pos
            vel_after = next_pos - curr_pos
            
            # Check for significant direction/speed change
            if len(vel_before) > 0 and len(vel_after) > 0:
                angle_change = self._angle_between(vel_before, vel_after)
                
                if angle_change > 45:  # Significant direction change
                    frame_idx = ball_positions[i]['frame']
                    
                    # Find nearby player
                    player_id, player_bbox = self._find_nearby_player(
                        tracks[frame_idx], curr_pos
                    )
                    
                    if player_id is not None:
                        candidates.append({
                            'frame': frame_idx,
                            'player_id': player_id,
                            'player_bbox': player_bbox,
                            'ball_position': curr_pos.tolist()
                        })
        
        return candidates
    
    def _extract_clip(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict],
        center_frame: int,
        player_id: int
    ) -> np.ndarray:
        """
        Extract temporal clip centered on stroke moment.
        
        Args:
            frames: All video frames
            tracks: Tracking data
            center_frame: Frame index of stroke
            player_id: ID of player performing stroke
        
        Returns:
            Clip tensor [T, H, W, C] or None if extraction fails
        """
        half_window = self.temporal_window // 2
        start_frame = max(0, center_frame - half_window)
        end_frame = min(len(frames), center_frame + half_window)
        
        clip_frames = []
        
        for frame_idx in range(start_frame, end_frame):
            if frame_idx >= len(frames) or frame_idx >= len(tracks):
                break
            
            frame = frames[frame_idx]
            
            # Find player bbox in this frame
            player_bbox = None
            for track in tracks[frame_idx].get('tracks', []):
                if track['track_id'] == player_id:
                    player_bbox = track['bbox']
                    break
            
            if player_bbox is None:
                # Player not found, use full frame
                clip_frames.append(frame)
            else:
                # Crop to player region with padding
                x1, y1, x2, y2 = [int(c) for c in player_bbox]
                h, w = frame.shape[:2]
                
                # Add padding
                pad = 50
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                cropped = frame[y1:y2, x1:x2]
                
                # Resize to standard size
                resized = cv2.resize(cropped, (224, 224))
                clip_frames.append(resized)
        
        # Pad if needed
        while len(clip_frames) < self.temporal_window:
            if len(clip_frames) > 0:
                clip_frames.append(clip_frames[-1])
            else:
                return None
        
        # Convert to tensor
        clip = np.stack(clip_frames[:self.temporal_window])
        return clip
    
    def _classify_clip(self, clip: np.ndarray) -> Tuple[str, float]:
        """
        Classify stroke type from clip.
        
        Args:
            clip: Video clip [T, H, W, C]
        
        Returns:
            Tuple of (stroke_class, confidence)
        """
        # Preprocess
        clip_normalized = clip.astype(np.float32) / 255.0
        
        # Convert to tensor [C, T, H, W]
        clip_tensor = torch.from_numpy(clip_normalized).permute(3, 0, 1, 2)
        clip_tensor = clip_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(clip_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        
        stroke_class = self.STROKE_CLASSES[pred_idx.item()]
        confidence_val = confidence.item()
        
        return stroke_class, confidence_val
    
    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in degrees."""
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    @staticmethod
    def _find_nearby_player(
        frame_tracks: Dict,
        ball_position: np.ndarray,
        max_distance: float = 200
    ) -> Tuple[int, List]:
        """Find player closest to ball position."""
        min_distance = float('inf')
        nearest_player_id = None
        nearest_bbox = None

        for track in frame_tracks.get('tracks', []):
            if track['class'] == 'player':
                bbox = track['bbox']
                player_center = np.array([
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ])

                distance = np.linalg.norm(ball_position - player_center)

                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    nearest_player_id = track['track_id']
                    nearest_bbox = bbox

        return nearest_player_id, nearest_bbox

    def classify(
        self,
        poses: List[Dict],
        tracks: List[Dict],
        frames: List[np.ndarray] = None
    ) -> List[Dict]:
        """
        Classify strokes from pose and track data.

        This is a simplified classification method that works without video frames.
        For full 3D CNN classification, use classify_sequence() with frames.

        Args:
            poses: List of pose data per frame
            tracks: List of tracking data per frame
            frames: Optional video frames (if provided, uses full 3D CNN)

        Returns:
            List of stroke event dictionaries
        """
        # If frames are provided, use the full classify_sequence method
        if frames is not None and len(frames) > 0:
            # Need FPS - estimate from track data or use default
            fps = 30.0  # Default assumption
            return self.classify_sequence(frames, tracks, fps)

        # Without frames, do simplified pose-based stroke detection
        # This detects stroke events based on pose keypoint motion
        stroke_events = []

        if not poses or len(poses) == 0:
            return stroke_events

        # Detect strokes from pose motion patterns
        fps = 30.0  # Default assumption

        for frame_idx in range(2, len(poses) - 2):
            frame_poses = poses[frame_idx] if isinstance(poses[frame_idx], list) else [poses[frame_idx]]

            for pose_data in frame_poses:
                if pose_data is None:
                    continue

                keypoints = pose_data.get('keypoints', [])
                if len(keypoints) < 10:
                    continue

                # Simple heuristic: detect arm motion indicative of stroke
                # Right wrist (10), left wrist (9), right elbow (8), left elbow (7)
                # Check if there's significant arm movement

                stroke_type = self._classify_from_pose(keypoints, frame_idx, poses)

                if stroke_type:
                    stroke_events.append({
                        'timestamp': frame_idx / fps,
                        'frame': frame_idx,
                        'player_id': pose_data.get('track_id', -1),
                        'stroke_type': stroke_type,
                        'confidence': 0.5,  # Lower confidence for pose-only detection
                        'player_bbox': pose_data.get('bbox'),
                        'detection_method': 'pose_heuristic'
                    })

        return stroke_events

    def _classify_from_pose(
        self,
        keypoints: List,
        frame_idx: int,
        all_poses: List
    ) -> str:
        """
        Classify stroke type from pose keypoints using heuristics.

        This is a simplified classifier that looks at arm positions
        to determine stroke type.

        Args:
            keypoints: Current frame keypoints
            frame_idx: Current frame index
            all_poses: All poses for temporal context

        Returns:
            Stroke type string or None
        """
        if len(keypoints) < 11:
            return None

        try:
            # COCO keypoint indices:
            # 5=left_shoulder, 6=right_shoulder
            # 7=left_elbow, 8=right_elbow
            # 9=left_wrist, 10=right_wrist

            right_wrist = keypoints[10] if len(keypoints) > 10 else None
            left_wrist = keypoints[9] if len(keypoints) > 9 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 else None
            left_shoulder = keypoints[5] if len(keypoints) > 5 else None

            if not all([right_wrist, left_wrist, right_shoulder, left_shoulder]):
                return None

            # Check confidence
            min_conf = 0.3
            for kp in [right_wrist, left_wrist, right_shoulder, left_shoulder]:
                if len(kp) < 3 or kp[2] < min_conf:
                    return None

            # Determine dominant hand based on wrist height relative to shoulder
            # Higher wrist = arm raised = potential stroke

            right_arm_raised = right_wrist[1] < right_shoulder[1] - 50
            left_arm_raised = left_wrist[1] < left_shoulder[1] - 50

            # Check for serve (both arms high, or dominant arm very high)
            if right_wrist[1] < right_shoulder[1] - 150 or left_wrist[1] < left_shoulder[1] - 150:
                return "Serve"

            # Check for groundstrokes based on arm position
            if right_arm_raised and not left_arm_raised:
                # Right arm dominant - likely forehand for right-hander
                if right_wrist[0] > right_shoulder[0]:
                    return "Forehand"
                else:
                    return "Backhand"
            elif left_arm_raised and not right_arm_raised:
                # Left arm dominant
                if left_wrist[0] < left_shoulder[0]:
                    return "Forehand"
                else:
                    return "Backhand"

            return None

        except (IndexError, TypeError):
            return None


class Simple3DCNN(nn.Module):
    """
    Simple 3D CNN for stroke classification.
    
    In production, replace with pre-trained X3D from pytorchvideo.
    """
    
    def __init__(self, num_classes: int = 8, temporal_depth: int = 16):
        super().__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
