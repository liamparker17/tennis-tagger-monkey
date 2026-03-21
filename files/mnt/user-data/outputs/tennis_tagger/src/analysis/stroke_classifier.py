"""
Stroke Classification Module
Classifies tennis strokes (forehand, backhand, volley, etc.)
"""

import numpy as np
from typing import List, Dict
import logging
import torch
import torch.nn as nn


class StrokeClassifier:
    """Classify tennis strokes"""
    
    STROKE_CLASSES = ['forehand', 'backhand', 'volley', 'smash', 'drop', 'lob', 'slice']
    
    def __init__(self, config: dict):
        self.config = config
        self.stroke_config = config.get('events', {}).get('stroke_classification', {})
        self.logger = logging.getLogger('StrokeClassifier')
        self.enabled = self.stroke_config.get('enabled', True)
        self.confidence_threshold = self.stroke_config.get('confidence_threshold', 0.6)
        
        # Load custom model if available
        self.model = self._load_model()
    
    def _load_model(self):
        """Load stroke classification model"""
        model_path = self.stroke_config.get('model', None)
        if model_path:
            try:
                # Load custom trained model
                model = torch.load(model_path)
                model.eval()
                return model
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}")
        return None
    
    def classify(self, poses: List[Dict], tracks: List[Dict], 
                frames: List[np.ndarray]) -> List[Dict]:
        """Classify strokes from pose and motion data"""
        if not self.enabled:
            return []
        
        strokes = []
        
        # Detect motion patterns in pose sequences
        for frame_idx in range(len(poses) - 10):
            # Get 10-frame window
            window_poses = poses[frame_idx:frame_idx + 10]
            
            for player_id in self._get_active_players(window_poses):
                # Extract features
                features = self._extract_features(window_poses, player_id)
                
                if features is None:
                    continue
                
                # Classify stroke
                stroke_type, confidence = self._classify_stroke(features)
                
                if confidence > self.confidence_threshold:
                    strokes.append({
                        'frame': frame_idx + 5,  # Middle of window
                        'player_id': player_id,
                        'type': stroke_type,
                        'confidence': confidence
                    })
        
        # Remove duplicates
        strokes = self._deduplicate_strokes(strokes)
        
        self.logger.info(f"Classified {len(strokes)} strokes")
        return strokes
    
    def _get_active_players(self, window_poses: List[Dict]) -> List[int]:
        """Get IDs of active players in window"""
        player_ids = set()
        for frame_poses in window_poses:
            for pose in frame_poses.get('poses', []):
                player_ids.add(pose.get('player_id', -1))
        return list(player_ids)
    
    def _extract_features(self, window_poses: List[Dict], 
                         player_id: int) -> np.ndarray:
        """Extract motion features from pose sequence"""
        features = []
        
        for frame_poses in window_poses:
            player_pose = None
            for pose in frame_poses.get('poses', []):
                if pose.get('player_id') == player_id:
                    player_pose = pose
                    break
            
            if player_pose is None:
                return None
            
            # Extract relevant keypoints (shoulders, elbows, wrists, hips)
            keypoints = player_pose.get('keypoints', [])
            if len(keypoints) < 33:
                return None
            
            # Key points for stroke classification
            relevant_points = [11, 12, 13, 14, 15, 16, 23, 24]  # Arms and hips
            frame_features = []
            for idx in relevant_points:
                frame_features.extend([keypoints[idx]['x'], keypoints[idx]['y']])
            
            features.extend(frame_features)
        
        return np.array(features)
    
    def _classify_stroke(self, features: np.ndarray) -> tuple:
        """Classify stroke from features"""
        if self.model is not None:
            # Use trained model
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                return self.STROKE_CLASSES[predicted.item()], confidence.item()
        else:
            # Simple heuristic classification
            return self._heuristic_classification(features)
    
    def _heuristic_classification(self, features: np.ndarray) -> tuple:
        """Simple heuristic stroke classification"""
        # Placeholder: analyze arm motion patterns
        # In practice, this would use motion analysis
        return 'forehand', 0.7
    
    def _deduplicate_strokes(self, strokes: List[Dict], 
                            window: int = 15) -> List[Dict]:
        """Remove duplicate strokes"""
        if not strokes:
            return []
        
        strokes_sorted = sorted(strokes, key=lambda x: x['frame'])
        deduplicated = [strokes_sorted[0]]
        
        for stroke in strokes_sorted[1:]:
            if (stroke['frame'] - deduplicated[-1]['frame'] > window or
                stroke['player_id'] != deduplicated[-1]['player_id']):
                deduplicated.append(stroke)
        
        return deduplicated
