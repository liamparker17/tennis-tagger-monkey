"""
Serve Detection Module
Detects serve motions and serve events
"""

import numpy as np
from typing import List, Dict
import logging


class ServeDetector:
    """Detect serve motions"""
    
    def __init__(self, config: dict):
        self.config = config
        self.serve_config = config.get('events', {}).get('serve_detection', {})
        self.logger = logging.getLogger('ServeDetector')
        self.enabled = self.serve_config.get('enabled', True)
        self.min_arm_angle = self.serve_config.get('min_arm_angle', 120)
    
    def detect(self, poses: List[Dict], tracks: List[Dict]) -> List[Dict]:
        """Detect serves from pose sequences"""
        if not self.enabled:
            return []
        
        serves = []
        
        # Analyze pose sequences for serve motion
        for frame_idx, pose_data in enumerate(poses):
            for pose in pose_data.get('poses', []):
                keypoints = pose.get('keypoints', [])
                
                if len(keypoints) < 33:  # MediaPipe has 33 keypoints
                    continue
                
                # Get relevant keypoints (right arm for right-handed serves)
                # MediaPipe: 12=right shoulder, 14=right elbow, 16=right wrist
                shoulder = keypoints[12]
                elbow = keypoints[14]
                wrist = keypoints[16]
                
                # Check arm extension (serve motion)
                arm_angle = self._calculate_arm_angle(shoulder, elbow, wrist)
                
                # Detect serve if arm is extended upward
                if arm_angle > self.min_arm_angle and wrist['y'] < shoulder['y']:
                    serves.append({
                        'frame': frame_idx,
                        'player_id': pose.get('player_id', -1),
                        'arm_angle': arm_angle,
                        'confidence': 0.8,
                        'type': 'serve'
                    })
        
        # Remove duplicate serves (same player within close frames)
        serves = self._deduplicate_serves(serves)
        
        self.logger.info(f"Detected {len(serves)} serves")
        return serves
    
    def _calculate_arm_angle(self, shoulder: Dict, elbow: Dict, 
                            wrist: Dict) -> float:
        """Calculate arm angle at elbow"""
        # Vector from elbow to shoulder
        v1 = np.array([shoulder['x'] - elbow['x'], 
                      shoulder['y'] - elbow['y']])
        # Vector from elbow to wrist
        v2 = np.array([wrist['x'] - elbow['x'], 
                      wrist['y'] - elbow['y']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        return angle
    
    def _deduplicate_serves(self, serves: List[Dict], 
                           window: int = 30) -> List[Dict]:
        """Remove duplicate serves within window"""
        if not serves:
            return []
        
        serves_sorted = sorted(serves, key=lambda x: x['frame'])
        deduplicated = [serves_sorted[0]]
        
        for serve in serves_sorted[1:]:
            if (serve['frame'] - deduplicated[-1]['frame'] > window or
                serve['player_id'] != deduplicated[-1]['player_id']):
                deduplicated.append(serve)
        
        return deduplicated
