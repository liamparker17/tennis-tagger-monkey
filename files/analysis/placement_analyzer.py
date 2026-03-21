"""
Placement Analyzer Module

Analyzes shot placements and court positioning using
court homography and ball trajectory data.
"""

import numpy as np
from typing import Dict, Tuple, List


class PlacementAnalyzer:
    """
    Analyze shot placements on tennis court.
    
    Maps ball positions to court zones and calculates
    shot depths, angles, and patterns.
    """
    
    PLACEMENT_ZONES = {
        'deuce': {
            'wide': 'Deuce Wide',
            'body': 'Deuce Body',
            't': 'Deuce T'
        },
        'ad': {
            'wide': 'Ad Wide',
            'body': 'Ad Body',
            't': 'Ad T'
        }
    }
    
    DEPTH_ZONES = {
        'baseline': 'Baseline',
        'mid_court': 'Mid-court',
        'net': 'Net'
    }
    
    def __init__(self, config: Dict):
        """
        Initialize placement analyzer.
        
        Args:
            config: Placement configuration
        """
        self.config = config
        self.zones = config.get('zones', {})
    
    def analyze_shot(
        self,
        ball_position: Tuple[float, float],
        court_info: Dict
    ) -> Dict:
        """
        Analyze shot placement.
        
        Args:
            ball_position: (x, y) ball position in image coords
            court_info: Court detection information
        
        Returns:
            Placement analysis dictionary
        """
        # Transform to court coordinates
        if court_info and 'homography' in court_info:
            court_pos = self._transform_to_court(ball_position, court_info)
        else:
            court_pos = ball_position
        
        # Determine zone
        zone = self._classify_zone(court_pos)
        
        # Determine depth
        depth = self._classify_depth(court_pos)
        
        # Calculate angle
        angle = self._calculate_angle(court_pos)
        
        return {
            'zone': zone,
            'depth': depth,
            'angle': angle,
            'court_position': court_pos
        }
    
    def _transform_to_court(
        self,
        position: Tuple[float, float],
        court_info: Dict
    ) -> Tuple[float, float]:
        """Transform image position to court coordinates."""
        import cv2
        
        H = court_info['homography']
        if H is None:
            return position
        
        point = np.array([[position]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)
        
        return tuple(transformed[0][0])
    
    def _classify_zone(self, position: Tuple[float, float]) -> str:
        """
        Classify court zone (Deuce/Ad Wide/Body/T).
        
        Args:
            position: (x, y) in normalized court coordinates [0, 1]
        
        Returns:
            Zone name
        """
        x, y = position
        
        # Determine side (deuce vs ad)
        if y < 0.5:
            side = 'deuce'
        else:
            side = 'ad'
        
        # Determine lateral position
        if x < 0.33:
            lateral = 'wide'
        elif x < 0.67:
            lateral = 'body'
        else:
            lateral = 't'
        
        return self.PLACEMENT_ZONES[side][lateral]
    
    def _classify_depth(self, position: Tuple[float, float]) -> str:
        """
        Classify shot depth.
        
        Args:
            position: (x, y) in normalized court coordinates
        
        Returns:
            Depth zone name
        """
        x, y = position
        
        if x < 0.2:
            return self.DEPTH_ZONES['baseline']
        elif x < 0.6:
            return self.DEPTH_ZONES['mid_court']
        else:
            return self.DEPTH_ZONES['net']
    
    def _calculate_angle(self, position: Tuple[float, float]) -> float:
        """
        Calculate shot angle.
        
        Args:
            position: (x, y) in court coordinates
        
        Returns:
            Angle in degrees
        """
        x, y = position
        
        # Calculate angle from center
        center_x, center_y = 0.5, 0.5
        
        dx = x - center_x
        dy = y - center_y
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def analyze_trajectory(
        self,
        ball_positions: List[Tuple[float, float]],
        court_info: Dict
    ) -> Dict:
        """
        Analyze ball trajectory.
        
        Args:
            ball_positions: List of (x, y) positions over time
            court_info: Court information
        
        Returns:
            Trajectory analysis
        """
        if len(ball_positions) < 2:
            return {}
        
        # Calculate trajectory direction
        start_pos = ball_positions[0]
        end_pos = ball_positions[-1]
        
        direction = np.array(end_pos) - np.array(start_pos)
        
        # Classify direction
        if np.abs(direction[0]) > np.abs(direction[1]):
            trajectory_type = 'down_the_line'
        else:
            trajectory_type = 'cross_court'
        
        return {
            'type': trajectory_type,
            'start': start_pos,
            'end': end_pos,
            'distance': np.linalg.norm(direction)
        }
