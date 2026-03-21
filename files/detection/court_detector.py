"""
Court Detector Module

Detects tennis court boundaries, lines, and establishes
coordinate system for placement analysis.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class CourtDetector:
    """
    Detect and track tennis court in video frames.
    
    Establishes homography transformation for:
    - Shot placement analysis
    - Player positioning
    - Ball trajectory mapping
    """
    
    def __init__(self, config: Dict):
        """
        Initialize court detector.
        
        Args:
            config: Court detection configuration
        """
        self.config = config
        self.method = config.get('method', 'line_detection')
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect court in frame.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Court information dict or None if detection fails
        """
        if self.method == 'line_detection':
            return self._detect_by_lines(frame)
        else:
            return self._detect_by_keypoints(frame)
    
    def _detect_by_lines(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect court using line detection.
        
        Args:
            frame: Input frame
        
        Returns:
            Court info dict with corners and homography
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) < 4:
            return None
        
        # Find court corners (simplified - would use more sophisticated method)
        h, w = frame.shape[:2]
        
        # Default court corners
        corners = np.array([
            [w * 0.1, h * 0.8],  # Bottom left
            [w * 0.9, h * 0.8],  # Bottom right
            [w * 0.2, h * 0.2],  # Top left
            [w * 0.8, h * 0.2]   # Top right
        ], dtype=np.float32)
        
        # Standard court dimensions (normalized)
        standard_court = np.array([
            [0, 1],  # Bottom left
            [1, 1],  # Bottom right
            [0, 0],  # Top left
            [1, 0]   # Top right
        ], dtype=np.float32)
        
        # Compute homography
        homography, _ = cv2.findHomography(corners, standard_court)
        
        return {
            'corners': corners,
            'homography': homography,
            'method': 'line_detection',
            'confidence': 0.8
        }
    
    def _detect_by_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect court using keypoint matching."""
        # Placeholder for keypoint-based detection
        return self._detect_by_lines(frame)
    
    def transform_point(
        self,
        point: Tuple[float, float],
        court_info: Dict
    ) -> Tuple[float, float]:
        """
        Transform image point to court coordinates.
        
        Args:
            point: (x, y) in image coordinates
            court_info: Court detection info
        
        Returns:
            (x, y) in normalized court coordinates [0, 1]
        """
        if court_info is None or 'homography' not in court_info:
            return point
        
        H = court_info['homography']
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, H)
        
        return tuple(transformed[0][0])
    
    def get_default(self) -> Dict:
        """Get default court info if detection fails."""
        return {
            'corners': None,
            'homography': None,
            'method': 'default',
            'confidence': 0.0
        }
