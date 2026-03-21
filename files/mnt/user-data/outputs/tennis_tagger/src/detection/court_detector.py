"""
Court Detection Module
Detects tennis court lines and boundaries
"""

import cv2
import numpy as np
from typing import Dict, Optional, List
import logging


class CourtDetector:
    """Detect tennis court lines"""
    
    def __init__(self, config: dict):
        self.config = config
        self.court_config = config.get('detection', {}).get('court_detector', {})
        self.logger = logging.getLogger('CourtDetector')
        self.enabled = self.court_config.get('enabled', True)
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect court lines in frame"""
        if not self.enabled:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                               threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Filter and classify lines (vertical, horizontal, service lines)
        court_lines = {
            'vertical': [],
            'horizontal': [],
            'service': [],
            'all': lines.tolist()
        }
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:  # Horizontal
                court_lines['horizontal'].append(line[0].tolist())
            elif 75 < angle < 105:  # Vertical
                court_lines['vertical'].append(line[0].tolist())
        
        return court_lines
