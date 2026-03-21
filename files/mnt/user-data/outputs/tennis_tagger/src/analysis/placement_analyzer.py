"""
Placement Analysis Module
Analyzes shot placements on the court
"""

import numpy as np
import logging
from typing import List, Dict, Optional


class PlacementAnalyzer:
    """Analyze shot placements"""
    
    def __init__(self, config: dict):
        self.config = config
        self.placement_config = config.get('placement', {})
        self.logger = logging.getLogger('PlacementAnalyzer')
        self.enabled = self.placement_config.get('enabled', True)
        self.court_regions = self.placement_config.get('court_regions', 9)
    
    def analyze(self, tracks: List[Dict], court_lines: Optional[Dict],
                strokes: List[Dict]) -> List[Dict]:
        """Analyze shot placements"""
        if not self.enabled:
            return []
        
        placements = []
        
        # Define court grid (simplified - would use court_lines in real implementation)
        court_bounds = self._estimate_court_bounds(court_lines)
        
        for stroke in strokes:
            frame_idx = stroke['frame']
            
            # Find ball position at stroke time
            if frame_idx < len(tracks):
                ball_tracks = tracks[frame_idx].get('ball_tracks', [])
                
                if ball_tracks:
                    ball_pos = ball_tracks[0].get('center', [0, 0])
                    
                    # Map to court region
                    region = self._map_to_court_region(ball_pos, court_bounds)
                    depth = self._calculate_depth(ball_pos, court_bounds)
                    
                    placements.append({
                        'stroke_frame': frame_idx,
                        'player_id': stroke['player_id'],
                        'stroke_type': stroke['type'],
                        'position': ball_pos,
                        'court_region': region,
                        'depth_zone': depth
                    })
        
        self.logger.info(f"Analyzed {len(placements)} placements")
        return placements
    
    def _estimate_court_bounds(self, court_lines: Optional[Dict]) -> Dict:
        """Estimate court boundaries"""
        # Placeholder: would use detected court lines
        # For now, return normalized coordinates
        return {
            'x_min': 0.2,
            'x_max': 0.8,
            'y_min': 0.1,
            'y_max': 0.9
        }
    
    def _map_to_court_region(self, position: List[float], 
                            court_bounds: Dict) -> int:
        """Map position to court region (1-9 grid)"""
        x, y = position
        
        # Normalize to court bounds
        x_norm = (x - court_bounds['x_min']) / (court_bounds['x_max'] - court_bounds['x_min'])
        y_norm = (y - court_bounds['y_min']) / (court_bounds['y_max'] - court_bounds['y_min'])
        
        # Clamp to [0, 1]
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))
        
        # Map to 3x3 grid
        col = int(x_norm * 3)
        row = int(y_norm * 3)
        
        if col >= 3:
            col = 2
        if row >= 3:
            row = 2
        
        return row * 3 + col + 1
    
    def _calculate_depth(self, position: List[float], 
                        court_bounds: Dict) -> int:
        """Calculate shot depth (1=shallow, 2=mid, 3=deep)"""
        y = position[1]
        y_norm = (y - court_bounds['y_min']) / (court_bounds['y_max'] - court_bounds['y_min'])
        
        if y_norm < 0.33:
            return 1  # Shallow
        elif y_norm < 0.67:
            return 2  # Mid
        else:
            return 3  # Deep
