"""
Rally Analysis Module
Segments match into rallies and analyzes rally patterns
"""

import logging
from typing import List, Dict


class RallyAnalyzer:
    """Analyze and segment rallies"""
    
    def __init__(self, config: dict):
        self.config = config
        self.rally_config = config.get('events', {}).get('rally_segmentation', {})
        self.logger = logging.getLogger('RallyAnalyzer')
        self.min_length = self.rally_config.get('min_rally_length', 2)
        self.max_gap = self.rally_config.get('max_gap_frames', 90)
    
    def segment(self, strokes: List[Dict], tracks: List[Dict]) -> List[Dict]:
        """Segment strokes into rallies"""
        if not strokes:
            return []
        
        rallies = []
        current_rally = []
        last_frame = -1000
        
        for stroke in sorted(strokes, key=lambda x: x['frame']):
            # Start new rally if gap too large
            if stroke['frame'] - last_frame > self.max_gap:
                if len(current_rally) >= self.min_length:
                    rallies.append(self._create_rally(current_rally))
                current_rally = [stroke]
            else:
                current_rally.append(stroke)
            
            last_frame = stroke['frame']
        
        # Add last rally
        if len(current_rally) >= self.min_length:
            rallies.append(self._create_rally(current_rally))
        
        self.logger.info(f"Segmented {len(rallies)} rallies")
        return rallies
    
    def _create_rally(self, strokes: List[Dict]) -> Dict:
        """Create rally object from strokes"""
        return {
            'start_frame': strokes[0]['frame'],
            'end_frame': strokes[-1]['frame'],
            'num_strokes': len(strokes),
            'strokes': strokes,
            'duration_frames': strokes[-1]['frame'] - strokes[0]['frame']
        }
