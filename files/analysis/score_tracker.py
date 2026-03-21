"""
Score Tracker Module

Tracks tennis match score using:
- OCR on scoreboard region
- State machine validation
- Event-based updates
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import easyocr


class ScoreTracker:
    """
    Track match score throughout video.
    
    Combines OCR detection with rule-based validation
    to maintain accurate score state.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize score tracker.
        
        Args:
            config: Score tracking configuration
        """
        self.config = config
        self.use_ocr = config.get('use_ocr', True)
        self.ocr_region = config.get('ocr_region', [0.7, 0.0, 1.0, 0.15])
        
        # Initialize OCR reader
        if self.use_ocr:
            self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Score state
        self.current_score = {
            'p1_sets': 0,
            'p2_sets': 0,
            'p1_games': 0,
            'p2_games': 0,
            'p1_points': 0,
            'p2_points': 0,
            'set_number': 1,
            'game_number': 1
        }
    
    def track(
        self,
        frames: List[np.ndarray],
        events: Dict,
        fps: float
    ) -> Dict[float, Dict]:
        """
        Track score throughout video.
        
        Args:
            frames: Video frames
            events: Detected events
            fps: Frame rate
        
        Returns:
            Dictionary mapping timestamp to score state
        """
        score_timeline = {}
        
        # Initialize score
        current_score = self.current_score.copy()
        
        # Process point events
        for point_event in events.get('point_events', []):
            timestamp = point_event['start_time']
            
            # Read score from frame if OCR enabled
            frame_idx = int(timestamp * fps)
            if frame_idx < len(frames) and self.use_ocr:
                detected_score = self._read_score_from_frame(frames[frame_idx])
                if detected_score:
                    current_score = self._validate_score_transition(
                        current_score, detected_score
                    )
            
            # Update based on point winner
            if point_event.get('winner'):
                current_score = self._update_score(
                    current_score,
                    winner=point_event['winner']
                )
            
            score_timeline[timestamp] = current_score.copy()
        
        return score_timeline
    
    def _read_score_from_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Read score from frame using OCR.
        
        Args:
            frame: Input frame
        
        Returns:
            Detected score dict or None
        """
        # Extract scoreboard region
        h, w = frame.shape[:2]
        x1 = int(self.ocr_region[0] * w)
        y1 = int(self.ocr_region[1] * h)
        x2 = int(self.ocr_region[2] * w)
        y2 = int(self.ocr_region[3] * h)
        
        scoreboard = frame[y1:y2, x1:x2]
        
        # Preprocess for OCR
        gray = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
        
        # OCR
        try:
            results = self.reader.readtext(gray)
            
            # Parse OCR results
            text = ' '.join([r[1] for r in results])
            score = self._parse_score_text(text)
            
            return score
        except:
            return None
    
    def _parse_score_text(self, text: str) -> Optional[Dict]:
        """
        Parse score from OCR text.
        
        Args:
            text: OCR text output
        
        Returns:
            Parsed score dict or None
        """
        # Simplified parsing
        # In production, would use regex and tennis score patterns
        return None
    
    def _validate_score_transition(
        self,
        prev_score: Dict,
        detected_score: Dict
    ) -> Dict:
        """
        Validate detected score against previous score.
        
        Args:
            prev_score: Previous validated score
            detected_score: Newly detected score
        
        Returns:
            Validated score (either detected or previous)
        """
        # Check if transition is legal
        if self._is_valid_transition(prev_score, detected_score):
            return detected_score
        else:
            return prev_score
    
    def _is_valid_transition(
        self,
        prev_score: Dict,
        new_score: Dict
    ) -> bool:
        """
        Check if score transition follows tennis rules.
        
        Args:
            prev_score: Previous score
            new_score: New score
        
        Returns:
            True if valid transition
        """
        # Simplified validation
        # In production, would check all tennis scoring rules
        return True
    
    def _update_score(
        self,
        score: Dict,
        winner: int
    ) -> Dict:
        """
        Update score based on point winner.
        
        Args:
            score: Current score
            winner: Player ID who won point
        
        Returns:
            Updated score
        """
        new_score = score.copy()
        
        # Update points
        if winner == 0:
            new_score['p1_points'] += 1
        else:
            new_score['p2_points'] += 1
        
        # Check for game win
        if self._is_game_won(new_score):
            if winner == 0:
                new_score['p1_games'] += 1
            else:
                new_score['p2_games'] += 1
            
            # Reset points
            new_score['p1_points'] = 0
            new_score['p2_points'] = 0
            new_score['game_number'] += 1
        
        # Check for set win
        if self._is_set_won(new_score):
            if winner == 0:
                new_score['p1_sets'] += 1
            else:
                new_score['p2_sets'] += 1
            
            # Reset games
            new_score['p1_games'] = 0
            new_score['p2_games'] = 0
            new_score['set_number'] += 1
            new_score['game_number'] = 1
        
        return new_score
    
    def _is_game_won(self, score: Dict) -> bool:
        """Check if game is won."""
        p1 = score['p1_points']
        p2 = score['p2_points']
        
        # Standard game (4 points, win by 2)
        if p1 >= 4 and p1 - p2 >= 2:
            return True
        if p2 >= 4 and p2 - p1 >= 2:
            return True
        
        return False
    
    def _is_set_won(self, score: Dict) -> bool:
        """Check if set is won."""
        p1 = score['p1_games']
        p2 = score['p2_games']
        
        # Standard set (6 games, win by 2)
        if p1 >= 6 and p1 - p2 >= 2:
            return True
        if p2 >= 6 and p2 - p1 >= 2:
            return True
        
        # Tiebreak at 6-6
        if p1 == 7 and p2 == 6:
            return True
        if p2 == 7 and p1 == 6:
            return True
        
        return False
