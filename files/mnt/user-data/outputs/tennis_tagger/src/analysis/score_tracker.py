"""
Score Tracking Module
Tracks and maintains match score throughout the video
"""

import logging
from typing import List, Dict


class ScoreTracker:
    """Track match score"""
    
    def __init__(self, config: dict):
        self.config = config
        self.score_config = config.get('events', {}).get('score_tracking', {})
        self.logger = logging.getLogger('ScoreTracker')
        self.enabled = self.score_config.get('enabled', True)
    
    def track(self, rallies: List[Dict], serves: List[Dict]) -> List[Dict]:
        """Track score progression"""
        if not self.enabled:
            return []
        
        # Initialize score
        score_history = []
        current_score = {
            'player1_points': 0,
            'player2_points': 0,
            'player1_games': 0,
            'player2_games': 0,
            'player1_sets': 0,
            'player2_sets': 0
        }
        
        # Process each rally
        for rally_idx, rally in enumerate(rallies):
            # Determine point winner (simplified logic)
            winner = self._determine_winner(rally, serves)
            
            # Update points
            if winner == 1:
                current_score['player1_points'] += 1
            else:
                current_score['player2_points'] += 1
            
            # Check for game won
            if self._game_won(current_score):
                if winner == 1:
                    current_score['player1_games'] += 1
                else:
                    current_score['player2_games'] += 1
                current_score['player1_points'] = 0
                current_score['player2_points'] = 0
            
            # Check for set won
            if self._set_won(current_score):
                if current_score['player1_games'] > current_score['player2_games']:
                    current_score['player1_sets'] += 1
                else:
                    current_score['player2_sets'] += 1
                current_score['player1_games'] = 0
                current_score['player2_games'] = 0
            
            score_history.append({
                'rally_idx': rally_idx,
                'frame': rally['end_frame'],
                'score': current_score.copy(),
                'point_winner': winner
            })
        
        return score_history
    
    def _determine_winner(self, rally: Dict, serves: List[Dict]) -> int:
        """Determine point winner from rally (simplified)"""
        # In a real implementation, this would analyze the last stroke,
        # detect errors, winners, etc.
        # For now, alternating winners as placeholder
        return (rally['start_frame'] // 1000) % 2 + 1
    
    def _game_won(self, score: Dict) -> bool:
        """Check if a game is won"""
        p1_pts = score['player1_points']
        p2_pts = score['player2_points']
        
        if p1_pts >= 4 and p1_pts - p2_pts >= 2:
            return True
        if p2_pts >= 4 and p2_pts - p1_pts >= 2:
            return True
        return False
    
    def _set_won(self, score: Dict) -> bool:
        """Check if a set is won"""
        p1_games = score['player1_games']
        p2_games = score['player2_games']
        
        if p1_games >= 6 and p1_games - p2_games >= 2:
            return True
        if p2_games >= 6 and p2_games - p1_games >= 2:
            return True
        return False
