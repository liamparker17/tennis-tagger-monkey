"""
Event Detector Module

Detects high-level tennis events:
- Serves
- Rallies
- Points
- Games
- Sets

Uses temporal analysis and ball trajectory patterns.
"""

import numpy as np
from typing import List, Dict
import torch


class EventDetector:
    """
    Detect tennis match events from video analysis.
    
    Combines multiple signals:
    - Ball trajectory patterns
    - Player poses
    - Temporal patterns
    - Score changes
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize event detector.
        
        Args:
            config: Event detection configuration
            device: Torch device
        """
        self.config = config
        self.device = device
        self.serve_confidence = config['serve_detection']['confidence']
    
    def detect_events(
        self,
        frames: List[np.ndarray],
        tracks: List[Dict],
        strokes: List[Dict],
        fps: float
    ) -> Dict:
        """
        Detect all events in match footage.
        
        Args:
            frames: Video frames
            tracks: Tracking results
            strokes: Classified strokes
            fps: Frame rate
        
        Returns:
            Dictionary of detected events
        """
        # Detect serves
        serve_events = self._detect_serves(strokes, tracks, fps)
        
        # Detect rallies
        rally_events = self._detect_rallies(strokes, fps)
        
        # Detect points
        point_events = self._detect_points(strokes, rally_events, fps)
        
        # Detect games
        game_events = self._detect_games(point_events)
        
        # Detect sets
        set_events = self._detect_sets(game_events)
        
        return {
            'serves': len(serve_events),
            'rallies': len(rally_events),
            'points': len(point_events),
            'games': len(game_events),
            'sets': len(set_events),
            'serve_events': serve_events,
            'rally_events': rally_events,
            'point_events': point_events,
            'game_events': game_events,
            'set_events': set_events
        }
    
    def _detect_serves(
        self,
        strokes: List[Dict],
        tracks: List[Dict],
        fps: float
    ) -> List[Dict]:
        """
        Detect serve events.
        
        Args:
            strokes: Classified strokes
            tracks: Tracking data
            fps: Frame rate
        
        Returns:
            List of serve event dicts
        """
        serve_events = []
        
        for stroke in strokes:
            if 'Serve' in stroke.get('stroke_type', ''):
                serve_events.append({
                    'timestamp': stroke['timestamp'],
                    'frame': stroke['frame'],
                    'player_id': stroke['player_id'],
                    'confidence': stroke['confidence'],
                    'type': stroke['stroke_type']
                })
        
        return serve_events
    
    def _detect_rallies(
        self,
        strokes: List[Dict],
        fps: float
    ) -> List[Dict]:
        """
        Segment rallies based on stroke patterns.
        
        Args:
            strokes: Classified strokes
            fps: Frame rate
        
        Returns:
            List of rally dicts
        """
        rallies = []
        current_rally = []
        max_gap = self.config['rally_segmentation']['max_gap']
        
        for i, stroke in enumerate(strokes):
            if len(current_rally) == 0:
                current_rally.append(stroke)
            else:
                # Check time gap
                time_gap = stroke['timestamp'] - current_rally[-1]['timestamp']
                
                if time_gap <= max_gap:
                    current_rally.append(stroke)
                else:
                    # End current rally
                    if len(current_rally) >= self.config['rally_segmentation']['min_strokes']:
                        rallies.append({
                            'start_time': current_rally[0]['timestamp'],
                            'end_time': current_rally[-1]['timestamp'],
                            'strokes': current_rally,
                            'length': len(current_rally)
                        })
                    
                    # Start new rally
                    current_rally = [stroke]
        
        # Add last rally
        if len(current_rally) >= self.config['rally_segmentation']['min_strokes']:
            rallies.append({
                'start_time': current_rally[0]['timestamp'],
                'end_time': current_rally[-1]['timestamp'],
                'strokes': current_rally,
                'length': len(current_rally)
            })
        
        return rallies
    
    def _detect_points(
        self,
        strokes: List[Dict],
        rallies: List[Dict],
        fps: float
    ) -> List[Dict]:
        """
        Detect individual points.
        
        Args:
            strokes: Classified strokes
            rallies: Detected rallies
            fps: Frame rate
        
        Returns:
            List of point dicts
        """
        points = []
        
        # Each rally typically corresponds to a point
        for rally in rallies:
            # Determine server/returner
            if len(rally['strokes']) > 0:
                first_stroke = rally['strokes'][0]
                
                if 'Serve' in first_stroke.get('stroke_type', ''):
                    server = first_stroke['player_id']
                    returner = self._get_other_player(rally['strokes'], server)
                else:
                    # Couldn't identify serve
                    server = None
                    returner = None
                
                # Determine point winner (simplified - would use ball trajectory)
                winner = self._determine_point_winner(rally['strokes'])
                
                points.append({
                    'start_time': rally['start_time'],
                    'end_time': rally['end_time'],
                    'server': server,
                    'returner': returner,
                    'winner': winner,
                    'rally_length': rally['length']
                })
        
        return points
    
    def _detect_games(self, points: List[Dict]) -> List[Dict]:
        """
        Detect game boundaries.
        
        Args:
            points: Detected points
        
        Returns:
            List of game dicts
        """
        games = []
        current_game_points = []
        
        # Simple heuristic: group ~4-10 points per game
        for i, point in enumerate(points):
            current_game_points.append(point)
            
            # Detect game end (simplified)
            if len(current_game_points) >= 4:
                if self._is_game_complete(current_game_points):
                    games.append({
                        'start_time': current_game_points[0]['start_time'],
                        'end_time': current_game_points[-1]['end_time'],
                        'points': current_game_points,
                        'num_points': len(current_game_points)
                    })
                    current_game_points = []
        
        # Add last game
        if current_game_points:
            games.append({
                'start_time': current_game_points[0]['start_time'],
                'end_time': current_game_points[-1]['end_time'],
                'points': current_game_points,
                'num_points': len(current_game_points)
            })
        
        return games
    
    def _detect_sets(self, games: List[Dict]) -> List[Dict]:
        """
        Detect set boundaries.
        
        Args:
            games: Detected games
        
        Returns:
            List of set dicts
        """
        sets = []
        current_set_games = []
        
        # Simple heuristic: ~6+ games per set
        for game in games:
            current_set_games.append(game)
            
            if len(current_set_games) >= 6:
                if self._is_set_complete(current_set_games):
                    sets.append({
                        'start_time': current_set_games[0]['start_time'],
                        'end_time': current_set_games[-1]['end_time'],
                        'games': current_set_games,
                        'num_games': len(current_set_games)
                    })
                    current_set_games = []
        
        # Add last set
        if current_set_games:
            sets.append({
                'start_time': current_set_games[0]['start_time'],
                'end_time': current_set_games[-1]['end_time'],
                'games': current_set_games,
                'num_games': len(current_set_games)
            })
        
        return sets
    
    def _get_other_player(
        self,
        strokes: List[Dict],
        player_id: int
    ) -> int:
        """Get ID of other player in rally."""
        for stroke in strokes:
            if stroke['player_id'] != player_id:
                return stroke['player_id']
        return None
    
    def _determine_point_winner(self, strokes: List[Dict]) -> int:
        """Determine which player won the point."""
        # Simplified: alternate between players, last stroke = winner
        if strokes:
            return strokes[-1]['player_id']
        return None
    
    def _is_game_complete(self, points: List[Dict]) -> bool:
        """Check if game is complete (simplified)."""
        # In production, would track actual score
        return len(points) >= 4
    
    def _is_set_complete(self, games: List[Dict]) -> bool:
        """Check if set is complete (simplified)."""
        # In production, would track actual score
        return len(games) >= 6
