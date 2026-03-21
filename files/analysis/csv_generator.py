"""
Dartfish CSV Generator

Generates tennis match tagging CSVs in exact Dartfish format with
all required columns for professional tennis analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta


class DartfishCSVGenerator:
    """
    Generate Dartfish-compatible CSV files from tennis analysis data.
    
    Produces CSV with comprehensive tagging including:
    - Point-level data
    - Serve and return information
    - Stroke sequences
    - Placements and depths
    - Scores and outcomes
    """
    
    # Dartfish column structure
    COLUMNS = [
        # Basic info
        "Name", "Position", "Duration",
        
        # Point level
        "0 - Point Level",
        
        # Server info (A)
        "A1: Server", "A2: Serve Data", "A3: Serve Placement",
        
        # Returner info (B)
        "B1: Returner", "B2: Return Data", "B3: Return Placement",
        
        # Serve +1 (C)
        "C1: Serve +1 Stroke", "C2: Serve +1 Data", "C3: Serve +1 Placement",
        
        # Return +1 (D)
        "D1: Return +1 Stroke", "D2: Return +1 Data", "D3: Return +1 Placement",
        
        # Last shot (E)
        "E1: Last Shot", "E2: Last Shot Winner", "E3: Last Shot Error", "E4: Last Shot Placement",
        
        # Point outcome (F)
        "F1: Point Won", "F2: Point Score",
        
        # Rally data (G)
        "G1: Rally Length", "G2: Total Strokes",
        
        # Net appearances (H)
        "H1: Server Net", "H2: Returner Net",
        
        # Additional metrics (I-Z)
        "I1: Winner Type", "I2: Error Type",
        "J1: Break Point", "J2: Set Point", "J3: Match Point",
        "K1: Deuce/Ad", "K2: Game Score",
        "L1: Set Number", "L2: Game Number",
        "M1: Serve Speed", "M2: Serve Number",
        "N1: First Serve In", "N2: Second Serve",
        "O1: Return Quality", "O2: Return Depth",
        "P1: Approach Shot", "P2: Passing Shot",
        "Q1: Court Position Server", "Q2: Court Position Returner",
        "R1: Dominant Hand Server", "R2: Dominant Hand Returner",
        "S1: Time Between Points",
        "T1: Video Timestamp", "T2: Frame Number",
        "U1: Confidence Score",
        "V1: Notes", "V2: Tags",
        "W1: Weather Conditions", "W2: Court Surface",
        "X1: Player 1 Name", "X2: Player 2 Name",
        "Y1: Tournament", "Y2: Round",
        "Z1: Serve Contact Depth", "Z2: Return Contact Depth",
        "Z3: Serve +1 Contact Depth"
    ]
    
    def __init__(self, config: Dict):
        """
        Initialize CSV generator.
        
        Args:
            config: Export configuration
        """
        self.config = config
    
    def generate(
        self,
        events: Dict,
        strokes: List[Dict],
        scores: Dict,
        court_info: Dict,
        video_metadata: Dict
    ) -> pd.DataFrame:
        """
        Generate complete Dartfish CSV from analysis data.
        
        Args:
            events: Detected events (serves, rallies, points)
            strokes: Classified strokes
            scores: Score tracking timeline
            court_info: Court detection information
            video_metadata: Video file metadata
        
        Returns:
            DataFrame with Dartfish format
        """
        rows = []
        
        # Get point-level events
        points = self._extract_points(events, strokes, scores)
        
        for point_idx, point in enumerate(points):
            row = self._create_point_row(
                point, point_idx, court_info, video_metadata
            )
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        
        return df
    
    def _extract_points(
        self,
        events: Dict,
        strokes: List[Dict],
        scores: Dict
    ) -> List[Dict]:
        """
        Extract point-level data from events and strokes.
        
        Args:
            events: Event detection results
            strokes: Stroke classification results
            scores: Score tracking data
        
        Returns:
            List of point dictionaries
        """
        points = []
        
        # Group strokes by point
        point_events = events.get('point_events', [])
        
        for point_event in point_events:
            start_time = point_event['start_time']
            end_time = point_event['end_time']
            
            # Find strokes in this point
            point_strokes = [
                s for s in strokes
                if start_time <= s['timestamp'] <= end_time
            ]
            
            # Extract point data
            point_data = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'strokes': point_strokes,
                'server': point_event.get('server'),
                'returner': point_event.get('returner'),
                'winner': point_event.get('winner'),
                'score': scores.get(start_time, {}),
                'rally_length': len(point_strokes),
                'serve_info': self._extract_serve_info(point_strokes),
                'return_info': self._extract_return_info(point_strokes),
                'last_shot': point_strokes[-1] if point_strokes else None
            }
            
            points.append(point_data)
        
        return points
    
    def _create_point_row(
        self,
        point: Dict,
        point_idx: int,
        court_info: Dict,
        video_metadata: Dict
    ) -> List:
        """
        Create a single CSV row for a point.
        
        Args:
            point: Point data dictionary
            point_idx: Point number
            court_info: Court information
            video_metadata: Video metadata
        
        Returns:
            List of values for CSV row
        """
        row = [""] * len(self.COLUMNS)
        
        # Basic info
        row[0] = f"Point {point_idx + 1}"  # Name
        row[1] = self._format_timestamp(point['start_time'])  # Position
        row[2] = f"{point['duration']:.2f}s"  # Duration
        
        # Point level
        row[3] = self._determine_point_level(point['score'])
        
        # Server info (A)
        row[4] = point.get('server', '')  # A1: Server
        serve_info = point.get('serve_info', {})
        row[5] = serve_info.get('type', '')  # A2: Serve Data
        row[6] = serve_info.get('placement', '')  # A3: Serve Placement
        
        # Returner info (B)
        row[7] = point.get('returner', '')  # B1: Returner
        return_info = point.get('return_info', {})
        row[8] = return_info.get('type', '')  # B2: Return Data
        row[9] = return_info.get('placement', '')  # B3: Return Placement
        
        # Serve +1 (C)
        if len(point['strokes']) > 2:
            serve_plus1 = point['strokes'][2]
            row[10] = serve_plus1.get('stroke_type', '')
            row[11] = self._format_stroke_data(serve_plus1)
            row[12] = self._determine_placement(serve_plus1, court_info)
        
        # Return +1 (D)
        if len(point['strokes']) > 3:
            return_plus1 = point['strokes'][3]
            row[13] = return_plus1.get('stroke_type', '')
            row[14] = self._format_stroke_data(return_plus1)
            row[15] = self._determine_placement(return_plus1, court_info)
        
        # Last shot (E)
        last_shot = point.get('last_shot')
        if last_shot:
            row[16] = last_shot.get('stroke_type', '')
            row[17] = self._is_winner(point)
            row[18] = self._determine_error_type(point)
            row[19] = self._determine_placement(last_shot, court_info)
        
        # Point outcome (F)
        row[20] = point.get('winner', '')  # F1: Point Won
        row[21] = self._format_score(point['score'])  # F2: Point Score
        
        # Rally data (G)
        row[22] = str(point['rally_length'])  # G1: Rally Length
        row[23] = str(len(point['strokes']))  # G2: Total Strokes
        
        # Net appearances (H)
        row[24], row[25] = self._count_net_approaches(point['strokes'])
        
        # Additional metrics (I-Z)
        row[26] = self._classify_winner_type(point)  # I1: Winner Type
        row[27] = self._determine_error_type(point)  # I2: Error Type
        
        # Situational data (J)
        row[28] = self._is_break_point(point['score'])
        row[29] = self._is_set_point(point['score'])
        row[30] = self._is_match_point(point['score'])
        
        # Score context (K)
        row[31] = self._get_deuce_ad(point['score'])
        row[32] = self._get_game_score(point['score'])
        
        # Match position (L)
        row[33] = str(point['score'].get('set_number', 1))
        row[34] = str(point['score'].get('game_number', 1))
        
        # Serve details (M-N)
        row[35] = serve_info.get('speed', 'N/A')
        row[36] = serve_info.get('serve_number', '1st')
        row[37] = serve_info.get('first_serve_in', 'Yes')
        row[38] = serve_info.get('second_serve', 'N/A')
        
        # Return quality (O)
        row[39] = return_info.get('quality', '')
        row[40] = return_info.get('depth', '')
        
        # Shot types (P)
        row[41] = self._detect_approach_shot(point['strokes'])
        row[42] = self._detect_passing_shot(point['strokes'])
        
        # Court positions (Q)
        row[43] = self._get_court_position(point['strokes'], 'server')
        row[44] = self._get_court_position(point['strokes'], 'returner')
        
        # Player info (R)
        row[45] = point.get('server_hand', 'Right')
        row[46] = point.get('returner_hand', 'Right')
        
        # Timing (S)
        row[47] = self._time_between_points(point_idx, point)
        
        # Technical (T-U)
        row[48] = self._format_timestamp(point['start_time'])
        row[49] = str(int(point['start_time'] * 30))  # Assuming 30 FPS
        row[50] = f"{self._calculate_confidence(point):.2f}"
        
        # Metadata (V-Y)
        row[51] = ""  # Notes
        row[52] = ""  # Tags
        row[53] = video_metadata.get('weather', 'Indoor')
        row[54] = video_metadata.get('surface', 'Hard')
        row[55] = video_metadata.get('player1', 'Player 1')
        row[56] = video_metadata.get('player2', 'Player 2')
        row[57] = video_metadata.get('tournament', '')
        row[58] = video_metadata.get('round', '')
        
        # Contact depths (Z)
        row[59] = self._get_contact_depth(serve_info)
        row[60] = self._get_contact_depth(return_info)
        row[61] = self._get_contact_depth(point['strokes'][2] if len(point['strokes']) > 2 else {})
        
        return row
    
    def _extract_serve_info(self, strokes: List[Dict]) -> Dict:
        """Extract serve information from strokes."""
        if len(strokes) > 0 and 'Serve' in strokes[0].get('stroke_type', ''):
            return {
                'type': strokes[0]['stroke_type'],
                'placement': 'Wide',  # Would be determined from ball position
                'speed': 'Fast',
                'serve_number': '1st',
                'first_serve_in': 'Yes'
            }
        return {}
    
    def _extract_return_info(self, strokes: List[Dict]) -> Dict:
        """Extract return information."""
        if len(strokes) > 1:
            return {
                'type': strokes[1].get('stroke_type', ''),
                'placement': 'Cross-court',
                'quality': 'Neutral',
                'depth': 'Deep'
            }
        return {}
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        secs = total_seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def _determine_point_level(self, score: Dict) -> str:
        """Determine point importance level."""
        if self._is_break_point(score) == "Yes":
            return "Critical"
        elif self._is_set_point(score) == "Yes":
            return "Very Important"
        else:
            return "Normal"
    
    def _format_score(self, score: Dict) -> str:
        """Format score string."""
        return f"{score.get('p1_points', 0)}-{score.get('p2_points', 0)}"
    
    def _determine_placement(self, stroke: Dict, court_info: Dict) -> str:
        """Determine shot placement zone."""
        # Would use ball position and court zones
        placements = ["Wide", "Body", "T", "Down the Line", "Cross-court"]
        return np.random.choice(placements)  # Placeholder
    
    def _format_stroke_data(self, stroke: Dict) -> str:
        """Format stroke data string."""
        return f"{stroke.get('stroke_type', 'Unknown')} - {stroke.get('confidence', 0):.2f}"
    
    def _is_winner(self, point: Dict) -> str:
        """Determine if last shot was a winner."""
        return "Yes" if point.get('winner') else "No"
    
    def _determine_error_type(self, point: Dict) -> str:
        """Classify error type if applicable."""
        error_types = ["None", "Net", "Long", "Wide", "Forced", "Unforced"]
        return error_types[0]  # Placeholder
    
    def _count_net_approaches(self, strokes: List[Dict]) -> tuple:
        """Count net approaches for each player."""
        return "0", "0"  # Placeholder
    
    def _classify_winner_type(self, point: Dict) -> str:
        """Classify type of winner."""
        winner_types = ["None", "Ace", "Service Winner", "Groundstroke", "Volley", "Smash"]
        return winner_types[0]  # Placeholder
    
    def _is_break_point(self, score: Dict) -> str:
        """Check if break point."""
        return "No"  # Placeholder
    
    def _is_set_point(self, score: Dict) -> str:
        """Check if set point."""
        return "No"  # Placeholder
    
    def _is_match_point(self, score: Dict) -> str:
        """Check if match point."""
        return "No"  # Placeholder
    
    def _get_deuce_ad(self, score: Dict) -> str:
        """Get deuce/ad status."""
        return ""  # Placeholder
    
    def _get_game_score(self, score: Dict) -> str:
        """Get current game score."""
        return "0-0"  # Placeholder
    
    def _detect_approach_shot(self, strokes: List[Dict]) -> str:
        """Detect if approach shot occurred."""
        return "No"  # Placeholder
    
    def _detect_passing_shot(self, strokes: List[Dict]) -> str:
        """Detect if passing shot occurred."""
        return "No"  # Placeholder
    
    def _get_court_position(self, strokes: List[Dict], player: str) -> str:
        """Get player's court position."""
        positions = ["Baseline", "Mid-court", "Net"]
        return positions[0]  # Placeholder
    
    def _time_between_points(self, point_idx: int, point: Dict) -> str:
        """Calculate time between points."""
        return "20s"  # Placeholder
    
    def _calculate_confidence(self, point: Dict) -> float:
        """Calculate overall confidence for point tagging."""
        if not point['strokes']:
            return 0.0
        
        confidences = [s.get('confidence', 0) for s in point['strokes']]
        return np.mean(confidences) if confidences else 0.0
    
    def _get_contact_depth(self, info: Dict) -> str:
        """Get contact depth for stroke."""
        depths = ["Baseline", "Mid-court", "Net"]
        return depths[0]  # Placeholder
