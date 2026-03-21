"""
CSV Generator Module
Generates Dartfish-compatible CSV files from analysis results
"""

import pandas as pd
import logging
from typing import Dict, List
from datetime import timedelta


class CSVGenerator:
    """Generate Dartfish-compatible CSV output"""
    
    # Define all CSV columns matching Dartfish format
    CSV_COLUMNS = [
        'Name', 'Position', 'Duration',
        '0 - Point Level',
        'A1: Server', 'A2: Serve Data', 'A3: Serve Placement',
        'B1: Returner', 'B2: Return Data', 'B3: Return Placement',
        'C1: Serve +1 Stroke', 'C2: Serve +1 Data', 'C3: Serve +1 Placement',
        'D1: Return +1 Stroke', 'D2: Return +1 Data', 'D3: Return +1 Placement',
        'E1: Last Shot', 'E2: Last Shot Winner', 'E3: Last Shot Error', 
        'E4: Last Shot Placement',
        'F1: Point Won', 'F2: Point Score', 'F3: Game Score', 'F4: Set Score',
        'G1: Rally Length', 'G2: Stroke Count',
        'H1: Net Appearance', 'H2: Net Winner', 'H3: Net Error',
        'I1: Fault Type', 'I2: Double Fault',
        'J1: Break Point', 'J2: BP Won', 'J3: BP Lost',
        'K1: Deuce', 'K2: Ad Point',
        'L1: Tiebreak', 'L2: TB Point',
        'M1: First Serve In', 'M2: Second Serve In',
        'N1: Ace', 'N2: Service Winner',
        'O1: Return Winner', 'O2: Passing Shot',
        'P1: Unforced Error', 'P2: Forced Error',
        'Q1: Winner Type', 'Q2: Error Type',
        'R1: Approach Shot', 'R2: Drop Shot', 'R3: Lob',
        'S1: Forehand Count', 'S2: Backhand Count', 'S3: Volley Count',
        'T1: Court Position Server', 'T2: Court Position Returner',
        'U1: Rally Pattern', 'U2: Stroke Sequence',
        'V1: Time Between Points', 'V2: Point Duration',
        'W1: Player 1 Name', 'W2: Player 2 Name',
        'X1: Weather Condition', 'X2: Court Surface',
        'Y1: Match Type', 'Y2: Round',
        'Z1: Serve Contact Height', 'Z2: Serve +1 Contact Depth'
    ]
    
    def __init__(self, config: dict):
        self.config = config
        self.csv_config = config.get('csv', {})
        self.logger = logging.getLogger('CSVGenerator')
        self.format_type = self.csv_config.get('format', 'dartfish')
    
    def generate(self, metadata: Dict, serves: List[Dict],
                strokes: List[Dict], rallies: List[Dict],
                scores: List[Dict], placements: List[Dict]) -> pd.DataFrame:
        """
        Generate Dartfish-compatible CSV from analysis results
        
        Args:
            metadata: Video metadata
            serves: Detected serves
            strokes: Classified strokes
            rallies: Segmented rallies
            scores: Score history
            placements: Shot placements
            
        Returns:
            pandas DataFrame ready for CSV export
        """
        self.logger.info("Generating CSV output...")
        
        # Initialize data rows
        rows = []
        
        # Process each rally as a point
        for rally_idx, rally in enumerate(rallies):
            row = self._create_row_template()
            
            # Basic information
            row['Name'] = f"Point_{rally_idx + 1}"
            row['Position'] = self._format_timestamp(rally['start_frame'], metadata)
            row['Duration'] = self._calculate_duration(rally, metadata)
            
            # Point level
            row['0 - Point Level'] = 'Standard'
            
            # Get serve for this rally
            serve = self._find_serve_for_rally(rally, serves)
            rally_strokes = rally.get('strokes', [])
            
            # A: Server information
            if serve:
                row['A1: Server'] = f"Player_{serve['player_id']}"
                row['A2: Serve Data'] = self._classify_serve_type(serve)
                row['A3: Serve Placement'] = self._get_placement_for_stroke(
                    serve['frame'], placements
                )
            
            # B: Returner information
            if len(rally_strokes) > 0:
                return_stroke = rally_strokes[0]
                returner_id = self._get_other_player(serve.get('player_id', 1) if serve else 1)
                row['B1: Returner'] = f"Player_{returner_id}"
                row['B2: Return Data'] = return_stroke.get('type', 'Unknown').title()
                row['B3: Return Placement'] = self._get_placement_for_stroke(
                    return_stroke['frame'], placements
                )
            
            # C: Serve +1 Stroke
            if len(rally_strokes) > 1 and serve:
                serve_plus_one = rally_strokes[1]
                if serve_plus_one['player_id'] == serve['player_id']:
                    row['C1: Serve +1 Stroke'] = serve_plus_one.get('type', '').title()
                    row['C2: Serve +1 Data'] = self._get_stroke_data(serve_plus_one)
                    row['C3: Serve +1 Placement'] = self._get_placement_for_stroke(
                        serve_plus_one['frame'], placements
                    )
            
            # D: Return +1 Stroke
            if len(rally_strokes) > 2:
                return_plus_one = rally_strokes[2]
                row['D1: Return +1 Stroke'] = return_plus_one.get('type', '').title()
                row['D2: Return +1 Data'] = self._get_stroke_data(return_plus_one)
                row['D3: Return +1 Placement'] = self._get_placement_for_stroke(
                    return_plus_one['frame'], placements
                )
            
            # E: Last Shot
            if rally_strokes:
                last_stroke = rally_strokes[-1]
                row['E1: Last Shot'] = last_stroke.get('type', '').title()
                row['E2: Last Shot Winner'] = self._is_winner(last_stroke)
                row['E3: Last Shot Error'] = self._is_error(last_stroke)
                row['E4: Last Shot Placement'] = self._get_placement_for_stroke(
                    last_stroke['frame'], placements
                )
            
            # F: Point outcome
            score_info = self._find_score_for_rally(rally_idx, scores)
            if score_info:
                row['F1: Point Won'] = f"Player_{score_info['point_winner']}"
                row['F2: Point Score'] = self._format_point_score(score_info['score'])
                row['F3: Game Score'] = self._format_game_score(score_info['score'])
                row['F4: Set Score'] = self._format_set_score(score_info['score'])
            
            # G: Rally statistics
            row['G1: Rally Length'] = rally['duration_frames']
            row['G2: Stroke Count'] = rally['num_strokes']
            
            # S: Stroke counts
            stroke_counts = self._count_stroke_types(rally_strokes)
            row['S1: Forehand Count'] = stroke_counts.get('forehand', 0)
            row['S2: Backhand Count'] = stroke_counts.get('backhand', 0)
            row['S3: Volley Count'] = stroke_counts.get('volley', 0)
            
            # V: Timing
            row['V2: Point Duration'] = self._calculate_duration(rally, metadata)
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=self.CSV_COLUMNS)
        
        self.logger.info(f"Generated CSV with {len(df)} rows")
        return df
    
    def _create_row_template(self) -> Dict:
        """Create empty row with all columns"""
        return {col: '' for col in self.CSV_COLUMNS}
    
    def _format_timestamp(self, frame: int, metadata: Dict) -> str:
        """Convert frame number to timestamp"""
        fps = metadata.get('target_fps', 30)
        seconds = frame / fps
        return str(timedelta(seconds=int(seconds)))
    
    def _calculate_duration(self, rally: Dict, metadata: Dict) -> float:
        """Calculate rally duration in seconds"""
        fps = metadata.get('target_fps', 30)
        return rally['duration_frames'] / fps
    
    def _find_serve_for_rally(self, rally: Dict, 
                              serves: List[Dict]) -> Dict:
        """Find the serve that started this rally"""
        rally_start = rally['start_frame']
        for serve in serves:
            if abs(serve['frame'] - rally_start) < 30:  # Within 1 second
                return serve
        return None
    
    def _classify_serve_type(self, serve: Dict) -> str:
        """Classify serve type (First, Second, Ace, etc.)"""
        # Simplified classification
        return "First Serve"
    
    def _get_placement_for_stroke(self, frame: int, 
                                  placements: List[Dict]) -> str:
        """Get court placement for stroke"""
        for placement in placements:
            if placement['stroke_frame'] == frame:
                region = placement.get('court_region', 5)
                return f"Region_{region}"
        return ""
    
    def _get_other_player(self, player_id: int) -> int:
        """Get the other player's ID"""
        return 2 if player_id == 1 else 1
    
    def _get_stroke_data(self, stroke: Dict) -> str:
        """Get additional stroke data"""
        return f"{stroke.get('type', 'Unknown').title()} (conf: {stroke.get('confidence', 0):.2f})"
    
    def _is_winner(self, stroke: Dict) -> str:
        """Determine if stroke was a winner"""
        # Placeholder logic
        return "Yes" if stroke.get('confidence', 0) > 0.8 else ""
    
    def _is_error(self, stroke: Dict) -> str:
        """Determine if stroke was an error"""
        # Placeholder logic
        return ""
    
    def _find_score_for_rally(self, rally_idx: int, 
                             scores: List[Dict]) -> Dict:
        """Find score information for rally"""
        if rally_idx < len(scores):
            return scores[rally_idx]
        return None
    
    def _format_point_score(self, score: Dict) -> str:
        """Format point score (0, 15, 30, 40, AD)"""
        p1_pts = score['player1_points']
        p2_pts = score['player2_points']
        
        # Tennis scoring
        points_map = {0: '0', 1: '15', 2: '30', 3: '40'}
        
        if p1_pts < 4 and p2_pts < 4:
            return f"{points_map.get(p1_pts, '40')}-{points_map.get(p2_pts, '40')}"
        else:
            return "Deuce" if p1_pts == p2_pts else f"AD-{1 if p1_pts > p2_pts else 2}"
    
    def _format_game_score(self, score: Dict) -> str:
        """Format game score"""
        return f"{score['player1_games']}-{score['player2_games']}"
    
    def _format_set_score(self, score: Dict) -> str:
        """Format set score"""
        return f"{score['player1_sets']}-{score['player2_sets']}"
    
    def _count_stroke_types(self, strokes: List[Dict]) -> Dict:
        """Count occurrences of each stroke type"""
        counts = {}
        for stroke in strokes:
            stroke_type = stroke.get('type', 'unknown')
            counts[stroke_type] = counts.get(stroke_type, 0) + 1
        return counts
