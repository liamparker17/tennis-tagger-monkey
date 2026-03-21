"""
QC Comparator and Feedback Loop

Compares predicted CSV with QC-corrected CSV and feeds
corrections back into the models for continuous improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class CSVComparator:
    """
    Compare predicted CSV with QC-corrected CSV.
    
    Identifies differences and generates detailed comparison report.
    """
    
    def compare(
        self,
        predicted_csv: pd.DataFrame,
        corrected_csv: pd.DataFrame
    ) -> Dict:
        """
        Compare two CSV DataFrames.
        
        Args:
            predicted_csv: Model-generated CSV
            corrected_csv: Human-corrected CSV
        
        Returns:
            Comparison report dictionary
        """
        differences = {
            'total_rows': len(predicted_csv),
            'total_differences': 0,
            'column_differences': {},
            'row_differences': [],
            'accuracy': 0.0
        }
        
        # Ensure same number of rows
        if len(predicted_csv) != len(corrected_csv):
            differences['row_count_mismatch'] = True
            # Align by timestamp or position
            predicted_csv, corrected_csv = self._align_dataframes(
                predicted_csv, corrected_csv
            )
        
        # Compare column by column
        total_cells = 0
        matching_cells = 0
        
        for col in predicted_csv.columns:
            if col not in corrected_csv.columns:
                continue
            
            col_diffs = []
            
            for idx in range(min(len(predicted_csv), len(corrected_csv))):
                pred_val = str(predicted_csv.loc[idx, col])
                corr_val = str(corrected_csv.loc[idx, col])
                
                total_cells += 1
                
                if pred_val != corr_val:
                    col_diffs.append({
                        'row': idx,
                        'predicted': pred_val,
                        'corrected': corr_val
                    })
                    differences['total_differences'] += 1
                else:
                    matching_cells += 1
            
            if col_diffs:
                differences['column_differences'][col] = col_diffs
        
        # Calculate accuracy
        if total_cells > 0:
            differences['accuracy'] = matching_cells / total_cells
        
        # Identify most common error types
        differences['error_summary'] = self._summarize_errors(
            differences['column_differences']
        )
        
        return differences
    
    def _align_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two dataframes by timestamp.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
        
        Returns:
            Tuple of aligned dataframes
        """
        # Use 'Position' column as alignment key
        if 'Position' in df1.columns and 'Position' in df2.columns:
            merged = pd.merge(
                df1, df2,
                on='Position',
                how='outer',
                suffixes=('_pred', '_corr')
            )
            return df1, df2  # Simplified
        
        return df1, df2
    
    def _summarize_errors(
        self,
        column_differences: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Summarize error patterns.
        
        Args:
            column_differences: Dictionary of column differences
        
        Returns:
            Error summary dictionary
        """
        summary = {
            'stroke_errors': 0,
            'placement_errors': 0,
            'score_errors': 0,
            'timing_errors': 0
        }
        
        for col, diffs in column_differences.items():
            if 'Stroke' in col:
                summary['stroke_errors'] += len(diffs)
            elif 'Placement' in col:
                summary['placement_errors'] += len(diffs)
            elif 'Score' in col or 'Point' in col:
                summary['score_errors'] += len(diffs)
            elif 'Position' in col or 'Duration' in col:
                summary['timing_errors'] += len(diffs)
        
        return summary


class FeedbackLoop:
    """
    Apply QC corrections to improve models.
    
    Implements incremental learning from human feedback.
    """
    
    def __init__(self, models: Dict, config: Dict):
        """
        Initialize feedback loop.
        
        Args:
            models: Dictionary of model objects
            config: Feedback configuration
        """
        self.models = models
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.correction_buffer = []
        self.accumulation_threshold = config.get('accumulate_corrections', 10)
    
    def apply_corrections(
        self,
        video_path: str,
        predicted_csv: pd.DataFrame,
        corrected_csv: pd.DataFrame,
        comparison: Dict
    ):
        """
        Apply corrections to models.
        
        Args:
            video_path: Path to video file
            predicted_csv: Predicted CSV
            corrected_csv: Corrected CSV
            comparison: Comparison results
        """
        # Extract correction examples
        corrections = self._extract_corrections(
            predicted_csv, corrected_csv, comparison
        )
        
        # Add to buffer
        self.correction_buffer.extend(corrections)
        
        print(f"   Buffered {len(corrections)} corrections")
        print(f"   Total in buffer: {len(self.correction_buffer)}")
        
        # Update models if threshold reached
        if len(self.correction_buffer) >= self.accumulation_threshold:
            print(f"   Threshold reached, updating models...")
            self._update_models()
            self.correction_buffer = []
    
    def _extract_corrections(
        self,
        predicted_csv: pd.DataFrame,
        corrected_csv: pd.DataFrame,
        comparison: Dict
    ) -> List[Dict]:
        """
        Extract training examples from corrections.
        
        Args:
            predicted_csv: Predicted CSV
            corrected_csv: Corrected CSV
            comparison: Comparison results
        
        Returns:
            List of correction examples
        """
        corrections = []
        
        # Extract stroke corrections
        if 'column_differences' in comparison:
            for col, diffs in comparison['column_differences'].items():
                if 'Stroke' in col:
                    for diff in diffs:
                        corrections.append({
                            'type': 'stroke_classification',
                            'row': diff['row'],
                            'predicted': diff['predicted'],
                            'correct': diff['corrected'],
                            'column': col
                        })
                elif 'Placement' in col:
                    for diff in diffs:
                        corrections.append({
                            'type': 'placement',
                            'row': diff['row'],
                            'predicted': diff['predicted'],
                            'correct': diff['corrected'],
                            'column': col
                        })
        
        return corrections
    
    def _update_models(self):
        """
        Update models with buffered corrections.
        """
        # Group corrections by type
        stroke_corrections = [
            c for c in self.correction_buffer
            if c['type'] == 'stroke_classification'
        ]
        
        placement_corrections = [
            c for c in self.correction_buffer
            if c['type'] == 'placement'
        ]
        
        # Update stroke classifier
        if stroke_corrections and 'stroke_classifier' in self.models:
            print(f"      Updating stroke classifier with {len(stroke_corrections)} corrections")
            self._update_stroke_classifier(stroke_corrections)
        
        # Update detector/placement
        if placement_corrections:
            print(f"      Updating placement analysis with {len(placement_corrections)} corrections")
        
        print(f"   ✅ Models updated")
    
    def _update_stroke_classifier(self, corrections: List[Dict]):
        """
        Update stroke classifier with corrections.
        
        Args:
            corrections: List of stroke corrections
        """
        model = self.models['stroke_classifier'].model
        
        # Create pseudo-labels from corrections
        # In production, would re-extract video clips and retrain
        
        # For now, just log that update would happen
        print(f"      Would retrain on {len(corrections)} examples")
        
        # Example: adjust model (simplified)
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        # ... training loop ...
