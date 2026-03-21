"""
Quality Control Feedback Module
Compares predicted CSVs with human-corrected CSVs and updates models
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import argparse


class QCFeedback:
    """Quality control and feedback loop"""
    
    def __init__(self, config: dict):
        self.config = config
        self.qc_config = config.get('qc', {})
        self.logger = logging.getLogger('QCFeedback')
        
        self.history_dir = Path('data/qc_history')
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_csvs(self, predicted_csv: str, 
                    corrected_csv: str) -> Dict:
        """
        Compare predicted CSV with human-corrected CSV
        
        Args:
            predicted_csv: Path to model-generated CSV
            corrected_csv: Path to human-corrected CSV
            
        Returns:
            Dictionary with comparison statistics
        """
        self.logger.info("Comparing predicted vs corrected CSVs...")
        
        # Load CSVs
        df_pred = pd.read_csv(predicted_csv)
        df_corr = pd.read_csv(corrected_csv)
        
        if len(df_pred) != len(df_corr):
            self.logger.warning(f"Row count mismatch: {len(df_pred)} vs {len(df_corr)}")
        
        # Compare each column
        column_accuracy = {}
        total_corrections = 0
        corrections_by_column = {}
        
        for col in df_pred.columns:
            if col not in df_corr.columns:
                continue
            
            # Calculate accuracy
            matches = (df_pred[col] == df_corr[col]).sum()
            total = len(df_pred)
            accuracy = matches / total if total > 0 else 0
            
            column_accuracy[col] = accuracy
            
            # Track corrections
            corrections = total - matches
            if corrections > 0:
                total_corrections += corrections
                corrections_by_column[col] = corrections
        
        # Overall accuracy
        overall_accuracy = np.mean(list(column_accuracy.values()))
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'predicted_file': predicted_csv,
            'corrected_file': corrected_csv,
            'total_rows': len(df_pred),
            'total_corrections': total_corrections,
            'overall_accuracy': overall_accuracy,
            'column_accuracy': column_accuracy,
            'corrections_by_column': corrections_by_column
        }
        
        self.logger.info(f"Overall accuracy: {overall_accuracy:.2%}")
        self.logger.info(f"Total corrections: {total_corrections}")
        
        return stats
    
    def identify_error_patterns(self, stats: Dict) -> Dict:
        """
        Identify systematic error patterns
        
        Args:
            stats: Comparison statistics
            
        Returns:
            Dictionary of error patterns
        """
        patterns = {
            'low_accuracy_columns': [],
            'high_correction_columns': [],
            'recommendations': []
        }
        
        # Find columns with low accuracy
        for col, acc in stats['column_accuracy'].items():
            if acc < 0.7:  # Less than 70% accuracy
                patterns['low_accuracy_columns'].append({
                    'column': col,
                    'accuracy': acc
                })
        
        # Find columns with many corrections
        corrections = stats['corrections_by_column']
        if corrections:
            avg_corrections = np.mean(list(corrections.values()))
            for col, count in corrections.items():
                if count > avg_corrections * 1.5:
                    patterns['high_correction_columns'].append({
                        'column': col,
                        'corrections': count
                    })
        
        # Generate recommendations
        if patterns['low_accuracy_columns']:
            patterns['recommendations'].append(
                "Consider retraining stroke classifier with more labeled data"
            )
        
        if any('Placement' in col for col in corrections.keys()):
            patterns['recommendations'].append(
                "Improve court detection and ball tracking for better placement accuracy"
            )
        
        if any('Score' in col for col in corrections.keys()):
            patterns['recommendations'].append(
                "Review score tracking logic and consider manual score input"
            )
        
        return patterns
    
    def generate_training_data(self, predicted_csv: str,
                              corrected_csv: str,
                              output_dir: str):
        """
        Generate training examples from corrections
        
        Args:
            predicted_csv: Path to predicted CSV
            corrected_csv: Path to corrected CSV
            output_dir: Directory to save training examples
        """
        self.logger.info("Generating training data from corrections...")
        
        df_pred = pd.read_csv(predicted_csv)
        df_corr = pd.read_csv(corrected_csv)
        
        training_examples = []
        
        for idx in range(len(df_pred)):
            row_pred = df_pred.iloc[idx]
            row_corr = df_corr.iloc[idx]
            
            # Find differences
            differences = {}
            for col in df_pred.columns:
                if col in df_corr.columns:
                    if row_pred[col] != row_corr[col]:
                        differences[col] = {
                            'predicted': row_pred[col],
                            'correct': row_corr[col]
                        }
            
            if differences:
                training_examples.append({
                    'row_index': idx,
                    'point_name': row_corr.get('Name', f'Point_{idx}'),
                    'corrections': differences
                })
        
        # Save training examples
        output_path = Path(output_dir) / f'training_examples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(training_examples, f, indent=2)
        
        self.logger.info(f"Saved {len(training_examples)} training examples to {output_path}")
    
    def save_history(self, stats: Dict):
        """Save comparison history"""
        history_file = self.history_dir / f'qc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(history_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved history to {history_file}")
    
    def generate_report(self, stats: Dict, patterns: Dict, 
                       output_path: str = None):
        """Generate QC report"""
        if output_path is None:
            output_path = f'qc_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Tennis Tagger QC Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Predicted File: {stats['predicted_file']}\n")
            f.write(f"Corrected File: {stats['corrected_file']}\n\n")
            
            f.write(f"Total Rows: {stats['total_rows']}\n")
            f.write(f"Total Corrections: {stats['total_corrections']}\n")
            f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2%}\n\n")
            
            f.write("Column Accuracy:\n")
            f.write("-" * 60 + "\n")
            for col, acc in sorted(stats['column_accuracy'].items(), 
                                 key=lambda x: x[1]):
                f.write(f"{col:40s} {acc:6.2%}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Error Patterns\n")
            f.write("=" * 60 + "\n\n")
            
            if patterns['low_accuracy_columns']:
                f.write("Low Accuracy Columns:\n")
                for item in patterns['low_accuracy_columns']:
                    f.write(f"  - {item['column']}: {item['accuracy']:.2%}\n")
                f.write("\n")
            
            if patterns['high_correction_columns']:
                f.write("High Correction Columns:\n")
                for item in patterns['high_correction_columns']:
                    f.write(f"  - {item['column']}: {item['corrections']} corrections\n")
                f.write("\n")
            
            if patterns['recommendations']:
                f.write("Recommendations:\n")
                for rec in patterns['recommendations']:
                    f.write(f"  - {rec}\n")
        
        self.logger.info(f"Report saved to {output_path}")


def main():
    """Command-line interface for QC feedback"""
    parser = argparse.ArgumentParser(description='Tennis Tagger QC Feedback')
    
    parser.add_argument('--predicted', type=str, required=True,
                       help='Path to predicted CSV')
    parser.add_argument('--corrected', type=str, required=True,
                       help='Path to corrected CSV')
    parser.add_argument('--update_models', action='store_true',
                       help='Generate training data for model updates')
    parser.add_argument('--report', action='store_true',
                       help='Generate QC report')
    parser.add_argument('--history_dir', type=str,
                       help='View QC history directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    qc = QCFeedback(config)
    
    # Compare CSVs
    stats = qc.compare_csvs(args.predicted, args.corrected)
    
    # Identify patterns
    patterns = qc.identify_error_patterns(stats)
    
    # Save history
    qc.save_history(stats)
    
    # Generate training data if requested
    if args.update_models:
        qc.generate_training_data(
            args.predicted,
            args.corrected,
            'data/training/corrections'
        )
    
    # Generate report if requested
    if args.report:
        qc.generate_report(stats, patterns)
    
    # Print summary
    print("\n" + "=" * 60)
    print("QC Feedback Summary")
    print("=" * 60)
    print(f"Overall Accuracy: {stats['overall_accuracy']:.2%}")
    print(f"Total Corrections: {stats['total_corrections']}")
    print(f"\nTop 5 Lowest Accuracy Columns:")
    
    sorted_cols = sorted(stats['column_accuracy'].items(), key=lambda x: x[1])
    for col, acc in sorted_cols[:5]:
        print(f"  {col}: {acc:.2%}")
    
    if patterns['recommendations']:
        print(f"\nRecommendations:")
        for rec in patterns['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
