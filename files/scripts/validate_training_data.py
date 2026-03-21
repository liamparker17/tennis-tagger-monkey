"""
Validation Script
Validates training data format and integrity
"""

import argparse
from pathlib import Path
import pandas as pd
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Validator')


def validate_csv_format(csv_path: Path) -> tuple:
    """Validate CSV format"""
    try:
        df = pd.read_csv(csv_path)
        
        # Required columns
        required_cols = [
            'Name', 'Position', 'Duration', '0 - Point Level',
            'A1: Server', 'F1: Point Won', 'F2: Point Score'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        if len(df) == 0:
            return False, "CSV is empty"
        
        return True, f"Valid CSV with {len(df)} rows"
        
    except Exception as e:
        return False, f"Error reading CSV: {e}"


def validate_video(video_path: Path) -> tuple:
    """Validate video file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False, "Cannot open video"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if frame_count == 0:
            return False, "Video has no frames"
        
        if fps == 0:
            return False, "Invalid FPS"
        
        return True, f"{width}x{height}, {fps:.2f}fps, {frame_count} frames"
        
    except Exception as e:
        return False, f"Error reading video: {e}"


def main():
    parser = argparse.ArgumentParser(description='Validate training data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with training data')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return
    
    logger.info(f"Validating training data in: {data_dir}")
    logger.info("=" * 60)
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.error("No CSV files found")
        return
    
    valid_pairs = 0
    total_pairs = 0
    issues = []
    
    for csv_file in csv_files:
        total_pairs += 1
        logger.info(f"\nChecking: {csv_file.name}")
        
        # Check CSV
        csv_valid, csv_msg = validate_csv_format(csv_file)
        logger.info(f"  CSV: {'✓' if csv_valid else '✗'} {csv_msg}")
        
        # Check for corresponding video
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        video_file = None
        
        for ext in video_extensions:
            potential_video = csv_file.with_suffix(ext)
            if potential_video.exists():
                video_file = potential_video
                break
        
        if video_file is None:
            logger.warning(f"  Video: ✗ No video file found")
            issues.append(f"{csv_file.name}: No corresponding video")
            continue
        
        # Check video
        video_valid, video_msg = validate_video(video_file)
        logger.info(f"  Video: {'✓' if video_valid else '✗'} {video_msg}")
        
        if csv_valid and video_valid:
            valid_pairs += 1
        else:
            if not csv_valid:
                issues.append(f"{csv_file.name}: {csv_msg}")
            if not video_valid:
                issues.append(f"{video_file.name}: {video_msg}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs: {total_pairs}")
    logger.info(f"Valid pairs: {valid_pairs}")
    logger.info(f"Issues: {len(issues)}")
    
    if issues:
        logger.info("\nIssues found:")
        for issue in issues:
            logger.info(f"  - {issue}")
    else:
        logger.info("\n✓ All training data is valid!")


if __name__ == "__main__":
    main()
