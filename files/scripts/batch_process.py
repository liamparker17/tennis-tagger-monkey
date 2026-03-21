"""
Batch Video Processing Script
Process multiple tennis videos in batch
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import TennisTagger, setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(description='Batch process tennis videos')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output CSVs')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='Video file pattern (e.g., *.mp4, *.mov)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate annotated videos')
    
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    if args.gpu:
        config['hardware']['use_gpu'] = True
    
    # Setup logging
    logger = setup_logging(config)
    
    # Find videos
    video_files = list(input_dir.glob(args.pattern))
    
    if not video_files:
        print(f"No videos found matching pattern: {args.pattern}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Initialize tagger
    tagger = TennisTagger(config, logger)
    
    # Process each video
    results = []
    failed = []
    
    for idx, video_path in enumerate(video_files, 1):
        logger.info("=" * 70)
        logger.info(f"Processing {idx}/{len(video_files)}: {video_path.name}")
        logger.info("=" * 70)
        
        # Determine output path
        output_csv = output_dir / f"{video_path.stem}_tagged.csv"
        
        try:
            # Process video
            stats = tagger.process_video(
                str(video_path),
                str(output_csv),
                visualize=args.visualize
            )
            
            results.append({
                'video': video_path.name,
                'status': 'success',
                'stats': stats
            })
            
            logger.info(f"✓ Successfully processed: {video_path.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {video_path.name}: {e}")
            failed.append({
                'video': video_path.name,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed videos:")
        for item in failed:
            logger.info(f"  - {item['video']}: {item['error']}")
    
    # Save summary report
    summary_path = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w') as f:
        f.write("Tennis Tagger Batch Processing Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total videos: {len(video_files)}\n")
        f.write(f"Successful: {len(results)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        if results:
            f.write("Successful:\n")
            f.write("-" * 70 + "\n")
            for result in results:
                f.write(f"  {result['video']}\n")
                stats = result['stats']
                f.write(f"    Frames: {stats['frames_processed']}, "
                       f"Serves: {stats['num_serves']}, "
                       f"Strokes: {stats['num_strokes']}, "
                       f"Rallies: {stats['num_rallies']}\n")
        
        if failed:
            f.write("\nFailed:\n")
            f.write("-" * 70 + "\n")
            for item in failed:
                f.write(f"  {item['video']}: {item['error']}\n")
    
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
