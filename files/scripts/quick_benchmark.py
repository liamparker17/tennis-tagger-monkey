"""
Quick Benchmark Script

Tests model loading, processing speed, and memory usage on a short video clip.
"""

import argparse
import sys
import os
from pathlib import Path
import time
import psutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import load_config, setup_logging, TennisTagger


def format_bytes(bytes_val):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description='Quick benchmark for Tennis Tagger')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to short test video (recommended: <30 seconds)')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    print("=" * 70)
    print("TENNIS TAGGER - QUICK BENCHMARK")
    print("=" * 70)
    print()

    # Check environment
    print("Environment Check:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Offline mode: {os.environ.get('DISABLE_ONLINE_MODEL_DOWNLOADS', 'Not set')}")
    print()

    # Load config
    config = load_config(args.config)

    # Check model paths
    print("Model Loading Check:")
    models_root = config.get('models', {}).get('models_root', 'models')
    player_model = config.get('detection', {}).get('player_detector', {}).get('model', 'yolov8x.pt')
    ball_model = config.get('detection', {}).get('ball_detector', {}).get('model', 'yolov8x.pt')

    player_model_path = Path(models_root) / player_model
    ball_model_path = Path(models_root) / ball_model

    print(f"  Models root: {models_root}")
    print(f"  Player detector: {player_model_path} - {'✓ Found' if player_model_path.exists() else '✗ Missing'}")
    print(f"  Ball detector: {ball_model_path} - {'✓ Found' if ball_model_path.exists() else '✗ Missing'}")
    print()

    # Get process info for memory tracking
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    print(f"Memory before init: {format_bytes(mem_before)}")
    print()

    # Initialize logger
    logger = setup_logging(config)
    logger.info("Starting benchmark...")

    # Initialize tagger (this loads models)
    print("Initialising tagger (loading models)...")
    init_start = time.time()

    try:
        tagger = TennisTagger(config, logger)
        init_time = time.time() - init_start
        print(f"  ✓ Initialisation complete in {init_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Initialisation failed: {e}")
        sys.exit(1)

    mem_after_init = process.memory_info().rss
    print(f"  Memory after init: {format_bytes(mem_after_init)} "
          f"(+{format_bytes(mem_after_init - mem_before)})")
    print()

    # Process video
    output_csv = args.video.replace('.mp4', '_benchmark.csv')

    print(f"Processing video: {args.video}")
    print(f"  Batch size: {args.batch}")
    print()

    try:
        stats = tagger.process_video(
            args.video,
            output_csv,
            visualize=False,
            batch_size=args.batch,
            checkpoint_interval=1000,
            resume=False
        )

        # Get peak memory
        mem_peak = process.memory_info().rss

        print()
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print()
        print("Processing Statistics:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Video duration: {stats['duration_seconds']:.1f}s")
        print(f"  Processing time: {stats['processing_time_seconds']:.1f}s")
        print(f"  Processing FPS: {stats['fps_processing']:.2f} fps")
        print(f"  Real-time factor: {stats['duration_seconds'] / stats['processing_time_seconds']:.2f}x")
        print()

        print("Detection Results:")
        print(f"  Serves: {stats['num_serves']}")
        print(f"  Strokes: {stats['num_strokes']}")
        print(f"  Rallies: {stats['num_rallies']}")
        print()

        print("Memory Usage:")
        print(f"  Before init: {format_bytes(mem_before)}")
        print(f"  After init: {format_bytes(mem_after_init)}")
        print(f"  Peak: {format_bytes(mem_peak)}")
        print(f"  Total increase: {format_bytes(mem_peak - mem_before)}")
        print()

        print(f"Output: {output_csv}")
        print()

        # Test checkpoint/resume
        print("Checkpointing Test:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.json"))
            if checkpoints:
                print(f"  ✗ Checkpoint files still exist (should be cleared): {len(checkpoints)}")
            else:
                print(f"  ✓ Checkpoint cleared after successful processing")
        else:
            print(f"  ✓ No checkpoints (as expected)")

        print()
        print("=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
