#!/usr/bin/env python3
"""
Tennis Match Auto-Tagging System
Main entry point for processing tennis videos
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_processor import VideoProcessor

# Detection modules
from detection.player_detector import PlayerDetector
from detection.ball_detector import BallDetector
from detection.court_detector import CourtDetector
from detection.serve_detector import ServeDetector
from detection.event_detector import EventDetector
from detection.pose_estimator import PoseEstimator
from detection.tracker import MultiObjectTracker as ObjectTracker
from detection.unified_detector import UnifiedDetector, get_unified_detector

# Analysis / tagging modules
from analysis.csv_generator import DartfishCSVGenerator as CSVGenerator
from analysis.comparator import CSVComparator as TagComparator  # or Comparator, match class name in comparator.py
from analysis.comparator import FeedbackLoop
from analysis.rally_analyzer import RallyAnalyzer
from analysis.placement_analyzer import PlacementAnalyzer
from analysis.stroke_classifier import StrokeClassifier

# ScoreTracker requires easyocr which is optional (heavy dependency)
try:
    from analysis.score_tracker import ScoreTracker
    HAS_SCORE_TRACKER = True
except ImportError:
    ScoreTracker = None
    HAS_SCORE_TRACKER = False

# Utilities
from utils.checkpointing import save_checkpoint, load_checkpoint, clear_checkpoint
from utils.stability import SceneStabilityChecker
from utils.video_database import VideoDatabase

# FVD and Registry (new unified system)
from frame_vector_data import FrameVectorData, create_fvd_manager
from video_registry import VideoRegistry, create_video_registry



def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    # Ensure log directory exists
    log_file = log_config.get('log_file', 'tennis_tagger.log')
    log_path = Path(log_file)
    if log_path.parent != Path('.'):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path))
        ] if log_config.get('save_logs', True) else [logging.StreamHandler()]
    )

    return logging.getLogger('TennisTagger')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TennisTagger:
    """Main tennis tagging pipeline"""

    def __init__(self, config: dict, logger: logging.Logger, progress_callback=None):
        self.config = config
        self.logger = logger
        self.progress_callback = progress_callback

        # Log hardware configuration
        hardware_config = config.get('hardware', {})
        use_gpu = hardware_config.get('use_gpu', True)
        self.logger.info(f"Hardware config: GPU={'enabled' if use_gpu else 'disabled'}")

        # Check actual GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.warning("No GPU detected, will use CPU (slower)")
        except:
            self.logger.warning("PyTorch not available, GPU check skipped")

        # Initialize components
        self.logger.info("Initializing detection models...")
        self.video_processor = VideoProcessor(config, progress_callback=progress_callback)

        # PERFORMANCE: Use unified detector (single YOLO for players+ball = 2x faster)
        self.unified_detector = get_unified_detector(config)
        self.player_detector = PlayerDetector(config)  # Keep for fallback/API compat
        self.ball_detector = BallDetector(config)      # Keep for fallback/API compat
        self.court_detector = CourtDetector(config)
        self.pose_estimator = PoseEstimator(config)

        self.logger.info("Initializing analysis modules...")
        self.serve_detector = ServeDetector(config)
        self.stroke_classifier = StrokeClassifier(config)
        self.rally_analyzer = RallyAnalyzer(config)
        self.score_tracker = ScoreTracker(config) if HAS_SCORE_TRACKER else None
        self.placement_analyzer = PlacementAnalyzer(config)

        self.csv_generator = CSVGenerator(config)

        self.logger.info("Initialization complete")
    
    def process_video(self, video_path: str, output_csv: str,
                     visualize: bool = False, batch_size: int = 32,
                     checkpoint_interval: int = 1000, resume: bool = False,
                     force_reprocess: bool = False, enable_pose: bool = False,
                     use_native_tracking: bool = True,
                     extract_fvd: bool = True,
                     frame_skip: int = 1) -> dict:
        """
        Process a tennis match video and generate tagged CSV (streaming/chunked).

        Args:
            video_path: Path to input video file
            output_csv: Path to output CSV file
            visualize: Whether to generate annotated video
            batch_size: Number of frames to process in parallel batches
            checkpoint_interval: Save checkpoint every N frames
            resume: Resume from checkpoint if available
            force_reprocess: Force reprocessing even if video already completed
            enable_pose: Enable pose estimation (SLOW - disable for speed)
            use_native_tracking: Use Ultralytics native tracking (Phase 1 optimization)
            extract_fvd: Extract Frame Vector Data for training resumption
            frame_skip: Process every Nth frame (1=all, 2=half, 3=third, etc.)
                        Setting to 3 gives ~10fps on 30fps video - usually enough for tennis

        Returns:
            Dictionary with processing statistics and results
        """
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Settings: batch_size={batch_size}, frame_skip={frame_skip}, "
                        f"enable_pose={enable_pose}, extract_fvd={extract_fvd}")
        start_time = time.time()

        # Initialize video database
        video_db = VideoDatabase()

        # Initialize FVD manager and video registry for unified system
        fvd_manager = None
        video_registry = None
        video_id = None

        if extract_fvd:
            try:
                fvd_manager = create_fvd_manager()
                video_registry = create_video_registry()

                # Register video (no copy - just reference)
                video_id, registry_entry = video_registry.add_video(video_path)
                self.logger.info(f"Video registered: {video_id}")
            except Exception as e:
                self.logger.warning(f"FVD/Registry init failed: {e} - continuing without FVD")

        # Check if already processed
        if not force_reprocess and video_db.is_processed(video_path):
            existing_status = video_db.get_video_status(video_path)
            self.logger.info(f"Video already processed: {video_path}")
            self.logger.info(f"  Completed at: {existing_status.get('completed_at')}")
            self.logger.info(f"  Output CSV: {existing_status.get('output_csv')}")
            self.logger.warning("Use --force flag to reprocess")

            return {
                'video_path': video_path,
                'status': 'already_processed',
                'output_csv': existing_status['output_csv'],
                'completed_at': existing_status.get('completed_at'),
                'frames_processed': existing_status.get('processed_frames', 0),
                'total_frames': existing_status.get('total_frames', 0)
            }

        # Check if can resume from previous incomplete processing
        can_resume_video, last_processed_frame = video_db.can_resume(video_path)
        if can_resume_video and resume:
            self.logger.info(f"Video database indicates resume possible from frame {last_processed_frame}")

        # Check for existing checkpoint if resume requested
        checkpoint = None
        existing_checkpoint = load_checkpoint(video_path)
        if existing_checkpoint:
            self.logger.info(f"Found existing checkpoint at frame {existing_checkpoint['last_frame_idx']}")
            if resume:
                checkpoint = existing_checkpoint
                self.logger.info(f"RESUMING from checkpoint at frame {checkpoint['last_frame_idx']}")
            else:
                self.logger.warning(f"NOT resuming - starting fresh (existing checkpoint will be overwritten)")
                # Clear the old checkpoint to avoid confusion
                clear_checkpoint(video_path)
        else:
            self.logger.info("No existing checkpoint found - starting fresh")

        # Get video metadata (no frame loading)
        metadata = self.video_processor.get_video_metadata(video_path)
        self.logger.info(f"Video: {metadata['total_frames']} frames, "
                        f"{metadata['duration']:.1f}s at {metadata['fps']:.1f} FPS")

        # Mark video processing as started in database
        video_db.mark_processing_started(
            video_path,
            metadata['total_frames'],
            output_csv
        )

        # Update video registry with metadata
        if video_registry and video_id:
            video_registry.update_video_metadata(
                video_id,
                fps=metadata.get('target_fps', metadata.get('fps')),
                duration=metadata.get('duration'),
                total_frames=metadata.get('total_frames')
            )
            video_registry.mark_processing_started(video_id)

        # Check if we can resume from FVD
        fvd_resume_frame = 0
        if fvd_manager:
            can_resume_fvd, fvd_resume_frame = fvd_manager.can_resume_from_fvd(video_path)
            if can_resume_fvd:
                self.logger.info(f"Found existing FVD at frame {fvd_resume_frame}")
                if resume:
                    self.logger.info(f"RESUMING from FVD at frame {fvd_resume_frame}")
                else:
                    self.logger.warning(f"NOT resuming - will overwrite existing FVD")
                    fvd_resume_frame = 0

        try:
            # Initialize result containers
            if checkpoint:
                all_detections = checkpoint['detections']
                all_tracks = checkpoint['tracks']
                all_poses = checkpoint['poses']
                start_frame_idx = checkpoint['last_frame_idx'] + 1
            else:
                all_detections = []
                all_tracks = []
                all_poses = []
                start_frame_idx = 0

            court_lines = None
            stability_checker = SceneStabilityChecker(threshold=0.85)
            court_detection_count = 0

            # Start FVD extraction if enabled
            if fvd_manager and extract_fvd:
                fvd_manager.start_fvd(
                    video_path,
                    fps=metadata.get('target_fps', metadata.get('fps')),
                    total_frames=metadata['total_frames']
                )

            # Process frames in batches using streaming
            self.logger.info("Starting detection (streaming mode, batch processing)...")
            total_processed = 0
            frame_batch = []
            frame_indices = []

            # Use ThreadPoolExecutor for parallel player/ball detection
            executor = ThreadPoolExecutor(max_workers=2)

            # Detect court ONCE at the very start (not every frame - huge speedup!)
            first_chunk = True

            self.logger.info(f"Starting frame iteration with checkpoint_interval={checkpoint_interval}, "
                           f"start_frame_idx={start_frame_idx}, frame_skip={frame_skip}")

            for chunk_idx, frame_chunk in enumerate(self.video_processor.iter_frame_chunks(
                    video_path, chunk_size=checkpoint_interval)):

                chunk_start_idx = frame_chunk[0][0]
                chunk_end_idx = frame_chunk[-1][0]

                self.logger.info(f"Received chunk {chunk_idx}: frames {chunk_start_idx}-{chunk_end_idx} "
                               f"(len={len(frame_chunk)})")

                # Skip already processed chunks when resuming
                if chunk_start_idx < start_frame_idx:
                    self.logger.info(f"SKIPPING chunk {chunk_idx} (start {chunk_start_idx} < resume point {start_frame_idx})")
                    continue

                # Log chunk info (helps debug frame indexing issues)
                frames_in_chunk = len(frame_chunk)
                frames_to_process = frames_in_chunk // frame_skip if frame_skip > 1 else frames_in_chunk
                self.logger.info(f"Chunk {chunk_idx + 1}: frames {chunk_start_idx}-{chunk_end_idx} "
                               f"({frames_in_chunk} frames, ~{frames_to_process} to process with skip={frame_skip})")

                frames_skipped_this_chunk = 0
                frames_processed_this_chunk = 0

                for frame_idx, frame in frame_chunk:
                    # Detect court ONLY ONCE at the start
                    if first_chunk and court_lines is None:
                        court_lines = self.court_detector.detect(frame)
                        court_detection_count += 1
                        self.logger.info("Court detected (will use for entire video)")
                        first_chunk = False

                    # FRAME SKIP: Only process every Nth frame for speed
                    if frame_skip > 1 and frame_idx % frame_skip != 0:
                        frames_skipped_this_chunk += 1
                        continue

                    frames_processed_this_chunk += 1

                    # Accumulate frames for batch processing
                    frame_batch.append(frame)
                    frame_indices.append(frame_idx)

                    # Process batch when full
                    if len(frame_batch) >= batch_size:
                        self._process_batch(
                            frame_batch, frame_indices, court_lines,
                            all_detections, all_tracks, all_poses, executor,
                            skip_pose=not enable_pose, use_native_tracking=use_native_tracking,
                            fvd_manager=fvd_manager if extract_fvd else None
                        )
                        total_processed += len(frame_batch)
                        frame_batch = []
                        frame_indices = []

                        # Progress reporting (adjusted for frame_skip)
                        if self.progress_callback:
                            # Calculate expected total frames after frame_skip
                            expected_total = metadata['total_frames'] // frame_skip if frame_skip > 1 else metadata['total_frames']
                            percent = int((total_processed / expected_total) * 100) if expected_total > 0 else 0
                            self.progress_callback('detection', total_processed, expected_total,
                                                 f'Detected {total_processed} / {expected_total} frames ({percent}%)')

                        # Update database progress
                        video_db.update_progress(video_path, total_processed)

                # Process remaining frames in chunk
                if frame_batch:
                    self._process_batch(
                        frame_batch, frame_indices, court_lines,
                        all_detections, all_tracks, all_poses, executor,
                        skip_pose=not enable_pose, use_native_tracking=use_native_tracking,
                        fvd_manager=fvd_manager if extract_fvd else None
                    )
                    total_processed += len(frame_batch)
                    frame_batch = []
                    frame_indices = []

                # Log chunk completion
                self.logger.debug(f"Chunk complete: processed {frames_processed_this_chunk}, "
                                f"skipped {frames_skipped_this_chunk}")

                # Save checkpoint after each chunk
                last_frame_in_chunk = frame_chunk[-1][0]
                save_checkpoint(
                    video_path, last_frame_in_chunk,
                    all_detections, all_tracks, all_poses,
                    output_csv
                )
                self.logger.info(f"Checkpoint saved at frame {last_frame_in_chunk} "
                               f"(total processed so far: {total_processed})")

                # Save FVD incrementally
                if fvd_manager and extract_fvd:
                    fvd_manager.save_fvd_incremental(checkpoint_interval=checkpoint_interval)

            executor.shutdown(wait=True)

            self.logger.info(f"Detection complete. {total_processed} frames processed, "
                            f"court detected {court_detection_count} times")

            # Continue with analysis (unchanged from original)
            self.logger.info("Analysing events...")
            if self.progress_callback:
                self.progress_callback('analysis', 0, 5, 'Analysing events...')

            serves = self.serve_detector.detect(all_poses, all_tracks)
            if self.progress_callback:
                self.progress_callback('analysis', 1, 5, f'Found {len(serves)} serves...')

            strokes = self.stroke_classifier.classify(all_poses, all_tracks, None)
            if self.progress_callback:
                self.progress_callback('analysis', 2, 5, f'Found {len(strokes)} strokes...')

            rallies = self.rally_analyzer.segment(strokes, all_tracks)
            if self.progress_callback:
                self.progress_callback('analysis', 3, 5, f'Found {len(rallies)} rallies...')

            # Score tracking requires easyocr (optional)
            if self.score_tracker is not None:
                scores = self.score_tracker.track(rallies, serves)
            else:
                scores = []  # Skip score tracking if easyocr not installed
            if self.progress_callback:
                self.progress_callback('analysis', 4, 5, 'Tracking scores...')

            placements = self.placement_analyzer.analyze(
                all_tracks, court_lines, strokes
            )
            if self.progress_callback:
                self.progress_callback('analysis', 5, 5, 'Analysing placements...')

            self.logger.info(f"Found {len(serves)} serves, {len(strokes)} strokes, "
                            f"{len(rallies)} rallies")

            # Generate CSV
            self.logger.info("Generating CSV output...")
            if self.progress_callback:
                self.progress_callback('generating', 0, 1, 'Generating CSV...')

            csv_data = self.csv_generator.generate(
                metadata={'duration': metadata['duration'], 'fps': metadata['target_fps']},
                serves=serves,
                strokes=strokes,
                rallies=rallies,
                scores=scores,
                placements=placements
            )

            # Save CSV
            csv_data.to_csv(output_csv, index=False)
            self.logger.info(f"CSV saved to: {output_csv}")
            if self.progress_callback:
                self.progress_callback('generating', 1, 1, f'✓ CSV saved to {output_csv}')

            # Mark video as completed in database
            video_db.mark_completed(video_path, output_csv)

            # Save FVD and update registry
            fvd_path = None
            if fvd_manager and extract_fvd:
                try:
                    # Update court lines in FVD
                    if court_lines is not None and fvd_manager._current_fvd:
                        fvd_manager._current_fvd['court_lines'] = fvd_manager._serialize_court_lines(court_lines)

                    fvd_path = fvd_manager.save_fvd(compress=True)
                    self.logger.info(f"FVD saved: {fvd_path}")

                    # Update registry with FVD and CSV paths
                    if video_registry and video_id:
                        video_registry.mark_processed(video_id, str(fvd_path), output_csv)
                except Exception as e:
                    self.logger.warning(f"FVD save failed: {e}")

            # Clear checkpoint after successful completion
            clear_checkpoint(video_path)
            self.logger.info("Checkpoint cleared (processing complete)")

            # Skip visualization for now (would need streaming approach too)
            if visualize:
                self.logger.warning("Visualization skipped (not compatible with streaming mode)")

            if self.progress_callback:
                self.progress_callback('complete', 1, 1, '✓ Processing complete!')

            elapsed_time = time.time() - start_time

            # Return statistics
            stats = {
                'video_path': video_path,
                'video_id': video_id,
                'frames_processed': total_processed,
                'duration_seconds': metadata['duration'],
                'processing_time_seconds': elapsed_time,
                'fps_processing': total_processed / elapsed_time if elapsed_time > 0 else 0,
                'num_serves': len(serves),
                'num_strokes': len(strokes),
                'num_rallies': len(rallies),
                'output_csv': output_csv,
                'fvd_path': str(fvd_path) if fvd_path else None,
                'status': 'completed'
            }

            return stats

        except Exception as e:
            # Mark video as failed in database
            error_message = f"{type(e).__name__}: {str(e)}"
            video_db.mark_failed(video_path, error_message)
            self.logger.error(f"Video processing failed: {error_message}")

            # Re-raise exception to be handled by caller
            raise

    def _process_batch(self, frame_batch, frame_indices, court_lines,
                      all_detections, all_tracks, all_poses, executor, skip_pose=False,
                      use_native_tracking=True, fvd_manager=None):
        """
        Process a batch of frames with unified detection.

        Args:
            frame_batch: List of frames
            frame_indices: Corresponding frame indices
            court_lines: Current court line detection
            all_detections: List to append detections
            all_tracks: List to append tracks
            all_poses: List to append poses
            executor: ThreadPoolExecutor (kept for API compat, not used for main detection)
            skip_pose: Skip pose estimation entirely (default False - we use GPU now!)
            use_native_tracking: Use Ultralytics native tracking (faster than custom tracker)
            fvd_manager: FVD manager for extracting frame vector data
        """
        # PERFORMANCE: Single YOLO inference for both players AND ball (2x faster!)
        # Previously ran two separate inference passes - one for players, one for ball
        player_detections_batch, ball_detections_batch = self.unified_detector.detect_all_batch(
            frame_batch, use_tracking=use_native_tracking
        )

        # Batch pose estimation (GPU-accelerated if using YOLOv8-pose)
        if skip_pose:
            poses_batch = [[] for _ in frame_batch]
        else:
            poses_batch = self.pose_estimator.estimate_batch(frame_batch, player_detections_batch)

        # Process each frame in batch sequentially for tracking
        for i, (frame_idx, frame) in enumerate(zip(frame_indices, frame_batch)):
            player_detections = player_detections_batch[i]
            ball_detections = ball_detections_batch[i]

            # Use native track IDs if available, otherwise fallback to custom tracker
            if use_native_tracking and player_detections and 'track_id' in player_detections[0]:
                # Native tracking already done - convert to track format
                player_tracks = [{'id': det.get('track_id', -1), 'bbox': det['bbox'], 'confidence': det['confidence']}
                                for det in player_detections]
            else:
                # Fallback to custom IOU tracker
                player_tracks = self.player_detector.track(player_detections)

            if use_native_tracking and ball_detections and 'track_id' in ball_detections[0]:
                ball_tracks = [{'id': det.get('track_id', -1), 'bbox': det['bbox'], 'confidence': det['confidence'],
                               'center': det.get('center')} for det in ball_detections]
            else:
                ball_tracks = self.ball_detector.track(ball_detections)

            all_detections.append({
                'frame_idx': frame_idx,
                'players': player_detections,
                'ball': ball_detections,
                'court': court_lines
            })
            all_tracks.append({
                'frame_idx': frame_idx,
                'player_tracks': player_tracks,
                'ball_tracks': ball_tracks
            })
            all_poses.append({
                'frame_idx': frame_idx,
                'poses': poses_batch[i]
            })

            # Add frame to FVD if manager is active
            if fvd_manager and fvd_manager._current_fvd is not None:
                try:
                    fvd_manager.add_frame(
                        frame_idx,
                        all_detections[-1],
                        all_tracks[-1],
                        all_poses[-1]
                    )
                except Exception as e:
                    pass  # Don't fail processing if FVD fails


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Tennis Match Auto-Tagging System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python main.py --video match.mp4 --output results.csv
  
  # With GPU and visualization
  python main.py --video match.mp4 --output results.csv --gpu --visualize
  
  # Custom config
  python main.py --video match.mp4 --output results.csv --config my_config.yaml
        '''
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate annotated video')
    parser.add_argument('--confidence', type=float,
                       help='Override detection confidence threshold')
    parser.add_argument('--model_path', type=str,
                       help='Path to custom trained model')
    parser.add_argument('--batch', type=int, default=128,
                       help='Batch size for parallel processing (default: 128, Phase 1 optimization for better GPU utilization)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Save checkpoint every N frames (default: 1000)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if video already completed')
    parser.add_argument('--enable-pose', action='store_true',
                       help='Enable pose estimation (SLOW - only use if needed for stroke analysis)')
    parser.add_argument('--native-tracking', action='store_true', default=True,
                       help='Use Ultralytics native tracking (ByteTrack/BoTSORT) for better performance (default: True)')
    parser.add_argument('--no-native-tracking', action='store_false', dest='native_tracking',
                       help='Disable native tracking and use custom IOU tracker')
    parser.add_argument('--extract-fvd', action='store_true', default=True,
                       help='Extract Frame Vector Data for training resumption (default: True)')
    parser.add_argument('--no-fvd', action='store_false', dest='extract_fvd',
                       help='Disable FVD extraction')

    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override config with command line args
    if args.gpu:
        config['hardware']['use_gpu'] = True
    if args.confidence:
        config['detection']['player_detector']['confidence'] = args.confidence
        config['detection']['ball_detector']['confidence'] = args.confidence
    if args.model_path:
        config['events']['stroke_classification']['model'] = args.model_path
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("Tennis Match Auto-Tagging System")
    logger.info("=" * 60)
    
    try:
        # Initialize tagger
        tagger = TennisTagger(config, logger)

        # Process video with new parameters
        stats = tagger.process_video(
            args.video,
            args.output,
            visualize=args.visualize,
            batch_size=args.batch,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            force_reprocess=args.force,
            enable_pose=args.enable_pose,
            use_native_tracking=args.native_tracking,
            extract_fvd=args.extract_fvd
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"Processed: {stats['frames_processed']} frames "
                   f"({stats['duration_seconds']:.1f} seconds)")
        logger.info(f"Processing time: {stats['processing_time_seconds']:.1f}s "
                   f"({stats['fps_processing']:.2f} FPS)")
        logger.info(f"Detected: {stats['num_serves']} serves, "
                   f"{stats['num_strokes']} strokes, "
                   f"{stats['num_rallies']} rallies")
        logger.info(f"Output saved to: {stats['output_csv']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
