"""
Threaded Video Processing for Gradio UI

Provides non-blocking video processing with real-time progress updates
for the Gradio training interface.
"""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import yaml


class VideoProcessingThread:
    """Thread-safe video processing with progress tracking"""

    def __init__(self):
        self.thread = None
        self.progress_queue = queue.Queue()
        self.is_running = False
        self.current_progress = {
            'stage': 'idle',
            'current': 0,
            'total': 100,
            'message': 'Ready to process',
            'percent': 0
        }
        self.result = None
        self.error = None
        self.batch_size = 32  # Default batch size (larger = better GPU utilization)
        self.device = 'auto'  # Default device setting
        self.frame_skip = 1  # Process every Nth frame (1=all, 3=3x faster)
        self._start_time = None  # Track when processing started

    def progress_callback(self, stage: str, current: int, total: int, message: str):
        """Called by video processor to report progress"""
        percent = int((current / total) * 100) if total > 0 else 0
        progress_data = {
            'stage': stage,
            'current': current,
            'total': total,
            'message': message,
            'percent': percent
        }
        self.current_progress = progress_data
        self.progress_queue.put(progress_data)

    def _process_worker(self, video_path: str, output_csv: str, config: dict, visualize: bool, resume: bool = False):
        """Worker function that runs in separate thread"""
        try:
            # Report initialization start
            self.progress_callback('starting', 0, 100, 'Importing modules...')

            # Import here to avoid loading models in main thread
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / "src"))

            # Try importing with better error reporting
            try:
                from main import TennisTagger, setup_logging
            except ImportError as ie:
                raise ImportError(
                    f"Failed to import processing modules: {str(ie)}\n"
                    f"This usually means dependencies are not installed.\n"
                    f"Please run: pip install -r requirements.txt"
                )

            # Report model loading
            self.progress_callback('starting', 10, 100, 'Loading detection models...')

            # Create tagger with progress callback
            logger = setup_logging(config)
            tagger = TennisTagger(config, logger, progress_callback=self.progress_callback)

            # Report processing start
            skip_msg = f" (processing every {self.frame_skip} frames)" if self.frame_skip > 1 else ""
            if resume:
                self.progress_callback('starting', 50, 100, f'Resuming from checkpoint...{skip_msg}')
            else:
                self.progress_callback('starting', 50, 100, f'Starting video processing...{skip_msg}')

            # Process video with UI-controlled parameters
            stats = tagger.process_video(
                video_path,
                output_csv,
                visualize=visualize,
                batch_size=self.batch_size,  # From UI control
                checkpoint_interval=1000,  # Save every 1000 frames
                resume=resume,  # Resume from checkpoint if requested
                extract_fvd=True,  # IMPORTANT: Extract FVD for training!
                frame_skip=self.frame_skip  # Speed optimization
            )

            self.result = stats
            self.error = None

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.error = error_msg
            self.result = None
            self.progress_callback('error', 0, 1, f'Error: {str(e)}')
        finally:
            self.is_running = False

    def start_processing(self, video_path: str, output_csv: str, config_path: str = 'config/config.yaml', visualize: bool = False, batch_size: int = 8, device: str = 'auto', resume: bool = False, frame_skip: int = 1):
        """Start video processing in background thread with configurable batch size and device"""
        if self.is_running:
            return False, "Processing already in progress"

        # Load config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return False, f"Error loading config: {str(e)}"

        # Apply UI-controlled hardware settings to config
        if device != 'auto':
            config['hardware'] = config.get('hardware', {})
            config['hardware']['use_gpu'] = (device == 'cuda')

        # Store batch_size, device, and frame_skip for worker
        self.batch_size = batch_size
        self.device = device
        self.frame_skip = frame_skip

        # Reset state
        self.result = None
        self.error = None
        self.is_running = True
        self._start_time = time.time()  # Track when processing started
        self.current_progress = {
            'stage': 'starting',
            'current': 0,
            'total': 100,
            'message': 'Resuming from checkpoint...' if resume else 'Initializing...',
            'percent': 0
        }

        # Start thread
        self.thread = threading.Thread(
            target=self._process_worker,
            args=(video_path, output_csv, config, visualize, resume),
            daemon=True
        )
        self.thread.start()

        return True, "Resuming from checkpoint..." if resume else "Processing started"

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress (non-blocking)"""
        # Drain queue to get latest progress
        while not self.progress_queue.empty():
            try:
                self.current_progress = self.progress_queue.get_nowait()
            except queue.Empty:
                break

        return self.current_progress

    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return not self.is_running and self.thread is not None

    def get_result(self) -> tuple:
        """Get processing result (result_dict, error_string)"""
        return self.result, self.error


# Global instance for Gradio interface
_global_processor = None

def get_video_processor() -> VideoProcessingThread:
    """Get or create global video processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = VideoProcessingThread()
    return _global_processor


def start_video_processing(video_path: str, output_csv: str, visualize: bool = False, batch_size: int = 8, device: str = 'auto', resume: bool = False, frame_skip: int = 1) -> tuple:
    """
    Start video processing in background with configurable settings

    Args:
        video_path: Path to video file
        output_csv: Output CSV path
        visualize: Whether to generate annotated video
        batch_size: Batch size for GPU processing (8-128, default 8)
        device: Device to use ('auto', 'cuda', or 'cpu', default 'auto')
        resume: Whether to resume from checkpoint if available
        frame_skip: Process every Nth frame (1=all, 3=3x faster)

    Returns:
        (success: bool, message: str)
    """
    processor = get_video_processor()
    return processor.start_processing(video_path, output_csv, visualize=visualize, batch_size=batch_size, device=device, resume=resume, frame_skip=frame_skip)


def get_processing_progress() -> str:
    """
    Get current processing progress as formatted string

    Returns:
        Formatted progress message with:
        - Frame counts with commas for readability
        - Percentage prominently displayed
        - Elapsed time in human-readable format (Xm Ys)
        - Stage name visible
    """
    processor = get_video_processor()
    progress = processor.get_progress()

    stage = progress['stage']
    current = progress['current']
    total = progress['total']
    message = progress['message']
    percent = progress['percent']

    # Format stage name
    stage_names = {
        'idle': 'Idle',
        'starting': 'Starting',
        'loading': 'Loading Video',
        'detection': 'Object Detection',
        'analysis': 'Event Analysis',
        'generating': 'Generating CSV',
        'visualization': 'Creating Video',
        'complete': 'Complete',
        'error': 'Error'
    }
    stage_display = stage_names.get(stage, stage.title())

    # Build progress bar
    bar_length = 30
    filled = int((percent / 100) * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)

    # Format frame counts with commas
    current_formatted = f"{current:,}"
    total_formatted = f"{total:,}"

    # Calculate and format elapsed time
    elapsed_str = ""
    if hasattr(processor, '_start_time') and processor._start_time is not None:
        elapsed_seconds = time.time() - processor._start_time
        elapsed_mins = int(elapsed_seconds // 60)
        elapsed_secs = int(elapsed_seconds % 60)
        if elapsed_mins > 0:
            elapsed_str = f"Elapsed: {elapsed_mins}m {elapsed_secs}s"
        else:
            elapsed_str = f"Elapsed: {elapsed_secs}s"

    # Format output
    if stage == 'complete':
        return f"✅ **{stage_display}**\n\n{bar} 100%\n\n{message}"
    elif stage == 'error':
        return f"❌ **{stage_display}**\n\n{message}"
    elif stage == 'idle':
        return f"⏸️ **{stage_display}**\n\n{message}"
    else:
        # Enhanced format: Processing: X / Y frames (Z%)
        frame_info = f"Processing: {current_formatted} / {total_formatted} frames ({percent:.1f}%)"
        stage_info = f"Stage: {stage_display}"

        output_lines = [
            f"🔄 **{stage_display}**",
            "",
            f"{bar} {percent}%",
            "",
            frame_info,
            stage_info,
        ]

        if elapsed_str:
            output_lines.append(elapsed_str)

        if message and message != stage_display:
            output_lines.append("")
            output_lines.append(message)

        return "\n".join(output_lines)


def is_processing_complete() -> bool:
    """Check if video processing is complete"""
    processor = get_video_processor()
    return processor.is_complete()


def get_processing_result() -> str:
    """
    Get final processing result as formatted string

    Returns:
        Formatted result message
    """
    processor = get_video_processor()
    result, error = processor.get_result()

    if error:
        return f"❌ **Processing Failed**\n\n{error}"

    if result:
        return f"""✅ **Processing Complete!**

**Video:** {result['video_path']}
**Frames Processed:** {result['frames_processed']}
**Duration:** {result['duration_seconds']:.1f} seconds

**Results:**
- 🎾 Serves: {result['num_serves']}
- 🏸 Strokes: {result['num_strokes']}
- 🔄 Rallies: {result['num_rallies']}

**Output:** {result['output_csv']}
"""

    return "No result available"
