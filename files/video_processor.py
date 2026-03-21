"""
Video Processor Module

Handles video loading, preprocessing, and frame extraction with
optimization for efficient processing of tennis match footage.

PERFORMANCE OPTIMIZATIONS:
- Async frame reading (prefetch frames while GPU processes)
- Double-buffered frame queue
- Reduced CPU→GPU data transfer bottleneck
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Generator, Dict
import warnings
import threading
import queue


class AsyncFrameReader:
    """
    Asynchronous frame reader that prefetches frames while GPU processes.

    This prevents disk I/O from blocking GPU inference, significantly
    improving throughput on systems with slow storage.
    """

    def __init__(self, video_path: str, buffer_size: int = 256, target_fps: float = 30, target_resolution=None):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.target_resolution = target_resolution

        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.total_frames = 0
        self.original_fps = 0
        self.frame_skip = 1

    def start(self):
        """Start the async reader thread"""
        self.stop_event.clear()
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()

    def stop(self):
        """Stop the async reader"""
        self.stop_event.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)

    def _reader_worker(self):
        """Background thread that reads frames"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.frame_queue.put(None)  # Signal error
            return

        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))

        frame_idx = 0
        logical_idx = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample at target FPS
            if frame_idx % self.frame_skip == 0:
                # Resize if needed
                if self.target_resolution:
                    frame = cv2.resize(frame, tuple(self.target_resolution), interpolation=cv2.INTER_LINEAR)

                # Wait for queue space - DON'T drop frames!
                while not self.stop_event.is_set():
                    try:
                        self.frame_queue.put((logical_idx, frame), timeout=0.5)
                        logical_idx += 1
                        break  # Successfully added, continue to next frame
                    except queue.Full:
                        # Queue full, wait and retry (don't drop the frame!)
                        continue

                if self.stop_event.is_set():
                    break

            frame_idx += 1

        cap.release()
        self.frame_queue.put(None)  # Signal end of stream

    def __iter__(self):
        """Iterate over frames"""
        while True:
            try:
                item = self.frame_queue.get(timeout=5.0)
                if item is None:
                    break
                yield item
            except queue.Empty:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class VideoProcessor:
    """
    Video processing and frame extraction for tennis analysis.

    Features:
    - Efficient video loading with frame skipping
    - Resolution normalization
    - FPS standardization
    - Memory-efficient batch processing
    - ASYNC frame reading for better GPU utilization
    """

    def __init__(self, config: dict, progress_callback=None):
        """
        Initialize video processor.

        Args:
            config: Video processing configuration
            progress_callback: Optional callback function for progress updates
                              Signature: callback(stage, current, total, message)
        """
        self.target_fps = config.get('fps', 30)
        self.target_resolution = config.get('resolution', None)
        self.skip_frames = config.get('skip_frames', 0)
        self.progress_callback = progress_callback

    def get_video_metadata(self, video_path: str) -> Dict:
        """
        Get video metadata without loading frames.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with fps, total_frames, duration
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        cap.release()

        return {
            'fps': original_fps,
            'total_frames': total_frames,
            'duration': duration,
            'target_fps': self.target_fps
        }

    def iter_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterator that yields frames one by one (memory efficient).

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to yield

        Yields:
            Tuple of (frame_index, frame_array)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling rate
        frame_skip = max(1, int(original_fps / self.target_fps))

        frame_idx = 0
        frames_yielded = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at target FPS
            if frame_idx % frame_skip == 0:
                # Skip additional frames if configured
                if self.skip_frames == 0 or (frames_yielded % (self.skip_frames + 1) == 0):
                    # Resize if needed
                    if self.target_resolution:
                        frame = cv2.resize(
                            frame,
                            tuple(self.target_resolution),
                            interpolation=cv2.INTER_LINEAR
                        )

                    yield (frames_yielded, frame)
                    frames_yielded += 1

                    if max_frames and frames_yielded >= max_frames:
                        break

            frame_idx += 1

        cap.release()

    def iter_frame_chunks(
        self,
        video_path: str,
        chunk_size: int = 1000,
        max_frames: Optional[int] = None,
        use_async: bool = True
    ) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        """
        Iterator that yields chunks of frames (memory efficient batching).

        PERFORMANCE: Uses async frame reading to prefetch while GPU processes.

        Args:
            video_path: Path to video file
            chunk_size: Number of frames per chunk
            max_frames: Maximum total frames to process
            use_async: Use async frame reading (default True for better perf)

        Yields:
            List of (frame_index, frame_array) tuples
        """
        if use_async:
            # PERFORMANCE: Async reading prefetches frames while GPU works
            with AsyncFrameReader(
                video_path,
                buffer_size=256,  # Keep 256 frames buffered (larger = less chance of blocking)
                target_fps=self.target_fps,
                target_resolution=self.target_resolution
            ) as reader:
                chunk = []
                frames_yielded = 0

                for frame_idx, frame in reader:
                    # Apply additional skip if configured
                    if self.skip_frames > 0 and frames_yielded % (self.skip_frames + 1) != 0:
                        frames_yielded += 1
                        continue

                    chunk.append((frame_idx, frame))
                    frames_yielded += 1

                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []

                    if max_frames and frames_yielded >= max_frames:
                        break

                if chunk:
                    yield chunk
        else:
            # Original sync approach
            chunk = []
            for frame_idx, frame in self.iter_frames(video_path, max_frames):
                chunk.append((frame_idx, frame))

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

            if chunk:
                yield chunk

    def load_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Load video file and extract frames.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None = all)

        Returns:
            Tuple of (frames, fps, duration_seconds)
        """
        if self.progress_callback:
            self.progress_callback('loading', 0, 100, 'Opening video file...')

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # Calculate frame sampling rate
        frame_skip = max(1, int(original_fps / self.target_fps))
        expected_frames = total_frames // frame_skip

        if self.progress_callback:
            self.progress_callback('loading', 0, expected_frames, f'Loading video... (0/{expected_frames} frames)')

        frames = []
        frame_idx = 0
        last_progress_report = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Sample frames at target FPS
            if frame_idx % frame_skip == 0:
                # Skip additional frames if configured
                if self.skip_frames == 0 or (len(frames) % (self.skip_frames + 1) == 0):
                    # Resize if needed
                    if self.target_resolution:
                        frame = cv2.resize(
                            frame,
                            tuple(self.target_resolution),
                            interpolation=cv2.INTER_LINEAR
                        )

                    frames.append(frame)

                    # Report progress every 10 frames or at key milestones
                    if self.progress_callback and (len(frames) - last_progress_report >= 10 or len(frames) % 50 == 0):
                        percent = int((len(frames) / expected_frames) * 100) if expected_frames > 0 else 0
                        self.progress_callback('loading', len(frames), expected_frames,
                                             f'Loading video... ({len(frames)}/{expected_frames} frames, {percent}%)')
                        last_progress_report = len(frames)

                    # Check max frames limit
                    if max_frames and len(frames) >= max_frames:
                        break

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        if self.progress_callback:
            self.progress_callback('loading', len(frames), len(frames), f'✓ Loaded {len(frames)} frames')

        return frames, self.target_fps, duration
    
    def extract_frame_at_time(
        self,
        video_path: str,
        timestamp: float
    ) -> np.ndarray:
        """
        Extract a single frame at specific timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
        
        Returns:
            Frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not extract frame at {timestamp}s")
        
        if self.target_resolution:
            frame = cv2.resize(
                frame,
                tuple(self.target_resolution),
                interpolation=cv2.INTER_LINEAR
            )
        
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to frame for model input.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Preprocessed frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def save_annotated_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30,
        codec: str = 'mp4v'
    ):
        """
        Save frames as video file.
        
        Args:
            frames: List of frames to save
            output_path: Output video path
            fps: Output FPS
            codec: Video codec (mp4v, h264, etc.)
        """
        if len(frames) == 0:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    @staticmethod
    def draw_annotation(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw text annotation on frame.
        
        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position
            color: BGR color tuple
            thickness: Text thickness
        
        Returns:
            Frame with annotation
        """
        annotated = frame.copy()
        cv2.putText(
            annotated,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness
        )
        return annotated
    
    @staticmethod
    def draw_bounding_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: (x1, y1, x2, y2) bounding box
            label: Optional label text
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Frame with bounding box
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated
    
    @staticmethod
    def extract_roi(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 0
    ) -> np.ndarray:
        """
        Extract region of interest from frame.
        
        Args:
            frame: Input frame
            bbox: (x1, y1, x2, y2) bounding box
            padding: Additional padding around bbox
        
        Returns:
            Cropped region
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return frame[y1:y2, x1:x2]
