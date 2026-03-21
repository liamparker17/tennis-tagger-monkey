"""
Frame Vector Data (FVD) System

Extracts and stores lightweight vector data from video frames for training resumption.
FVD files are compact JSON (~2-5MB per hour of video vs 2-5GB for video).

Key benefits:
- Training can resume from FVD without reprocessing video
- FVD can be shared across machines for distributed training
- Extracted once during first video process, reused forever

FVD Format:
{
    "video_path": "C:/Videos/match.mp4",
    "video_hash": "abc123...",
    "total_frames": 5400,
    "fps": 30,
    "court_lines": [[x1,y1,x2,y2], ...],
    "frames": {
        "0": {
            "players": [
                {"bbox": [100,200,150,400], "id": 1, "pose": [[x,y,c], ...]},
                {"bbox": [500,100,550,300], "id": 2, "pose": [[x,y,c], ...]}
            ],
            "ball": {"x": 320, "y": 240, "c": 0.95}
        },
        ...
    }
}
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import gzip

logger = logging.getLogger('FrameVectorData')


class FrameVectorData:
    """
    Manages Frame Vector Data extraction, storage, and loading.

    FVD contains per-frame detection data in a compact format that allows
    training to resume without reprocessing video.
    """

    # FVD file version for compatibility
    FVD_VERSION = "1.0"

    def __init__(self, fvd_dir: str = "data/fvd"):
        """
        Initialize FVD manager.

        Args:
            fvd_dir: Directory to store FVD files
        """
        self.fvd_dir = Path(fvd_dir)
        self.fvd_dir.mkdir(parents=True, exist_ok=True)

        # In-memory FVD for incremental building
        self._current_fvd: Optional[Dict] = None
        self._current_video_path: Optional[str] = None

    @staticmethod
    def compute_video_hash(video_path: str, chunk_size: int = 10 * 1024 * 1024) -> str:
        """
        Compute hash of video file (first 10MB for speed).

        Args:
            video_path: Path to video file
            chunk_size: Size of chunk to hash (default 10MB)

        Returns:
            SHA256 hash string
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        hasher = hashlib.sha256()

        with open(video_path, 'rb') as f:
            # Hash first chunk for speed on large files
            data = f.read(chunk_size)
            hasher.update(data)

            # Also include file size in hash
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            hasher.update(str(file_size).encode())

        return hasher.hexdigest()

    def get_fvd_path(self, video_path: str) -> Path:
        """
        Get FVD file path for a video.

        Args:
            video_path: Path to video file

        Returns:
            Path to FVD file
        """
        video_name = Path(video_path).stem
        return self.fvd_dir / f"{video_name}.fvd.json"

    def get_fvd_path_compressed(self, video_path: str) -> Path:
        """
        Get compressed FVD file path for a video.

        Args:
            video_path: Path to video file

        Returns:
            Path to compressed FVD file
        """
        video_name = Path(video_path).stem
        return self.fvd_dir / f"{video_name}.fvd.json.gz"

    def fvd_exists(self, video_path: str) -> bool:
        """
        Check if FVD exists for a video.

        Args:
            video_path: Path to video file

        Returns:
            True if FVD exists and matches video hash
        """
        fvd_path = self.get_fvd_path(video_path)
        fvd_path_gz = self.get_fvd_path_compressed(video_path)

        # Check either uncompressed or compressed
        if fvd_path.exists():
            try:
                with open(fvd_path, 'r') as f:
                    header = json.load(f)

                # Verify hash matches
                current_hash = self.compute_video_hash(video_path)
                return header.get('video_hash') == current_hash
            except:
                return False

        if fvd_path_gz.exists():
            try:
                with gzip.open(fvd_path_gz, 'rt') as f:
                    header = json.load(f)

                current_hash = self.compute_video_hash(video_path)
                return header.get('video_hash') == current_hash
            except:
                return False

        return False

    def start_fvd(self, video_path: str, fps: float, total_frames: int,
                  court_lines: Optional[List] = None) -> Dict:
        """
        Start building a new FVD for a video.

        Args:
            video_path: Path to video file
            fps: Video frames per second
            total_frames: Total number of frames
            court_lines: Court line detections (once per video)

        Returns:
            FVD header dictionary
        """
        video_hash = self.compute_video_hash(video_path)

        self._current_fvd = {
            'version': self.FVD_VERSION,
            'video_path': str(Path(video_path).absolute()),
            'video_hash': video_hash,
            'total_frames': total_frames,
            'fps': fps,
            'court_lines': self._serialize_court_lines(court_lines) if court_lines else None,
            'created_at': datetime.now().isoformat(),
            'last_frame_idx': -1,
            'frames': {}
        }

        self._current_video_path = video_path

        logger.info(f"Started FVD for: {Path(video_path).name}")
        return self._current_fvd

    def add_frame(self, frame_idx: int, detections: Dict, tracks: Dict, poses: Dict) -> None:
        """
        Add frame data to current FVD.

        Args:
            frame_idx: Frame index
            detections: Detection results (players, ball, court)
            tracks: Tracking results (player_tracks, ball_tracks)
            poses: Pose estimation results
        """
        if self._current_fvd is None:
            raise RuntimeError("No FVD in progress. Call start_fvd() first.")

        frame_data = self._serialize_frame(detections, tracks, poses)

        # Store as string key for JSON compatibility
        self._current_fvd['frames'][str(frame_idx)] = frame_data
        self._current_fvd['last_frame_idx'] = frame_idx

    def _serialize_frame(self, detections: Dict, tracks: Dict, poses: Dict) -> Dict:
        """
        Serialize frame data to compact format.

        Args:
            detections: Raw detection results
            tracks: Raw tracking results
            poses: Raw pose results

        Returns:
            Compact frame dictionary
        """
        frame_data = {
            'players': [],
            'ball': None
        }

        # Process player tracks (with poses if available)
        player_tracks = tracks.get('player_tracks', [])
        frame_poses = poses.get('poses', [])

        for i, track in enumerate(player_tracks):
            player_data = {
                'bbox': self._round_bbox(track.get('bbox', [])),
                'id': track.get('id', -1),
                'conf': round(track.get('confidence', 0), 3)
            }

            # Add pose if available
            if i < len(frame_poses) and frame_poses[i]:
                pose = frame_poses[i]
                keypoints = pose.get('keypoints', [])
                if keypoints:
                    # Compress keypoints: [[x, y, conf], ...]
                    player_data['pose'] = [
                        [round(kp[0], 1), round(kp[1], 1), round(kp[2], 2)]
                        for kp in keypoints
                    ]

            frame_data['players'].append(player_data)

        # Process ball tracks
        ball_tracks = tracks.get('ball_tracks', [])
        if ball_tracks:
            best_ball = max(ball_tracks, key=lambda b: b.get('confidence', 0))
            center = best_ball.get('center')
            if center:
                frame_data['ball'] = {
                    'x': round(center[0], 1),
                    'y': round(center[1], 1),
                    'c': round(best_ball.get('confidence', 0), 3)
                }

        return frame_data

    def _round_bbox(self, bbox: List) -> List:
        """Round bounding box coordinates to integers."""
        return [int(round(x)) for x in bbox] if bbox else []

    def _serialize_court_lines(self, court_lines: Any) -> List:
        """Serialize court lines to list format."""
        if court_lines is None:
            return None

        # Handle different court line formats
        if hasattr(court_lines, 'tolist'):
            return court_lines.tolist()
        elif isinstance(court_lines, list):
            return [[int(round(x)) for x in line] for line in court_lines]
        else:
            return None

    def save_fvd(self, compress: bool = True) -> Path:
        """
        Save current FVD to disk.

        Args:
            compress: Whether to gzip compress the file

        Returns:
            Path to saved FVD file
        """
        if self._current_fvd is None or self._current_video_path is None:
            raise RuntimeError("No FVD in progress. Call start_fvd() first.")

        self._current_fvd['completed_at'] = datetime.now().isoformat()
        self._current_fvd['frame_count'] = len(self._current_fvd['frames'])

        if compress:
            fvd_path = self.get_fvd_path_compressed(self._current_video_path)
            with gzip.open(fvd_path, 'wt', encoding='utf-8') as f:
                json.dump(self._current_fvd, f, separators=(',', ':'))
        else:
            fvd_path = self.get_fvd_path(self._current_video_path)
            with open(fvd_path, 'w') as f:
                json.dump(self._current_fvd, f, separators=(',', ':'))

        # Calculate file size
        file_size = fvd_path.stat().st_size / (1024 * 1024)
        logger.info(f"FVD saved: {fvd_path.name} ({file_size:.2f} MB)")

        # Clear current FVD
        saved_fvd = self._current_fvd
        self._current_fvd = None
        self._current_video_path = None

        return fvd_path

    def save_fvd_incremental(self, checkpoint_interval: int = 1000) -> Optional[Path]:
        """
        Save FVD incrementally during processing.

        Args:
            checkpoint_interval: Save every N frames

        Returns:
            Path to saved file if saved, None otherwise
        """
        if self._current_fvd is None:
            return None

        last_frame = self._current_fvd.get('last_frame_idx', 0)

        # Only save at intervals
        if last_frame > 0 and last_frame % checkpoint_interval == 0:
            # Save as partial FVD
            partial_path = self.fvd_dir / f"{Path(self._current_video_path).stem}.fvd.partial.json.gz"

            with gzip.open(partial_path, 'wt', encoding='utf-8') as f:
                json.dump(self._current_fvd, f, separators=(',', ':'))

            logger.debug(f"FVD incremental save at frame {last_frame}")
            return partial_path

        return None

    def load_fvd(self, video_path: str) -> Optional[Dict]:
        """
        Load FVD for a video.

        Args:
            video_path: Path to video file

        Returns:
            FVD dictionary or None if not found
        """
        fvd_path = self.get_fvd_path(video_path)
        fvd_path_gz = self.get_fvd_path_compressed(video_path)

        # Try compressed first
        if fvd_path_gz.exists():
            with gzip.open(fvd_path_gz, 'rt', encoding='utf-8') as f:
                fvd = json.load(f)
            logger.info(f"Loaded FVD: {fvd_path_gz.name}")
            return fvd

        # Try uncompressed
        if fvd_path.exists():
            with open(fvd_path, 'r') as f:
                fvd = json.load(f)
            logger.info(f"Loaded FVD: {fvd_path.name}")
            return fvd

        # Try partial (from interrupted processing)
        partial_path = self.fvd_dir / f"{Path(video_path).stem}.fvd.partial.json.gz"
        if partial_path.exists():
            with gzip.open(partial_path, 'rt', encoding='utf-8') as f:
                fvd = json.load(f)
            logger.info(f"Loaded partial FVD: {partial_path.name}")
            return fvd

        return None

    def load_fvd_header(self, video_path: str) -> Optional[Dict]:
        """
        Load only FVD header (without frames) for quick metadata access.

        Args:
            video_path: Path to video file

        Returns:
            FVD header (without frames) or None
        """
        fvd = self.load_fvd(video_path)
        if fvd:
            # Return header only
            return {k: v for k, v in fvd.items() if k != 'frames'}
        return None

    def get_fvd_frame(self, fvd: Dict, frame_idx: int) -> Optional[Dict]:
        """
        Get data for a specific frame from FVD.

        Args:
            fvd: Loaded FVD dictionary
            frame_idx: Frame index

        Returns:
            Frame data or None
        """
        return fvd.get('frames', {}).get(str(frame_idx))

    def iter_fvd_frames(self, fvd: Dict):
        """
        Iterate over frames in FVD.

        Args:
            fvd: Loaded FVD dictionary

        Yields:
            (frame_idx, frame_data) tuples
        """
        frames = fvd.get('frames', {})
        for frame_idx in sorted(frames.keys(), key=int):
            yield int(frame_idx), frames[frame_idx]

    def can_resume_from_fvd(self, video_path: str) -> Tuple[bool, int]:
        """
        Check if processing can resume from FVD.

        Args:
            video_path: Path to video file

        Returns:
            (can_resume, last_frame_idx)
        """
        fvd = self.load_fvd(video_path)

        if fvd is None:
            return False, 0

        # Verify hash matches
        try:
            current_hash = self.compute_video_hash(video_path)
            if fvd.get('video_hash') != current_hash:
                logger.warning(f"Video hash mismatch - video may have changed")
                return False, 0
        except FileNotFoundError:
            return False, 0

        last_frame = fvd.get('last_frame_idx', -1)
        total_frames = fvd.get('total_frames', 0)

        # Can resume if not complete
        if last_frame >= 0 and last_frame < total_frames - 1:
            return True, last_frame

        # Already complete
        if last_frame >= total_frames - 1:
            return False, last_frame  # Complete, no resume needed

        return False, 0

    def list_fvd_files(self) -> List[Dict]:
        """
        List all FVD files.

        Returns:
            List of FVD metadata dictionaries
        """
        fvd_files = []

        for fvd_file in self.fvd_dir.glob("*.fvd.json*"):
            if '.partial' in fvd_file.name:
                continue

            try:
                if fvd_file.suffix == '.gz':
                    with gzip.open(fvd_file, 'rt', encoding='utf-8') as f:
                        # Read only first part for header
                        content = f.read(10000)
                        # Parse partial JSON to get header
                        header = json.loads(content.split('"frames"')[0].rstrip(',') + '}')
                else:
                    with open(fvd_file, 'r') as f:
                        content = f.read(10000)
                        header = json.loads(content.split('"frames"')[0].rstrip(',') + '}')

                fvd_files.append({
                    'path': str(fvd_file),
                    'video_path': header.get('video_path', 'Unknown'),
                    'video_name': Path(header.get('video_path', 'Unknown')).name,
                    'total_frames': header.get('total_frames', 0),
                    'frame_count': header.get('frame_count', 0),
                    'fps': header.get('fps', 0),
                    'created_at': header.get('created_at', ''),
                    'size_mb': fvd_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Error reading FVD {fvd_file}: {e}")

        return sorted(fvd_files, key=lambda x: x.get('created_at', ''), reverse=True)

    def delete_fvd(self, video_path: str) -> bool:
        """
        Delete FVD for a video.

        Args:
            video_path: Path to video file

        Returns:
            True if deleted
        """
        fvd_path = self.get_fvd_path(video_path)
        fvd_path_gz = self.get_fvd_path_compressed(video_path)
        partial_path = self.fvd_dir / f"{Path(video_path).stem}.fvd.partial.json.gz"

        deleted = False

        for path in [fvd_path, fvd_path_gz, partial_path]:
            if path.exists():
                path.unlink()
                deleted = True
                logger.info(f"Deleted FVD: {path.name}")

        return deleted

    def get_frames_for_timestamp_range(self, fvd: Dict,
                                         start_ms: float, end_ms: float) -> List[Tuple[int, Dict]]:
        """
        Get FVD frames for a timestamp range.

        This is used to extract frames corresponding to annotated events
        (e.g., a point that starts at timestamp X and lasts Y milliseconds).

        Args:
            fvd: Loaded FVD dictionary
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds

        Returns:
            List of (frame_idx, frame_data) tuples
        """
        fps = fvd.get('fps', 30)
        frames = fvd.get('frames', {})

        # Convert milliseconds to frame indices
        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps)

        # Ensure valid range
        start_frame = max(0, start_frame)
        total_frames = fvd.get('total_frames', 0)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

        # Collect frames in range
        result = []
        for frame_idx in range(start_frame, end_frame + 1):
            frame_key = str(frame_idx)
            if frame_key in frames:
                result.append((frame_idx, frames[frame_key]))

        return result

    def get_frames_for_point(self, fvd: Dict, position_us: int,
                              duration_us: int = 0, padding_frames: int = 30) -> List[Tuple[int, Dict]]:
        """
        Get FVD frames for an annotated point.

        Convenience method that handles microsecond timestamps from CSV annotations.

        Args:
            fvd: Loaded FVD dictionary
            position_us: Point start position in microseconds (from CSV)
            duration_us: Point duration in microseconds
            padding_frames: Extra frames before/after

        Returns:
            List of (frame_idx, frame_data) tuples
        """
        fps = fvd.get('fps', 30)
        frames = fvd.get('frames', {})

        # Convert microseconds to frame indices
        start_frame = int((position_us / 1_000_000) * fps)
        if duration_us > 0:
            end_frame = int(((position_us + duration_us) / 1_000_000) * fps)
        else:
            # Default to 5 seconds if no duration
            end_frame = start_frame + int(5 * fps)

        # Add padding
        start_frame = max(0, start_frame - padding_frames)
        total_frames = fvd.get('total_frames', 0)
        end_frame = end_frame + padding_frames
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

        # Collect frames in range
        result = []
        for frame_idx in range(start_frame, end_frame + 1):
            frame_key = str(frame_idx)
            if frame_key in frames:
                result.append((frame_idx, frames[frame_key]))

        return result

    def get_training_data_from_fvd(self, fvd: Dict) -> Dict:
        """
        Extract training-ready data from FVD.

        This provides the detection data in a format suitable for
        model training without needing to reload the video.

        Args:
            fvd: Loaded FVD dictionary

        Returns:
            Training data dictionary
        """
        training_data = {
            'video_path': fvd.get('video_path'),
            'fps': fvd.get('fps'),
            'total_frames': fvd.get('total_frames'),
            'court_lines': fvd.get('court_lines'),
            'player_tracks': [],  # List of all player tracks
            'ball_positions': [],  # List of all ball positions
            'poses': []  # List of all poses
        }

        for frame_idx, frame_data in self.iter_fvd_frames(fvd):
            # Collect player data
            for player in frame_data.get('players', []):
                training_data['player_tracks'].append({
                    'frame_idx': frame_idx,
                    'id': player.get('id'),
                    'bbox': player.get('bbox'),
                    'conf': player.get('conf'),
                    'pose': player.get('pose')
                })

            # Collect ball data
            ball = frame_data.get('ball')
            if ball:
                training_data['ball_positions'].append({
                    'frame_idx': frame_idx,
                    'x': ball.get('x'),
                    'y': ball.get('y'),
                    'conf': ball.get('c')
                })

        return training_data


# Convenience function for use in main.py
def create_fvd_manager(fvd_dir: str = "data/fvd") -> FrameVectorData:
    """
    Create FVD manager instance.

    Args:
        fvd_dir: Directory for FVD storage

    Returns:
        FrameVectorData instance
    """
    return FrameVectorData(fvd_dir)
