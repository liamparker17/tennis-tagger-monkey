"""
Video Processing Database

Tracks which videos have been processed and enables:
1. Skip already-processed videos
2. Resume from checkpoints
3. Incremental training with memory
4. Processing history

NOTE: This is the legacy video database. For new code, prefer using
VideoRegistry from video_registry.py which supports:
- No video duplication (references only)
- FVD integration
- Better video tracking

This class is kept for backward compatibility and will sync with VideoRegistry.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger('VideoDatabase')


class VideoDatabase:
    """Track processed videos and enable resume functionality"""

    def __init__(self, db_path: str = "data/video_database.json",
                 sync_with_registry: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = self._load_db()
        self.sync_with_registry = sync_with_registry
        self._registry = None

    def _get_registry(self):
        """Get video registry for syncing (lazy load)."""
        if self._registry is None and self.sync_with_registry:
            try:
                # Import here to avoid circular imports
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from video_registry import VideoRegistry
                self._registry = VideoRegistry()
            except ImportError:
                logger.debug("VideoRegistry not available, sync disabled")
                self.sync_with_registry = False
        return self._registry

    def _load_db(self) -> dict:
        """Load database from JSON file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted database, creating new one")
                return {'videos': {}, 'version': '1.0'}
        return {'videos': {}, 'version': '1.0'}

    def _save_db(self):
        """Save database to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)

    def get_video_fingerprint(self, video_path: str) -> str:
        """
        Generate unique fingerprint for a video.
        Uses: file path, size, and modification time

        Args:
            video_path: Path to video file

        Returns:
            16-character hexadecimal fingerprint
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Combine path, size, and mtime for uniqueness
        stat = video_path.stat()
        fingerprint_data = f"{video_path.absolute()}_{stat.st_size}_{stat.st_mtime}"

        # Hash it
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def is_processed(self, video_path: str) -> bool:
        """
        Check if video has been fully processed

        Args:
            video_path: Path to video file

        Returns:
            True if video is marked as completed
        """
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint not in self.db['videos']:
            return False

        video_data = self.db['videos'][fingerprint]
        return video_data.get('status') == 'completed'

    def get_video_status(self, video_path: str) -> Optional[Dict]:
        """
        Get processing status for a video

        Args:
            video_path: Path to video file

        Returns:
            Video status dictionary or None if not in database
        """
        fingerprint = self.get_video_fingerprint(video_path)
        return self.db['videos'].get(fingerprint)

    def mark_processing_started(self, video_path: str, total_frames: int, output_csv: str):
        """
        Mark video processing as started

        Args:
            video_path: Path to video file
            total_frames: Total number of frames in video
            output_csv: Path where CSV will be saved
        """
        fingerprint = self.get_video_fingerprint(video_path)

        self.db['videos'][fingerprint] = {
            'path': str(Path(video_path).absolute()),
            'name': Path(video_path).name,
            'fingerprint': fingerprint,
            'total_frames': total_frames,
            'processed_frames': 0,
            'output_csv': str(output_csv),
            'status': 'processing',
            'started_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self._save_db()
        logger.info(f"Marked video {Path(video_path).name} as processing")

    def update_progress(self, video_path: str, processed_frames: int):
        """
        Update processing progress

        Args:
            video_path: Path to video file
            processed_frames: Number of frames processed so far
        """
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint in self.db['videos']:
            self.db['videos'][fingerprint]['processed_frames'] = processed_frames
            self.db['videos'][fingerprint]['updated_at'] = datetime.now().isoformat()

            # Calculate percentage
            total = self.db['videos'][fingerprint]['total_frames']
            percentage = (processed_frames / total * 100) if total > 0 else 0
            self.db['videos'][fingerprint]['percentage'] = round(percentage, 1)

            self._save_db()

    def mark_completed(self, video_path: str, output_csv: str, fvd_path: str = None):
        """
        Mark video processing as completed

        Args:
            video_path: Path to video file
            output_csv: Path to output CSV file
            fvd_path: Path to FVD file (optional)
        """
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint in self.db['videos']:
            self.db['videos'][fingerprint]['status'] = 'completed'
            self.db['videos'][fingerprint]['completed_at'] = datetime.now().isoformat()
            self.db['videos'][fingerprint]['output_csv'] = str(output_csv)
            self.db['videos'][fingerprint]['processed_frames'] = self.db['videos'][fingerprint]['total_frames']
            self.db['videos'][fingerprint]['percentage'] = 100.0
            if fvd_path:
                self.db['videos'][fingerprint]['fvd_path'] = str(fvd_path)
            self._save_db()
            logger.info(f"Marked video {Path(video_path).name} as completed")

        # Sync with registry if available
        registry = self._get_registry()
        if registry:
            try:
                video_id = registry.find_video_by_path(video_path)
                if video_id:
                    registry.mark_processed(video_id, fvd_path or "", output_csv)
            except Exception as e:
                logger.debug(f"Registry sync failed: {e}")

    def mark_failed(self, video_path: str, error_message: str):
        """
        Mark video processing as failed

        Args:
            video_path: Path to video file
            error_message: Error message describing failure
        """
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint in self.db['videos']:
            self.db['videos'][fingerprint]['status'] = 'failed'
            self.db['videos'][fingerprint]['error'] = error_message
            self.db['videos'][fingerprint]['failed_at'] = datetime.now().isoformat()
            self._save_db()
            logger.error(f"Marked video {Path(video_path).name} as failed: {error_message}")

    def get_incomplete_videos(self) -> List[Dict]:
        """
        Get all videos that are partially processed

        Returns:
            List of video status dictionaries for incomplete videos
        """
        incomplete = []
        for fingerprint, data in self.db['videos'].items():
            if data['status'] == 'processing':
                incomplete.append(data)
        return sorted(incomplete, key=lambda x: x['updated_at'], reverse=True)

    def get_completed_videos(self) -> List[Dict]:
        """
        Get all videos that have been fully processed

        Returns:
            List of video status dictionaries for completed videos
        """
        completed = []
        for fingerprint, data in self.db['videos'].items():
            if data['status'] == 'completed':
                completed.append(data)
        return sorted(completed, key=lambda x: x['completed_at'], reverse=True)

    def can_resume(self, video_path: str) -> tuple:
        """
        Check if video can be resumed from checkpoint

        Args:
            video_path: Path to video file

        Returns:
            (can_resume: bool, last_frame: int)
        """
        try:
            status = self.get_video_status(video_path)

            if status and status['status'] == 'processing':
                # Check if checkpoint exists
                from utils.checkpointing import get_checkpoint_path
                checkpoint_path = get_checkpoint_path(video_path)

                if checkpoint_path.exists():
                    return True, status['processed_frames']

        except Exception as e:
            logger.error(f"Error checking resume capability: {e}")

        return False, 0

    def get_processing_stats(self) -> Dict:
        """
        Get overall processing statistics

        Returns:
            Dictionary with total, completed, processing, failed counts
        """
        stats = {
            'total': len(self.db['videos']),
            'completed': 0,
            'processing': 0,
            'failed': 0
        }

        for data in self.db['videos'].values():
            status = data.get('status', 'unknown')
            if status in stats:
                stats[status] += 1

        return stats

    def clear_old_entries(self, days: int = 30):
        """
        Clear completed or failed entries older than specified days

        Args:
            days: Number of days after which to clear entries
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []

        for fingerprint, data in self.db['videos'].items():
            status = data.get('status')
            if status in ['completed', 'failed']:
                updated = datetime.fromisoformat(data['updated_at'])
                if updated < cutoff:
                    to_remove.append(fingerprint)

        for fingerprint in to_remove:
            del self.db['videos'][fingerprint]

        if to_remove:
            self._save_db()
            logger.info(f"Cleared {len(to_remove)} old entries from database")

        return len(to_remove)
