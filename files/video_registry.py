"""
Video Registry System

Manages video references by path - videos stay where they are (no duplication).
Only stores references, hashes, and paths to associated FVD/CSV files.

Key principles:
- Videos stay in place (external drives, NAS, cloud)
- Only reference path + hash stored
- Hash used to detect if video changed
- FVD generated and stored locally

Registry Format (video_registry.json):
{
    "version": "1.0",
    "videos": {
        "match_001": {
            "path": "C:/Users/liamp/Videos/Tennis/match1.mp4",
            "hash": "sha256...",
            "fvd_path": "data/fvd/match_001.fvd.json.gz",
            "csv_path": "data/output/match_001.csv",
            "status": "processed",
            "used_for_training": true,
            "added_at": "2025-01-15T...",
            "processed_at": "2025-01-15T...",
            "file_size": 1234567890,
            "duration_seconds": 3600.5,
            "fps": 30,
            "total_frames": 108015
        }
    }
}
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger('VideoRegistry')


class VideoRegistry:
    """
    Manages video references without duplication.

    Videos stay in their original locations. The registry tracks:
    - Original video path
    - Video hash (for change detection)
    - Associated FVD and CSV paths
    - Processing status
    - Training usage
    """

    REGISTRY_VERSION = "1.0"

    def __init__(self, registry_path: str = "data/video_registry.json"):
        """
        Initialize video registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from JSON file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    # Migrate old versions if needed
                    if 'version' not in data:
                        data = {'version': self.REGISTRY_VERSION, 'videos': data}
                    return data
            except json.JSONDecodeError:
                logger.warning("Corrupted registry, creating new one")
                return {'version': self.REGISTRY_VERSION, 'videos': {}}
        return {'version': self.REGISTRY_VERSION, 'videos': {}}

    def _save_registry(self):
        """Save registry to JSON file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    @staticmethod
    def compute_video_hash(video_path: str, chunk_size: int = 10 * 1024 * 1024) -> str:
        """
        Compute SHA256 hash of video file (first 10MB for speed).

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

    def _generate_video_id(self, video_path: str) -> str:
        """
        Generate unique ID for a video.

        Args:
            video_path: Path to video file

        Returns:
            Unique video ID
        """
        video_name = Path(video_path).stem

        # Clean name for use as ID
        clean_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in video_name)

        # Add suffix if ID already exists
        base_id = clean_name
        counter = 0

        while clean_name in self.registry['videos']:
            # Check if it's the same video
            existing = self.registry['videos'][clean_name]
            existing_path = Path(existing.get('path', ''))

            if existing_path.exists() and str(existing_path.absolute()) == str(Path(video_path).absolute()):
                # Same video, return existing ID
                return clean_name

            counter += 1
            clean_name = f"{base_id}_{counter}"

        return clean_name

    def add_video(self, video_path: str, auto_process: bool = False) -> Tuple[str, Dict]:
        """
        Add video to registry (no copying!).

        Args:
            video_path: Path to video file
            auto_process: Whether to trigger processing after adding

        Returns:
            (video_id, video_entry)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Check if already in registry
        existing_id = self.find_video_by_path(video_path)
        if existing_id:
            logger.info(f"Video already in registry: {existing_id}")
            return existing_id, self.registry['videos'][existing_id]

        # Generate ID and hash
        video_id = self._generate_video_id(str(video_path))
        video_hash = self.compute_video_hash(str(video_path))

        # Get file info
        stat = video_path.stat()

        # Create entry (video stays in place!)
        entry = {
            'path': str(video_path.absolute()),
            'name': video_path.name,
            'hash': video_hash,
            'fvd_path': None,
            'csv_path': None,
            'status': 'pending',
            'used_for_training': False,
            'added_at': datetime.now().isoformat(),
            'processed_at': None,
            'file_size': stat.st_size,
            'duration_seconds': None,
            'fps': None,
            'total_frames': None
        }

        self.registry['videos'][video_id] = entry
        self._save_registry()

        logger.info(f"Added video to registry: {video_id} -> {video_path}")

        return video_id, entry

    def find_video_by_path(self, video_path: str) -> Optional[str]:
        """
        Find video ID by path.

        Args:
            video_path: Path to video file

        Returns:
            Video ID or None
        """
        video_path = Path(video_path)
        abs_path = str(video_path.absolute())

        for video_id, entry in self.registry['videos'].items():
            if entry.get('path') == abs_path:
                return video_id

        return None

    def find_video_by_hash(self, video_hash: str) -> Optional[str]:
        """
        Find video ID by hash.

        Args:
            video_hash: Video hash

        Returns:
            Video ID or None
        """
        for video_id, entry in self.registry['videos'].items():
            if entry.get('hash') == video_hash:
                return video_id
        return None

    def get_video(self, video_id: str) -> Optional[Dict]:
        """
        Get video entry by ID.

        Args:
            video_id: Video ID

        Returns:
            Video entry or None
        """
        return self.registry['videos'].get(video_id)

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """
        Get video file path.

        Args:
            video_id: Video ID

        Returns:
            Path to video file or None
        """
        entry = self.get_video(video_id)
        if entry:
            return Path(entry['path'])
        return None

    def video_exists(self, video_id: str) -> bool:
        """
        Check if video file exists on disk.

        Args:
            video_id: Video ID

        Returns:
            True if video file exists
        """
        path = self.get_video_path(video_id)
        return path is not None and path.exists()

    def video_changed(self, video_id: str) -> bool:
        """
        Check if video has changed since being added.

        Args:
            video_id: Video ID

        Returns:
            True if video hash differs from stored hash
        """
        entry = self.get_video(video_id)
        if not entry:
            return True

        path = Path(entry['path'])
        if not path.exists():
            return True

        try:
            current_hash = self.compute_video_hash(str(path))
            return current_hash != entry.get('hash')
        except:
            return True

    def update_video_path(self, video_id: str, new_path: str) -> bool:
        """
        Update video path (if video was moved).

        Args:
            video_id: Video ID
            new_path: New path to video

        Returns:
            True if updated successfully
        """
        entry = self.get_video(video_id)
        if not entry:
            return False

        new_path = Path(new_path)
        if not new_path.exists():
            raise FileNotFoundError(f"Video not found: {new_path}")

        # Verify hash matches
        new_hash = self.compute_video_hash(str(new_path))
        if new_hash != entry.get('hash'):
            logger.warning(f"Hash mismatch - this may be a different video")

        entry['path'] = str(new_path.absolute())
        entry['hash'] = new_hash
        self._save_registry()

        logger.info(f"Updated video path: {video_id} -> {new_path}")
        return True

    def update_video_metadata(self, video_id: str,
                               fps: float = None,
                               duration: float = None,
                               total_frames: int = None) -> bool:
        """
        Update video metadata after processing.

        Args:
            video_id: Video ID
            fps: Frames per second
            duration: Duration in seconds
            total_frames: Total frame count

        Returns:
            True if updated
        """
        entry = self.get_video(video_id)
        if not entry:
            return False

        if fps is not None:
            entry['fps'] = fps
        if duration is not None:
            entry['duration_seconds'] = duration
        if total_frames is not None:
            entry['total_frames'] = total_frames

        self._save_registry()
        return True

    def mark_processing_started(self, video_id: str) -> bool:
        """Mark video processing as started."""
        entry = self.get_video(video_id)
        if not entry:
            return False

        entry['status'] = 'processing'
        entry['processing_started_at'] = datetime.now().isoformat()
        self._save_registry()
        return True

    def mark_processed(self, video_id: str, fvd_path: str, csv_path: str) -> bool:
        """
        Mark video as processed.

        Args:
            video_id: Video ID
            fvd_path: Path to FVD file
            csv_path: Path to output CSV

        Returns:
            True if updated
        """
        entry = self.get_video(video_id)
        if not entry:
            return False

        entry['status'] = 'processed'
        entry['fvd_path'] = str(fvd_path)
        entry['csv_path'] = str(csv_path)
        entry['processed_at'] = datetime.now().isoformat()
        self._save_registry()

        logger.info(f"Marked video as processed: {video_id}")
        return True

    def mark_failed(self, video_id: str, error: str) -> bool:
        """Mark video processing as failed."""
        entry = self.get_video(video_id)
        if not entry:
            return False

        entry['status'] = 'failed'
        entry['error'] = error
        entry['failed_at'] = datetime.now().isoformat()
        self._save_registry()
        return True

    def mark_used_for_training(self, video_id: str) -> bool:
        """Mark video as used for training."""
        entry = self.get_video(video_id)
        if not entry:
            return False

        entry['used_for_training'] = True
        entry['last_trained_at'] = datetime.now().isoformat()
        self._save_registry()
        return True

    def get_videos_by_status(self, status: str) -> List[Dict]:
        """
        Get all videos with given status.

        Args:
            status: Status filter (pending, processing, processed, failed)

        Returns:
            List of video entries with their IDs
        """
        videos = []
        for video_id, entry in self.registry['videos'].items():
            if entry.get('status') == status:
                videos.append({**entry, 'id': video_id})
        return sorted(videos, key=lambda x: x.get('added_at', ''), reverse=True)

    def get_processed_videos(self) -> List[Dict]:
        """Get all processed videos."""
        return self.get_videos_by_status('processed')

    def get_pending_videos(self) -> List[Dict]:
        """Get all pending videos."""
        return self.get_videos_by_status('pending')

    def get_training_videos(self) -> List[Dict]:
        """Get all videos used for training."""
        videos = []
        for video_id, entry in self.registry['videos'].items():
            if entry.get('used_for_training'):
                videos.append({**entry, 'id': video_id})
        return sorted(videos, key=lambda x: x.get('last_trained_at', ''), reverse=True)

    def get_missing_videos(self) -> List[Dict]:
        """
        Get videos where file no longer exists.

        Returns:
            List of missing video entries
        """
        missing = []
        for video_id, entry in self.registry['videos'].items():
            path = Path(entry.get('path', ''))
            if not path.exists():
                missing.append({**entry, 'id': video_id})
        return missing

    def remove_video(self, video_id: str, delete_fvd: bool = False) -> bool:
        """
        Remove video from registry.

        Args:
            video_id: Video ID
            delete_fvd: Also delete associated FVD file

        Returns:
            True if removed
        """
        if video_id not in self.registry['videos']:
            return False

        entry = self.registry['videos'][video_id]

        # Optionally delete FVD
        if delete_fvd and entry.get('fvd_path'):
            fvd_path = Path(entry['fvd_path'])
            if fvd_path.exists():
                fvd_path.unlink()
                logger.info(f"Deleted FVD: {fvd_path}")

        del self.registry['videos'][video_id]
        self._save_registry()

        logger.info(f"Removed video from registry: {video_id}")
        return True

    def get_stats(self) -> Dict:
        """
        Get registry statistics.

        Returns:
            Statistics dictionary
        """
        videos = self.registry['videos']

        stats = {
            'total': len(videos),
            'pending': 0,
            'processing': 0,
            'processed': 0,
            'failed': 0,
            'used_for_training': 0,
            'total_size_gb': 0,
            'missing': 0
        }

        for entry in videos.values():
            status = entry.get('status', 'pending')
            if status in stats:
                stats[status] += 1

            if entry.get('used_for_training'):
                stats['used_for_training'] += 1

            stats['total_size_gb'] += entry.get('file_size', 0) / (1024**3)

            if not Path(entry.get('path', '')).exists():
                stats['missing'] += 1

        stats['total_size_gb'] = round(stats['total_size_gb'], 2)

        return stats

    def list_all_videos(self) -> List[Dict]:
        """
        List all videos in registry.

        Returns:
            List of video entries with IDs
        """
        videos = []
        for video_id, entry in self.registry['videos'].items():
            # Add file existence check
            path = Path(entry.get('path', ''))
            exists = path.exists()

            videos.append({
                **entry,
                'id': video_id,
                'exists': exists
            })

        return sorted(videos, key=lambda x: x.get('added_at', ''), reverse=True)

    def import_from_video_database(self, video_db_path: str = "data/video_database.json") -> int:
        """
        Import videos from old video_database.json.

        Args:
            video_db_path: Path to old video database

        Returns:
            Number of videos imported
        """
        db_path = Path(video_db_path)
        if not db_path.exists():
            return 0

        with open(db_path, 'r') as f:
            old_db = json.load(f)

        imported = 0
        for fingerprint, entry in old_db.get('videos', {}).items():
            video_path = entry.get('path')
            if not video_path:
                continue

            path = Path(video_path)
            if not path.exists():
                continue

            try:
                video_id, new_entry = self.add_video(video_path)

                # Copy over status
                old_status = entry.get('status')
                if old_status == 'completed':
                    new_entry['status'] = 'processed'
                    new_entry['csv_path'] = entry.get('output_csv')
                    new_entry['processed_at'] = entry.get('completed_at')

                # Copy metadata
                new_entry['total_frames'] = entry.get('total_frames')

                self._save_registry()
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import {video_path}: {e}")

        logger.info(f"Imported {imported} videos from old database")
        return imported


# Convenience function
def create_video_registry(registry_path: str = "data/video_registry.json") -> VideoRegistry:
    """
    Create video registry instance.

    Args:
        registry_path: Path to registry file

    Returns:
        VideoRegistry instance
    """
    return VideoRegistry(registry_path)
