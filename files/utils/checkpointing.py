"""
Checkpointing utilities for resumable video processing.

Allows saving and loading processing state to resume after failures.
Now also supports FVD (Frame Vector Data) incremental saves for training resumption.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime


logger = logging.getLogger('Checkpointing')


# =============================================================================
# FVD CHECKPOINT INTEGRATION
# =============================================================================

def get_fvd_checkpoint_info(video_path: str, fvd_dir: str = "data/fvd") -> Tuple[bool, int, str]:
    """
    Check if FVD checkpoint exists and get last processed frame.

    Args:
        video_path: Path to video file
        fvd_dir: Directory containing FVD files

    Returns:
        (has_fvd, last_frame_idx, fvd_path)
    """
    video_name = Path(video_path).stem
    fvd_dir = Path(fvd_dir)

    # Check for partial FVD (in-progress)
    partial_path = fvd_dir / f"{video_name}.fvd.partial.json.gz"
    if partial_path.exists():
        try:
            import gzip
            with gzip.open(partial_path, 'rt') as f:
                data = json.load(f)
                last_frame = data.get('last_frame_idx', -1)
                return True, last_frame, str(partial_path)
        except:
            pass

    # Check for complete FVD
    for ext in ['.fvd.json.gz', '.fvd.json']:
        fvd_path = fvd_dir / f"{video_name}{ext}"
        if fvd_path.exists():
            try:
                if ext.endswith('.gz'):
                    import gzip
                    with gzip.open(fvd_path, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(fvd_path, 'r') as f:
                        data = json.load(f)

                last_frame = data.get('last_frame_idx', data.get('total_frames', 0) - 1)
                return True, last_frame, str(fvd_path)
            except:
                pass

    return False, 0, ""


def can_resume_from_any(video_path: str,
                        checkpoint_dir: str = "checkpoints",
                        fvd_dir: str = "data/fvd") -> Tuple[str, int]:
    """
    Check if processing can resume from either checkpoint or FVD.

    Args:
        video_path: Path to video file
        checkpoint_dir: Directory containing checkpoints
        fvd_dir: Directory containing FVD files

    Returns:
        (source, last_frame_idx) where source is 'checkpoint', 'fvd', or 'none'
    """
    # Check standard checkpoint first
    checkpoint = load_checkpoint(video_path, checkpoint_dir)
    if checkpoint:
        checkpoint_frame = checkpoint.get('last_frame_idx', 0)
    else:
        checkpoint_frame = 0

    # Check FVD
    has_fvd, fvd_frame, fvd_path = get_fvd_checkpoint_info(video_path, fvd_dir)

    # Return the one with most progress
    if checkpoint_frame > 0 and checkpoint_frame >= fvd_frame:
        return 'checkpoint', checkpoint_frame
    elif has_fvd and fvd_frame > 0:
        return 'fvd', fvd_frame
    else:
        return 'none', 0


def get_checkpoint_path(video_path: str, checkpoint_dir: str = "checkpoints") -> Path:
    """
    Get checkpoint file path for a video.

    Args:
        video_path: Path to video file
        checkpoint_dir: Directory to store checkpoints

    Returns:
        Path to checkpoint JSON file
    """
    video_name = Path(video_path).stem
    checkpoint_path = Path(checkpoint_dir) / f"{video_name}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def save_checkpoint(
    video_path: str,
    last_frame_idx: int,
    detections: list,
    tracks: list,
    poses: list,
    output_csv: str,
    checkpoint_dir: str = "checkpoints"
) -> Path:
    """
    Save processing checkpoint with BACKUP PROTECTION.

    If an existing checkpoint has MORE data than what we're about to save,
    the existing checkpoint is backed up to prevent accidental data loss.

    Args:
        video_path: Path to video being processed
        last_frame_idx: Last successfully processed frame index
        detections: Accumulated detections list
        tracks: Accumulated tracks list
        poses: Accumulated poses list
        output_csv: Output CSV path
        checkpoint_dir: Directory to store checkpoints

    Returns:
        Path to saved checkpoint file
    """
    import shutil
    checkpoint_path = get_checkpoint_path(video_path, checkpoint_dir)
    data_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_data"

    # PROTECTION: Check if existing checkpoint has more data
    if checkpoint_path.exists() and data_dir.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                existing_meta = json.load(f)

            existing_frame = existing_meta.get('last_frame_idx', 0)
            existing_detections = existing_meta.get('num_detections', 0)

            # If existing checkpoint has significantly more data, BACK IT UP
            if existing_frame > last_frame_idx + 1000 or existing_detections > len(detections) * 2:
                backup_dir = checkpoint_path.parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{checkpoint_path.stem}_{timestamp}"

                # Backup metadata
                backup_meta = backup_dir / f"{backup_name}.json"
                shutil.copy2(checkpoint_path, backup_meta)

                # Backup data directory
                backup_data = backup_dir / f"{backup_name}_data"
                if data_dir.exists():
                    shutil.copytree(data_dir, backup_data)

                logger.warning(
                    f"BACKUP CREATED: Existing checkpoint had {existing_frame:,} frames, "
                    f"new has {last_frame_idx:,}. Backup saved to: {backup_dir}"
                )
        except Exception as e:
            logger.warning(f"Could not check existing checkpoint: {e}")

    checkpoint_data = {
        'version': '1.0',
        'video_path': str(video_path),
        'last_frame_idx': last_frame_idx,
        'output_csv': str(output_csv),
        'timestamp': datetime.now().isoformat(),
        'num_detections': len(detections),
        'num_tracks': len(tracks),
        'num_poses': len(poses)
    }

    # Save checkpoint metadata
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Save accumulated data (separate files for size)
    data_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(data_dir / "detections.pkl", 'wb') as f:
        pickle.dump(detections, f)
    with open(data_dir / "tracks.pkl", 'wb') as f:
        pickle.dump(tracks, f)
    with open(data_dir / "poses.pkl", 'wb') as f:
        pickle.dump(poses, f)

    logger.info(f"Checkpoint saved: {checkpoint_path} (frame {last_frame_idx})")
    return checkpoint_path


def load_checkpoint(
    video_path: str,
    checkpoint_dir: str = "checkpoints"
) -> Optional[Dict[str, Any]]:
    """
    Load processing checkpoint if exists.

    Args:
        video_path: Path to video file
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Checkpoint data dictionary, or None if not found
    """
    checkpoint_path = get_checkpoint_path(video_path, checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    # Load metadata
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)

    # Load accumulated data
    data_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_data"

    import pickle
    try:
        with open(data_dir / "detections.pkl", 'rb') as f:
            checkpoint_data['detections'] = pickle.load(f)
        with open(data_dir / "tracks.pkl", 'rb') as f:
            checkpoint_data['tracks'] = pickle.load(f)
        with open(data_dir / "poses.pkl", 'rb') as f:
            checkpoint_data['poses'] = pickle.load(f)
    except FileNotFoundError as e:
        logger.warning(f"Checkpoint data incomplete: {e}")
        return None

    logger.info(f"Checkpoint loaded: {checkpoint_path} (frame {checkpoint_data['last_frame_idx']})")
    return checkpoint_data


def clear_checkpoint(video_path: str, checkpoint_dir: str = "checkpoints",
                     clear_fvd_partial: bool = True, fvd_dir: str = "data/fvd"):
    """
    Clear checkpoint files for a video.

    Args:
        video_path: Path to video file
        checkpoint_dir: Directory containing checkpoints
        clear_fvd_partial: Also clear partial FVD files
        fvd_dir: Directory containing FVD files
    """
    checkpoint_path = get_checkpoint_path(video_path, checkpoint_dir)

    # Remove metadata file
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Remove data directory
    data_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_data"
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)

    # Clear partial FVD if requested (but not completed FVD)
    if clear_fvd_partial:
        video_name = Path(video_path).stem
        partial_path = Path(fvd_dir) / f"{video_name}.fvd.partial.json.gz"
        if partial_path.exists():
            partial_path.unlink()
            logger.info(f"Partial FVD cleared for: {video_path}")

    logger.info(f"Checkpoint cleared for: {video_path}")
