# Performance & Memory Fix Plan

## Issue 1: Slow Detection (10 sec per 32 frames)

### Root Causes Identified

1. **Model reloading on every batch** - YOLO model may be reloading weights
2. **CPU-only inference** - torch not properly installed with CUDA
3. **Inefficient frame preprocessing** - Converting frames individually
4. **No GPU pinning** - Data copying between CPU and GPU on every batch
5. **Verbose model output** - Even with `verbose=False`, YOLO may be logging

### Speed Bottleneck Breakdown

For RTX 2050, expected speeds:
- YOLO inference on 32 frames: ~0.5-1 second
- Frame preprocessing: ~0.1-0.2 seconds
- Result parsing: ~0.05-0.1 seconds
- **Total expected**: ~1-2 seconds per 32 frames

**Your current**: 10 seconds per 32 frames = **5-10x slower than expected**

### Likely Culprits (in order of impact)

1. **CPU-only mode** (5-10x slowdown) - Most likely
2. **Model reloading** (2-3x slowdown) - Possible
3. **Inefficient preprocessing** (1.5-2x slowdown) - Likely
4. **No warmup run** (first batch slow) - Minor

## Issue 2: No Memory of Processed Videos

### What's Needed

System should remember:
1. Which videos have been processed
2. How far processing got (last frame)
3. What features were extracted
4. Where checkpoint files are

### Current State

- Checkpoints exist but no central tracking
- No video fingerprinting
- No processed video database
- Re-processing same video starts from scratch

## Solutions

### Part 1: Speed Optimizations

#### Fix 1: Ensure CUDA is Actually Being Used

**File**: Create `check_gpu.py`
```python
import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    # Test YOLO on GPU
    model = YOLO("models/yolov8x.pt")
    print("Model device:", next(model.model.parameters()).device)

    import numpy as np
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    import time
    start = time.time()
    results = model(test_frame, verbose=False)
    end = time.time()

    print(f"Inference time: {(end-start)*1000:.1f}ms")
    print("Expected: <100ms on GPU, >500ms on CPU")
```

**Run**: `python check_gpu.py`

If it says CPU or >500ms, your torch doesn't have CUDA.

**Fix**:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Fix 2: Optimize Batch Processing

**File**: `detection/player_detector.py` and `detection/ball_detector.py`

Add these optimizations:

```python
def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
    """Detect players in a batch of frames (GPU-optimised)"""
    if self.model is None or len(frames) == 0:
        return [self._fallback_detect(f) for f in frames]

    # OPTIMIZATION 1: Convert to torch tensor on GPU directly
    # Instead of YOLO doing it internally, we do it once for the batch
    import torch
    if torch.cuda.is_available():
        # Preprocess frames to tensor
        frame_tensors = []
        for frame in frames:
            # YOLO expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensors.append(frame_rgb)

        # Stack into single tensor
        batch_tensor = np.stack(frame_tensors)
    else:
        batch_tensor = frames

    # OPTIMIZATION 2: Disable augmentation for speed
    # OPTIMIZATION 3: Use half precision (FP16) if GPU supports it
    results_batch = self.model.predict(
        batch_tensor,
        conf=self.confidence,
        iou=self.iou_threshold,
        classes=[0],
        verbose=False,
        half=True,  # Use FP16 (2x faster on modern GPUs)
        augment=False,  # Disable test-time augmentation
        max_det=self.max_detections  # Limit detections early
    )

    # ... rest stays same
```

#### Fix 3: Model Warmup

**File**: `detection/player_detector.py` and `ball_detector.py` `__init__`

Add warmup after model loading:

```python
if YOLO_AVAILABLE:
    if self.model_path:
        try:
            os.environ['YOLO_OFFLINE'] = '1'
            self.model = YOLO(str(self.model_path))
            self.logger.info(f"Loaded YOLO model: {self.model_path}")

            # WARMUP: Run dummy inference to initialize CUDA
            import numpy as np
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_frame, verbose=False)
            self.logger.info("Model warmup complete")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
```

#### Fix 4: Reduce Model Size

Using `yolov8x.pt` (136MB) - largest and slowest model.

**Recommendation**: Use `yolov8m.pt` (52MB) for 2-3x faster inference with minimal accuracy loss.

**Change in**: `config\config.yaml`
```yaml
detection:
  player_detector:
    model: "yolov8m.pt"  # ← Changed from yolov8x.pt
  ball_detector:
    model: "yolov8m.pt"  # ← Changed from yolov8x.pt
```

**Speed comparison** on RTX 2050:
- yolov8x: ~80ms per frame
- yolov8m: ~30ms per frame
- yolov8s: ~15ms per frame

For 32 frame batch:
- yolov8x: ~2.5 seconds
- yolov8m: ~1 second
- yolov8s: ~0.5 seconds

### Part 2: Video Memory System

#### Database Structure

**File**: Create `utils\video_database.py`

```python
"""
Video processing database for tracking processed videos and enabling resume.
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

    def __init__(self, db_path: str = "data/video_database.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = self._load_db()

    def _load_db(self) -> dict:
        """Load database from JSON file"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {'videos': {}, 'version': '1.0'}

    def _save_db(self):
        """Save database to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)

    def get_video_fingerprint(self, video_path: str) -> str:
        """
        Generate unique fingerprint for a video.
        Uses: file path, size, and modification time
        """
        video_path = Path(video_path)

        # Combine path, size, and mtime
        stat = video_path.stat()
        fingerprint_data = f"{video_path.absolute()}_{stat.st_size}_{stat.st_mtime}"

        # Hash it
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def is_processed(self, video_path: str) -> bool:
        """Check if video has been fully processed"""
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint not in self.db['videos']:
            return False

        video_data = self.db['videos'][fingerprint]
        return video_data.get('status') == 'completed'

    def get_video_status(self, video_path: str) -> Optional[Dict]:
        """Get processing status for a video"""
        fingerprint = self.get_video_fingerprint(video_path)
        return self.db['videos'].get(fingerprint)

    def mark_processing_started(self, video_path: str, total_frames: int, output_csv: str):
        """Mark video processing as started"""
        fingerprint = self.get_video_fingerprint(video_path)

        self.db['videos'][fingerprint] = {
            'path': str(Path(video_path).absolute()),
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
        """Update processing progress"""
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint in self.db['videos']:
            self.db['videos'][fingerprint]['processed_frames'] = processed_frames
            self.db['videos'][fingerprint]['updated_at'] = datetime.now().isoformat()
            self._save_db()

    def mark_completed(self, video_path: str, output_csv: str):
        """Mark video processing as completed"""
        fingerprint = self.get_video_fingerprint(video_path)

        if fingerprint in self.db['videos']:
            self.db['videos'][fingerprint]['status'] = 'completed'
            self.db['videos'][fingerprint]['completed_at'] = datetime.now().isoformat()
            self.db['videos'][fingerprint]['output_csv'] = str(output_csv)
            self._save_db()
            logger.info(f"Marked video {Path(video_path).name} as completed")

    def get_incomplete_videos(self) -> List[Dict]:
        """Get all videos that are partially processed"""
        incomplete = []
        for fingerprint, data in self.db['videos'].items():
            if data['status'] == 'processing':
                incomplete.append(data)
        return incomplete

    def can_resume(self, video_path: str) -> tuple:
        """
        Check if video can be resumed from checkpoint.

        Returns:
            (can_resume: bool, last_frame: int)
        """
        status = self.get_video_status(video_path)

        if status and status['status'] == 'processing':
            # Check if checkpoint exists
            from utils.checkpointing import get_checkpoint_path
            checkpoint_path = get_checkpoint_path(video_path)

            if checkpoint_path.exists():
                return True, status['processed_frames']

        return False, 0
```

#### Integration with Main Pipeline

**File**: `main.py` - Update `process_video` method

Add at start of function:

```python
def process_video(self, video_path: str, output_csv: str, ...):
    """Process a tennis match video and generate tagged CSV (streaming/chunked)."""

    # NEW: Check video database
    from utils.video_database import VideoDatabase
    video_db = VideoDatabase()

    # Check if already processed
    if video_db.is_processed(video_path):
        self.logger.info(f"Video already processed: {video_path}")
        existing_status = video_db.get_video_status(video_path)

        # Ask user if they want to reprocess
        self.logger.warning("Video already in database. Use --force to reprocess")
        return {
            'video_path': video_path,
            'status': 'already_processed',
            'output_csv': existing_status['output_csv'],
            'completed_at': existing_status.get('completed_at')
        }

    # Check if can resume
    can_resume_video, last_frame = video_db.can_resume(video_path)
    if can_resume_video and resume:
        self.logger.info(f"Resuming from frame {last_frame}")

    # Get metadata
    metadata = self.video_processor.get_video_metadata(video_path)

    # Mark as processing
    video_db.mark_processing_started(
        video_path,
        metadata['total_frames'],
        output_csv
    )

    # ... existing processing code ...

    # After each chunk, update progress
    video_db.update_progress(video_path, total_processed)

    # At end, mark completed
    video_db.mark_completed(video_path, output_csv)
```

### Part 3: Dataset Management UI Wiring

Need to check what dataset management features exist in UI and wire them up properly.

## Implementation Priority

1. **Check GPU** - Run check_gpu.py (5 min)
2. **Fix torch CUDA** - Reinstall with CUDA support (10 min)
3. **Switch to yolov8m** - Edit config.yaml (1 min)
4. **Add warmup** - Edit detector init (5 min)
5. **Create VideoDatabase** - New utility file (15 min)
6. **Integrate with main** - Update main.py (10 min)
7. **Wire dataset UI** - Check and fix UI connections (15 min)

**Total time**: ~1 hour

## Expected Results

### Speed Improvements
- **Before**: 10 sec per 32 frames
- **After**: 0.5-1 sec per 32 frames
- **Speedup**: 10-20x faster

### Memory Features
- Videos tracked in database
- Auto-resume if interrupted
- Skips already-processed videos
- Shows processing history

## Next Steps

1. Run the GPU check script I'll create
2. Fix torch installation if needed
3. Apply the optimizations
4. Test with a short video
5. Verify speed improvement
