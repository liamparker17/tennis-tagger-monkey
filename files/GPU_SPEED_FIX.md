# GPU Speed Fix - Critical

## Problem Identified

**Symptoms:**
- 40 minutes for 5640 frames (extremely slow)
- GPU utilization at 1-3% (GPU not being used)
- Taking 6-8 minutes per 1000 frames
- At this rate: **819,750 frames would take 82+ hours!**

**Root Cause:**
YOLO models were not explicitly told to use GPU. Even though PyTorch detected CUDA, the models defaulted to CPU inference.

---

## Fixes Applied

### 1. Explicit GPU Device Setting

**File: `detection/player_detector.py`**

Added explicit GPU device detection and model placement:

```python
# In __init__ (lines 36-49):
import torch
if torch.cuda.is_available():
    self.device = '0'  # GPU device
    self.logger.info(f"Using GPU for detection: {torch.cuda.get_device_name(0)}")
else:
    self.device = 'cpu'
    self.logger.warning("GPU not available, using CPU (will be slow)")

self.model = YOLO(self.model_path)
# Force model to GPU
if self.device == '0':
    self.model.to('cuda')
```

Added `device=self.device` to all predict() calls:
- Line 90: `detect()` method
- Line 135: `detect_batch()` method

**File: `detection/ball_detector.py`**

Same changes:
- Lines 36-48: Device detection and model placement
- Line 78: Added device to `detect()`
- Line 119: Added device to `detect_batch()`

### 2. Increased Default Batch Size

**File: `main.py`**

Changed default batch size from 8 to 32:
```python
parser.add_argument('--batch', type=int, default=32,
                   help='Batch size for parallel processing (default: 32)')
```

**Why:**
- Batch size of 8 is too small for GPU
- GPUs are optimized for larger batches
- 32 frames provides better throughput
- RTX 2050 can handle 32 easily

---

## Expected Performance

### Before Fix:
- **40 minutes** for 5,640 frames
- **~424ms per frame**
- **Total time for 819,750 frames: ~82 hours**

### After Fix (Expected):
- **1-2 seconds** for 32 frames
- **~40-60ms per frame** (7-10x faster)
- **Total time for 819,750 frames: ~9-13 hours**

With further optimization (smaller model), could get down to:
- **<1 second** for 32 frames
- **~15-20ms per frame**
- **Total time: ~4-5 hours**

---

## How to Test

### Option 1: Quick Test Script

```bash
cd C:\Users\liamp\Downloads\files
python test_gpu_detection.py
```

This will:
1. Check CUDA availability
2. Load YOLO model and verify device
3. Test single frame speed
4. Test batch (32 frames) speed
5. Report GPU memory usage

**Expected output:**
```
Average: 50-80ms per frame
Batch time: 1500-2500ms for 32 frames
✓ GPU is properly configured and being used
```

### Option 2: Resume Your Current Video

Your video is at frame 5640 with a checkpoint. Resume processing:

```bash
cd C:\Users\liamp\Downloads\files
python main.py --video "C:\Users\liamp\Videos\Andrew Johnson vs Alberto Pulido Moreno.mp4" --output output.csv --resume --batch 32
```

Monitor GPU usage with Task Manager:
- GPU utilization should jump to 60-90%
- GPU memory should be ~2-4 GB
- Speed should be much faster

---

## Additional Optimizations (If Still Slow)

### Option A: Switch to Smaller Model

YOLOv8x is the largest, slowest model. Try YOLOv8m for 2-3x speed:

**Edit: `config/config.yaml`**
```yaml
detection:
  player_detector:
    model: "yolov8m.pt"  # Changed from yolov8x.pt
  ball_detector:
    model: "yolov8m.pt"
```

Download the model:
```bash
# The model will auto-download on first use
# Or manually download:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### Option B: Increase Batch Size Further

If GPU memory allows, try batch=64 or batch=128:

```bash
python main.py --video video.mp4 --output output.csv --batch 64
```

Monitor GPU memory in Task Manager. If it hits 4GB limit, reduce batch size.

### Option C: Enable FP16 (Half Precision)

**Edit: `detection/player_detector.py` and `ball_detector.py`**

Change line in detect_batch():
```python
half=True,  # Changed from False - uses FP16 for 2x speed
```

**Warning:** Only works if GPU supports FP16 (RTX 2050 does support it).

---

## Verification Checklist

After restarting processing, verify:

- [ ] Log shows: "Using GPU for detection: NVIDIA GeForce RTX 2050"
- [ ] GPU utilization in Task Manager: 60-90%
- [ ] GPU memory usage: 2-4 GB
- [ ] Processing speed: <2 seconds per 32 frames
- [ ] Estimated total time: <15 hours (acceptable)

---

## Troubleshooting

### If GPU Still Not Used:

1. **Check PyTorch CUDA installation:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print: `True`

   If False, reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Check YOLO device:**
   ```bash
   python test_gpu_detection.py
   ```
   Should show device as `cuda:0`, not `cpu`

3. **Update NVIDIA drivers:**
   - Download latest drivers from NVIDIA website
   - Restart computer after install

### If Still Slow with GPU:

1. **Try smaller model (yolov8m.pt)** - 2-3x faster
2. **Increase batch size to 64** - Better GPU utilization
3. **Enable FP16 (half=True)** - 2x faster on compatible GPUs
4. **Check GPU temperature** - Throttles if overheating
5. **Close other GPU applications** - Free up resources

---

## Summary

**Changes Made:**
1. Added explicit GPU device setting to both detectors
2. Added `device` parameter to all predict() calls
3. Forced models to CUDA with `.to('cuda')`
4. Increased default batch size from 8 to 32

**Expected Result:**
- **7-10x faster processing**
- From 82 hours → **9-13 hours** for full video
- GPU utilization: 60-90% instead of 1-3%

**Next Steps:**
1. Run `python test_gpu_detection.py` to verify GPU works
2. Resume your video with `--resume --batch 32`
3. Monitor GPU usage in Task Manager
4. If still slow, try yolov8m.pt model (smaller/faster)

The fix is critical - without GPU, this system is not practical for production use.
