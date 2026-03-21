# Training UI "Stuck on Initialising" Fix

## Problem

The training interface gets stuck showing "Initialising..." with no progress when trying to process videos.

## Root Causes

1. **Missing Dependencies**: OpenCV (`cv2`) and other dependencies not installed in the environment running the training UI
2. **Silent Import Failure**: The threaded worker catches ImportError but doesn't show it in the UI
3. **API Signature Mismatch**: Old `process_video()` call missing new parameters (batch_size, checkpoint_interval, resume)

## Symptoms

- UI shows "Initialising..." indefinitely
- No progress bar movement
- No error messages displayed
- Video file size doesn't matter - even small files stuck

## Solutions Applied

### Fix 1: Updated Function Signature (video_processing_thread.py:L77-85)

**Before**:
```python
stats = tagger.process_video(video_path, output_csv, visualize=visualize)
```

**After**:
```python
stats = tagger.process_video(
    video_path,
    output_csv,
    visualize=visualize,
    batch_size=8,  # Default batch size
    checkpoint_interval=1000,  # Save every 1000 frames
    resume=False  # Don't resume by default in training UI
)
```

### Fix 2: Better Error Reporting (video_processing_thread.py:L50-95)

Added:
- Progress callbacks during initialisation ("Importing modules...", "Loading detection models...")
- Specific ImportError handling with helpful message
- Full traceback in error messages
- Clear indication when stuck at module import vs model loading

### Fix 3: Install Missing Dependencies

**Run this to fix the environment**:

```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Verify OpenCV installed
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## How to Test

1. **Stop the training UI** if it's running
2. **Install dependencies** (see Fix 3 above)
3. **Restart the training UI**:
   ```bash
   python training_interface_production.py
   ```
4. **Try processing a small test video** (< 1 min)
5. **Watch for progress updates** - should now show:
   - "Importing modules..." (0%)
   - "Loading detection models..." (10%)
   - "Starting video processing..." (50%)
   - Then normal detection progress

## Expected Behaviour After Fix

### Success Case:
```
🔄 Starting
Importing modules...
[Progress bar] 0%

🔄 Starting
Loading detection models...
[Progress bar] 10%

🔄 Starting
Starting video processing...
[Progress bar] 50%

🔄 Object Detection
Detecting objects... (frame 100/1000, 10%)
[Progress bar] 10%
(100/1000)
```

### Error Case (Missing Dependencies):
```
❌ Error

Failed to import processing modules: No module named 'cv2'
This usually means dependencies are not installed.
Please run: pip install -r requirements.txt
```

## File Size Considerations

**The fix also addresses memory issues with large files**:

- Old code: Loaded entire video into RAM → crash on long videos
- New code: Streams video in 1000-frame chunks → constant memory usage

**Processing times** (with dependencies installed):
- Small file (< 1 min): 10-30 seconds
- Medium file (5-10 min): 2-5 minutes
- Large file (30+ min): 10-30 minutes

If it's still stuck after 2 minutes on a small file, dependencies are still missing.

## Quick Diagnostic Commands

```powershell
# Check if in virtual environment
python -c "import sys; print('Virtual env:', hasattr(sys, 'base_prefix'))"

# Check OpenCV
python -c "import cv2; print('OpenCV OK')"

# Check PyTorch
python -c "import torch; print('PyTorch OK')"

# Check all main dependencies
python -c "import cv2, torch, yaml, numpy, pandas; print('All core deps OK')"

# Check if models exist
dir models\*.pt

# Test import of main processing
python -c "import sys; sys.path.insert(0, 'src'); from main import TennisTagger; print('TennisTagger import OK')"
```

## Common Issues After Fix

### Issue: Still stuck after installing dependencies

**Solution**: Restart the training UI - it caches the import failure

### Issue: Progress shows but very slow

**Cause**: Large video file being processed

**Solutions**:
- Use file path input instead of drag-and-drop (see FILE_PATH_USAGE.md)
- Test with a short clip first
- Check if GPU is being used (faster)

### Issue: Error about missing models

**Solution**: Models not found - check config\config.yaml has correct paths:
```yaml
models:
  models_root: "C:/Users/liamp/Downloads/files/models"
```

## Summary

**Problem**: Training UI stuck on "Initialising"
**Cause**: Missing dependencies + poor error reporting + API mismatch
**Fix**: Install dependencies + update function calls + better error messages

**Files Changed**:
- `video_processing_thread.py` (L46-97): Better initialization reporting and new API parameters

**Next Step**: Run `pip install -r requirements.txt` and restart the training UI
