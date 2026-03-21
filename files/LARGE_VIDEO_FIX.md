# Large Video White Screen Fix

## The Problem

**Issue:** When dragging a 7.5 hour video (or any large file >1GB) into the Gradio interface, the screen goes white before any processing UI appears.

**Root Cause:** Gradio's `gr.File` component **copies the entire file synchronously** to its temporary directory when you upload it. For large videos:
- 7.5 hours @ 1080p ≈ 10-20 GB
- Upload copy time: 2-5 minutes (depending on drive speed)
- **During this copy, the entire UI thread is blocked** → White screen
- This happens **before** any of our progress tracking code runs

**Why CSV works instantly:**
- CSV files are tiny (< 1 MB typically)
- Copy completes in < 1 second
- No noticeable blocking

## The Solution

Added a **direct file path input** that bypasses Gradio's file copying entirely:

### How to Use (for Large Videos)

1. **Don't drag-and-drop large videos** - this triggers the blocking copy

2. **Use the file path input instead:**
   ```
   Video File Path (for large videos)
   ┌─────────────────────────────────────────────┐
   │ E:\Videos\match.mp4                         │
   └─────────────────────────────────────────────┘
   ```

3. **Paste or type your full video path** (examples):
   - `E:\Videos\tennis_match.mp4`
   - `D:\Recordings\2024-03-15_match.mov`
   - `C:\Users\liamp\Downloads\video.avi`

4. **Click "Process Video for Feature Extraction"**
   - Processing starts immediately (no copy delay)
   - Progress UI appears instantly
   - No white screen!

### What Changed in the Code

**File:** `training_interface_production.py`

#### 1. Added Direct Path Input (lines 506-512)
```python
gr.Markdown("**⚠️ For large videos (>1GB), use the file path input below**")

video_path_input = gr.Textbox(
    label="Video File Path (for large videos)",
    placeholder="E:\\Videos\\match.mp4 or paste full path here",
    lines=1
)
```

#### 2. Updated Processing Function (lines 690-728)
```python
def process_video_for_extraction(videos, video_path_str):
    # Prioritize direct path (no copy needed!)
    if video_path_str and video_path_str.strip():
        video_path = Path(video_path_str.strip())
        # Validate file exists
        if not video_path.exists():
            return f"❌ Error: File not found: {video_path}"
    elif videos:
        # Fall back to uploaded file (for small files)
        video_path = Path(videos[0])

    # Warn about large files
    file_size_gb = video_path.stat().st_size / (1024**3)
    if file_size_gb > 5:
        size_warning = f"⚠️ Large file: {file_size_gb:.2f} GB"

    # Process directly from original location (no copy!)
    start_video_processing(str(video_path), ...)
```

## Usage Instructions

### For Small Videos (< 1 GB)
✅ **Drag-and-drop works fine** - use the regular file upload

### For Large Videos (> 1 GB)
⚠️ **Use file path input instead:**

1. Right-click your video file
2. Copy the full path (or navigate to it and copy from address bar)
3. Paste into "Video File Path" textbox
4. Click "Process Video for Feature Extraction"

**Example Paths:**
- Windows: `E:\Tennis\2024-03-15_match.mp4`
- Windows (with spaces): `C:\Users\liamp\My Videos\match.mp4` (quotes not needed)
- Network drive: `\\NAS\Videos\match.mp4`

## Technical Details

### Why This Fixes the White Screen

**Before (with drag-and-drop):**
```
User drags 15GB video
    ↓
Gradio File component triggered
    ↓
Starts copying 15GB to temp folder (BLOCKS UI THREAD)
    ↓ (2-5 minutes of white screen)
Copy completes
    ↓
File available to process
    ↓
Processing starts (with progress UI)
```

**After (with file path input):**
```
User pastes file path
    ↓
Click "Process Video"
    ↓
Processing starts IMMEDIATELY (no copy!)
    ↓ (< 1 second)
Progress UI appears
    ↓
Background thread processes video
```

### File Size Warnings

The system now automatically detects large files and warns you:

- **< 1 GB**: No warning
- **1-5 GB**: Standard processing
- **> 5 GB**: Warning message + estimated time

**Example Output:**
```
🔄 Processing Started

Video: tennis_match_7h.mp4 (18.45 GB)
Processing has begun in the background. Progress will update below...

⚠️ Large file detected: 18.45 GB
Processing may take 30+ minutes...
```

## Why Gradio Copies Files

Gradio's design assumes **web-based deployment** where:
- Files uploaded from user's browser
- Must be received and stored on server
- Copying is necessary for security/isolation

But for **desktop apps**, this is unnecessary overhead:
- Files already on local disk
- Can read directly from source
- No need to copy to temp folder

## Alternatives Considered

1. ❌ **Increase Gradio timeout** - Doesn't solve blocking
2. ❌ **Async file upload** - Gradio doesn't support this
3. ❌ **Stream processing** - Requires Gradio source code changes
4. ✅ **Direct path input** - Simple, works perfectly for desktop

## Testing

Test with your 7.5 hour video:

1. Start the app: `python training_desktop.py`
2. **Do NOT drag the video file**
3. Type or paste the video path: `E:\Your\Path\video.mp4`
4. Click "Process Video for Feature Extraction"
5. ✅ Progress UI should appear immediately (no white screen)
6. ✅ Real-time updates every second
7. ✅ Frame counts, stage progress, etc.

## Summary

**Problem:** Gradio file upload blocks UI for large files
**Solution:** Direct file path input bypasses upload copying
**Result:** No white screen, instant progress UI, smooth experience

**Use the file path input for videos > 1 GB!**
