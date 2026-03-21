# How to Use File Path Input (Avoiding White Screen)

## The Problem

Dragging large video files (>1GB) into Gradio causes:
- **White screen** for 2-5 minutes while file copies
- UI completely frozen during upload
- No progress indicator
- Very frustrating experience

## The Solution: Use File Path Input

### Step-by-Step Instructions

#### 1. Get Your Video File Path

**Option A: Copy as Path (Windows)**
1. Navigate to your video file in File Explorer
2. Hold **Shift** and **right-click** the file
3. Select **"Copy as path"**
4. The path is now in your clipboard with quotes: `"C:\Users\liamp\Videos\match.mp4"`

**Option B: Copy from Address Bar**
1. Navigate to your video file in File Explorer
2. Click the address bar at the top
3. Copy the folder path
4. Add your filename: `C:\Users\liamp\Videos\match.mp4`

#### 2. Paste Into the App

1. Open the Training interface
2. Find the **"Video File Path"** textbox at the top
3. Paste your path (quotes will be automatically removed)
4. Example paths:
   - `C:\Users\liamp\Videos\match.mp4`
   - `"C:\Users\liamp\Videos\match.mp4"` (quotes OK - they're removed)
   - `E:\Tennis\2024-03-15.mp4`
   - `D:\Recordings\long_match_7h.mov`

#### 3. Upload Your CSV

Drag-and-drop your CSV file (CSVs are small, so no white screen issue)

#### 4. Process or Train

Click either:
- **"🎬 Process Video for Feature Extraction"** - to extract features
- **"✅ Submit Video+CSV Pair for Training"** - to save for training

## What Changed

### Path Cleaning

All functions now strip quotes and whitespace:
```python
clean_path = video_path_str.strip().strip('"').strip("'").strip()
# "C:\Users\liamp\Videos\match.mp4" → C:\Users\liamp\Videos\match.mp4
```

### Upload Progress Indicator

If you do drag-and-drop (small files):
- Shows file size after upload completes
- Warns if file > 1GB (suggesting path input next time)
- Example: `✅ Upload Complete! 1 file(s) uploaded (2.34 GB)`

### Better UI Guidance

- Clear labels: "Video File Path (for large videos - RECOMMENDED)"
- Helpful placeholder: `C:\Users\liamp\Videos\match.mp4`
- Tooltip: "Paste the full file path here (right-click file → Copy as path). No quotes needed."
- Tip: "💡 Right-click your video file and select 'Copy as path', then paste above"

## Comparison

### Using Drag-and-Drop (BAD for large files)

```
User drags 7.5 hour video (18 GB)
    ↓
[WHITE SCREEN - 3-5 minutes]
    ↓
Gradio copies 18 GB to temp folder
    ↓
[WHITE SCREEN continues]
    ↓
Finally upload completes
    ↓
Can start processing
```

**Time wasted:** 3-5 minutes of white screen
**User experience:** Terrible

### Using File Path Input (GOOD!)

```
User copies file path
    ↓
Pastes into textbox (1 second)
    ↓
Clicks button
    ↓
Processing starts IMMEDIATELY
    ↓
Progress UI shows real-time updates
```

**Time wasted:** 0 seconds
**User experience:** Perfect!

## Common Issues & Solutions

### Issue: "Video file not found"

**Cause:** Path has quotes or typos

**Solution:**
- Make sure path is correct
- Quotes are OK (they're automatically removed)
- Check file actually exists at that location
- Use absolute path, not relative

### Issue: "No matching video-CSV pairs found"

**Cause:** Video and CSV have different base names

**Solution:**
- Video: `match1.mp4`
- CSV: `match1.csv`
- Base names must match!

### Issue: Still see white screen

**Cause:** Using drag-and-drop instead of file path

**Solution:**
- **DON'T drag the video**
- Use the file path input instead
- Only drag-and-drop files < 1 GB

## Technical Details

### Why Gradio Copies Files

Gradio was designed for web apps where:
- Files come from user's browser
- Server needs to receive and store them
- Copying is necessary for security

For desktop apps:
- Files already on local disk
- No need to copy
- Can read directly from source
- **Much faster!**

### File Path vs Upload

| Method | Small Files (<1GB) | Large Files (>1GB) |
|--------|-------------------|-------------------|
| **Drag-and-drop** | ✅ OK | ❌ White screen |
| **File path** | ✅ OK | ✅ Perfect! |

**Recommendation:** Always use file path for videos, regardless of size.

## Summary

1. ✅ **Right-click video → Copy as path**
2. ✅ **Paste into "Video File Path" textbox**
3. ✅ **Upload CSV** (drag-and-drop OK for small files)
4. ✅ **Click process or submit button**
5. ✅ **No white screen - instant progress!**

**Never drag-and-drop large videos again!**
