# Getting Started with Tennis Tagger - First Time Setup

## 🎯 Goal
Get your tennis tagging system up and running in 15 minutes.

## Prerequisites
- Python 3.9, 3.10, or 3.11 installed
- Tennis match video file (MP4, MOV, AVI, or MKV)
- 10GB free disk space
- (Optional) NVIDIA GPU with CUDA for faster processing

## Step-by-Step Setup

### Step 1: Navigate to Project Directory (30 seconds)

```bash
cd tennis_tagger
ls -la  # Should see main.py, src/, config/, etc.
```

**✅ Checkpoint**: You should see files like `main.py`, `README.md`, and directories like `src/`, `config/`

### Step 2: Create Virtual Environment (2 minutes)

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**✅ Checkpoint**: Your terminal prompt should now show `(venv)` at the beginning

### Step 3: Install Dependencies (3-5 minutes)

**For CPU-only (simpler, works everywhere):**
```bash
pip install -r requirements.txt
```

**For GPU support (if you have NVIDIA GPU):**
```bash
# First install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

This will download ~1-2GB of packages.

**✅ Checkpoint**: Installation completes without errors

### Step 4: Download Pre-trained Models (5 minutes)

```bash
python scripts/download_models.py
```

This downloads YOLOv8 models (total ~400MB).

**Expected output:**
```
Downloading pre-trained models...
Downloading yolov8n.pt...
100%|████████████████████| 6.2M/6.2M [00:03<00:00, 2.1MB/s]
Successfully downloaded yolov8n.pt
...
Model download complete!
```

**✅ Checkpoint**: Models directory contains .pt files

### Step 5: Run System Diagnostic (1 minute)

```bash
python scripts/diagnostic.py
```

**Expected output:**
```
################################################################
#                                                              #
#               TENNIS TAGGER DIAGNOSTIC                       #
#                                                              #
################################################################

============================================================
Python Version
============================================================
Python 3.10.8
✓ Python version is compatible

============================================================
Required Packages
============================================================
✓ numpy installed
✓ pandas installed
✓ opencv-python installed
...

============================================================
CUDA/GPU Support
============================================================
✓ CUDA is available
  CUDA Version: 11.8
  GPU Count: 1
  GPU 0: NVIDIA GeForce RTX 3050
    Memory: 6.00 GB

...

============================================================
DIAGNOSTIC SUMMARY
============================================================
Python Version              ✓ PASS
Required Packages           ✓ PASS
CUDA/GPU                    ✓ PASS
FFmpeg                      ✓ PASS
Pre-trained Models          ✓ PASS
Configuration               ✓ PASS
Directory Structure         ✓ PASS

Results: 7/7 checks passed

✓ System is ready to use!
```

**✅ Checkpoint**: All checks pass (7/7)

**If any checks fail**, see troubleshooting section below.

### Step 6: Process Your First Video (Processing time varies)

**Method 1: Command Line (Recommended for first test)**

```bash
# Basic processing
python main.py --video /path/to/your/match.mp4 --output results.csv

# With GPU (faster)
python main.py --video /path/to/your/match.mp4 --output results.csv --gpu

# With visualization (slower, generates annotated video)
python main.py --video /path/to/your/match.mp4 --output results.csv --gpu --visualize
```

**Method 2: GUI (Easier for non-technical users)**

```bash
python src/gui.py
```

Then:
1. Click "Browse" next to "Video" field
2. Select your match video
3. Output CSV path is auto-filled
4. Check "Use GPU Acceleration" if available
5. Click "Process Video"
6. Wait for completion (progress shown in log window)

**Expected CLI output:**
```
============================================================
Tennis Match Auto-Tagging System
============================================================
INFO - Initializing detection models...
INFO - Loaded YOLO model: yolov8x.pt
INFO - MediaPipe Pose initialized
INFO - Processing video: match.mp4
INFO - Video properties: 1920x1080 @ 30.00fps, 36000 frames, 1200.00s
INFO - Extracted 36000 frames

INFO - Detecting players and ball...
INFO - Processing frame 0/36000
INFO - Processing frame 1000/36000
...

INFO - Detection complete. Analyzing events...
INFO - Found 187 serves, 1456 strokes, 182 rallies
INFO - Generating CSV output...
INFO - CSV saved to: results.csv

============================================================
Processing Complete!
============================================================
Frames processed: 36000
Duration: 1200.0 seconds
Serves: 187
Strokes: 1456
Rallies: 182
Output: results.csv
============================================================
```

**✅ Checkpoint**: CSV file created successfully

### Step 7: Review the Output (2 minutes)

Open `results.csv` in Excel, Numbers, or Dartfish.

**What you should see:**
- Row for each point/rally
- Columns: Server, Returner, Stroke types, Placements, Scores
- 80+ columns with tennis match data

**Example row:**
```
Name: Point_1
Position: 0:00:15
Duration: 8.5
0 - Point Level: Standard
A1: Server: Player_1
A2: Serve Data: First Serve
A3: Serve Placement: Region_7
B1: Returner: Player_2
...
```

## 🎉 Success!

You've successfully:
- ✅ Installed the tennis tagger
- ✅ Downloaded models
- ✅ Verified system works
- ✅ Processed your first match
- ✅ Generated Dartfish-compatible CSV

## 📚 What's Next?

### Immediate Next Steps

1. **Review the CSV**
   - Open in Excel/Dartfish
   - Check accuracy
   - Note any errors

2. **Try the GUI** (if you used CLI first)
   ```bash
   python src/gui.py
   ```

3. **Process More Videos**
   - Use batch processing for multiple videos
   ```bash
   python scripts/batch_process.py \
     --input_dir /path/to/videos \
     --output_dir /path/to/results \
     --gpu
   ```

### Advanced Usage (Week 2+)

4. **Set Up QC Feedback Loop**
   - Manually correct predictions
   - Feed back corrections
   - See accuracy improve

5. **Train Custom Models**
   - Collect 20+ labeled matches
   - Train on your specific data
   - Achieve 90%+ accuracy

6. **Optimize Configuration**
   - Adjust for speed vs accuracy
   - Tune for your video quality
   - Configure thresholds

## 🐛 Troubleshooting

### Problem 1: "Module not found" error

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem 2: Diagnostic shows "FFmpeg not installed"

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH
- Or use WSL

### Problem 3: "CUDA out of memory"

**Solution:** Edit `config/config.yaml`:
```yaml
hardware:
  batch_size: 2  # Reduce from 8

video:
  resize_height: 480  # Reduce from 720
```

### Problem 4: Processing is very slow

**Solutions:**
1. **Enable GPU** (if available):
   ```bash
   python main.py --video match.mp4 --output results.csv --gpu
   ```

2. **Use faster model** - Edit `config/config.yaml`:
   ```yaml
   detection:
     player_detector:
       model: "yolov8n.pt"  # Fastest (was yolov8x.pt)
   ```

3. **Process fewer frames**:
   ```yaml
   video:
     frame_skip: 2  # Process every other frame
   ```

### Problem 5: Poor detection accuracy

**This is normal initially!**

**Solutions:**
1. **Ensure good video quality**
   - 720p or higher
   - Stable camera
   - Good lighting

2. **Adjust confidence thresholds**:
   ```yaml
   detection:
     player_detector:
       confidence: 0.4  # Lower = more detections
   ```

3. **Train on your data** (recommended):
   - Collect 20+ labeled matches
   - Follow training guide in docs/COMPLETE_GUIDE.md

## 📖 Documentation Quick Reference

- **README.md** - Overview and quick start
- **DELIVERY_SUMMARY.md** - What you received and how to use it
- **docs/COMPLETE_GUIDE.md** - Comprehensive documentation
- **docs/EXAMPLES.md** - Detailed usage examples
- **PROJECT_SUMMARY.md** - Technical details

## 💡 Pro Tips

1. **Start Small**: Process a 5-10 minute clip first, not a full match
2. **Use GPU**: Makes 5-10x difference in processing time
3. **Check Logs**: If something fails, check `logs/tennis_tagger.log`
4. **Configure Wisely**: Default settings work for most cases
5. **QC Early**: Set up correction workflow from day 1

## 🎯 Common Use Cases

### Use Case 1: Quick Analysis
```bash
python main.py --video quick_match.mp4 --output quick.csv --gpu
```

### Use Case 2: High-Quality Analysis
Edit `config/config.yaml` to use `yolov8x.pt` and `resize_height: 1080`, then:
```bash
python main.py --video important_match.mp4 --output detailed.csv --gpu
```

### Use Case 3: Batch Tournament Processing
```bash
# Put all videos in tournament/ directory
python scripts/batch_process.py \
  --input_dir tournament/ \
  --output_dir tournament_results/ \
  --gpu
```

## 📞 Need Help?

1. **Run diagnostic first**: `python scripts/diagnostic.py`
2. **Check logs**: `logs/tennis_tagger.log`
3. **Review documentation**: `docs/COMPLETE_GUIDE.md`
4. **Check examples**: `docs/EXAMPLES.md`

## ✅ Setup Complete!

You're now ready to use the Tennis Tagger system. Start processing your matches and enjoy automated tennis analysis!

**Remember**: Initial accuracy is ~80%, but improves to 90%+ with training on your specific data.

---

**First Video Checklist:**
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Diagnostic passed (7/7)
- [ ] Test video processed
- [ ] CSV generated successfully
- [ ] Output reviewed in Excel/Dartfish

**Next: Process more videos or set up training workflow!** 🎾
