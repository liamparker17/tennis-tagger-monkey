# Tennis Tagger - Example Usage & Walkthrough

## Example 1: Process Your First Match

### Step 1: Prepare Your Video
Place your video in the `data/inference` directory:
```bash
cp /path/to/my_match.mp4 data/inference/
```

### Step 2: Run the Tagger
```bash
python main.py \
  --video data/inference/my_match.mp4 \
  --output data/output/my_match_tagged.csv \
  --gpu
```

### Expected Output
```
2024-01-15 10:00:00 - TennisTagger - INFO - ============================================================
2024-01-15 10:00:00 - TennisTagger - INFO - Tennis Match Auto-Tagging System
2024-01-15 10:00:00 - TennisTagger - INFO - ============================================================
2024-01-15 10:00:00 - TennisTagger - INFO - Initializing detection models...
2024-01-15 10:00:01 - VideoProcessor - INFO - Loading video: data/inference/my_match.mp4
2024-01-15 10:00:01 - VideoProcessor - INFO - Video properties: 1920x1080 @ 30.00fps, 54000 frames, 1800.00s
2024-01-15 10:00:01 - VideoProcessor - INFO - Extracted 54000 frames
2024-01-15 10:00:02 - TennisTagger - INFO - Detecting players and ball...
2024-01-15 10:00:02 - TennisTagger - INFO - Processing frame 0/54000
2024-01-15 10:05:00 - TennisTagger - INFO - Processing frame 10000/54000
...
2024-01-15 10:15:00 - TennisTagger - INFO - Detection complete. Analyzing events...
2024-01-15 10:15:05 - TennisTagger - INFO - Found 245 serves, 1823 strokes, 234 rallies
2024-01-15 10:15:06 - CSVGenerator - INFO - Generating CSV output...
2024-01-15 10:15:07 - CSVGenerator - INFO - Generated CSV with 234 rows
2024-01-15 10:15:07 - TennisTagger - INFO - CSV saved to: data/output/my_match_tagged.csv
2024-01-15 10:15:07 - TennisTagger - INFO - ============================================================
2024-01-15 10:15:07 - TennisTagger - INFO - Processing Complete!
2024-01-15 10:15:07 - TennisTagger - INFO - ============================================================
```

### Step 3: Review the Output
Open `data/output/my_match_tagged.csv` in Excel or Dartfish.

## Example 2: Using the QC Feedback Loop

### Workflow
```bash
# 1. Generate predictions
python main.py \
  --video data/inference/match_final.mp4 \
  --output data/output/match_predicted.csv

# 2. Human reviews and corrects the CSV
# (Edit in Excel, save as match_corrected.csv)

# 3. Apply QC feedback
python src/qc_feedback.py \
  --predicted data/output/match_predicted.csv \
  --corrected data/output/match_corrected.csv \
  --update_models \
  --report

# Expected output:
# ============================================================
# QC Feedback Summary
# ============================================================
# Overall Accuracy: 87.34%
# Total Corrections: 156
#
# Top 5 Lowest Accuracy Columns:
#   C1: Serve +1 Stroke: 72.45%
#   B2: Return Data: 78.23%
#   E3: Last Shot Error: 81.56%
#   A3: Serve Placement: 83.12%
#   D1: Return +1 Stroke: 84.67%
#
# Recommendations:
#   - Consider retraining stroke classifier with more labeled data
#   - Improve court detection and ball tracking for better placement accuracy
```

## Example 3: Training Custom Models

### Scenario: You have 50 labeled matches

#### Organize Your Data
```bash
data/training/
├── wimbledon_2023_final.mp4
├── wimbledon_2023_final.csv
├── us_open_2023_sf1.mp4
├── us_open_2023_sf1.csv
...
└── french_open_2023_qf2.mp4
└── french_open_2023_qf2.csv
```

#### Validate Data
```bash
python scripts/validate_training_data.py --data_dir data/training

# Output:
# Validating training data in: data/training
# ============================================================
#
# Checking: wimbledon_2023_final.csv
#   CSV: ✓ Valid CSV with 187 rows
#   Video: ✓ 1920x1080, 30.00fps, 45000 frames
#
# ...
#
# ============================================================
# VALIDATION SUMMARY
# ============================================================
# Total pairs: 50
# Valid pairs: 50
# Issues: 0
#
# ✓ All training data is valid!
```

#### Train Models
```bash
# Train stroke classifier
python scripts/train_custom.py \
  --task stroke_classification \
  --data_dir data/training \
  --epochs 50 \
  --batch_size 16

# Expected output:
# Training stroke_classification model...
# Using device: cuda
# Loaded 50 samples
# Epoch 1/50: 100%|████████| 40/40 [00:15<00:00, 2.67it/s, loss=1.234, acc=65.2]
# Epoch 1: Val Loss: 1.156, Val Acc: 68.45%
# Saved best model
# ...
# Epoch 50: Val Loss: 0.234, Val Acc: 91.23%
# Training complete!
```

#### Update Configuration
```yaml
# config/config.yaml
events:
  stroke_classification:
    model: "models/stroke_classification_best.pt"
    confidence_threshold: 0.7  # Adjust based on validation
```

#### Test Improved Model
```bash
python main.py \
  --video data/inference/test_match.mp4 \
  --output data/output/test_improved.csv \
  --model_path models/stroke_classification_best.pt
```

## Example 4: Batch Processing Tournament

### Scenario: Process all matches from a tournament

```bash
# Place all videos in tournament directory
data/inference/tournament_2024/
├── round_1_match_1.mp4
├── round_1_match_2.mp4
├── round_1_match_3.mp4
...
├── semifinal_1.mp4
├── semifinal_2.mp4
└── final.mp4

# Batch process
python scripts/batch_process.py \
  --input_dir data/inference/tournament_2024 \
  --output_dir data/output/tournament_2024 \
  --gpu \
  --pattern "*.mp4"

# Expected output:
# Found 15 videos to process
# ======================================================================
# Processing 1/15: round_1_match_1.mp4
# ======================================================================
# ...
# ✓ Successfully processed: round_1_match_1.mp4
# ...
# ======================================================================
# Processing 15/15: final.mp4
# ======================================================================
# ...
# ✓ Successfully processed: final.mp4
#
# ======================================================================
# BATCH PROCESSING SUMMARY
# ======================================================================
# Total videos: 15
# Successful: 15
# Failed: 0
#
# Summary saved to: data/output/tournament_2024/batch_summary_20240115_143022.txt
```

## Example 5: Performance Optimization

### Scenario: Video processing is too slow

#### Check Current Performance
```bash
python main.py --video test.mp4 --output test.csv

# Takes 2 hours for 1 hour video (0.5x real-time)
```

#### Option 1: Enable GPU
```bash
# Install GPU dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Process with GPU
python main.py --video test.mp4 --output test.csv --gpu

# Now takes 15 minutes (4x real-time)
```

#### Option 2: Reduce Resolution
```yaml
# config/config.yaml
video:
  resize_height: 480  # Down from 720
  frame_skip: 2       # Process every other frame
```

#### Option 3: Use Faster Model
```yaml
# config/config.yaml
detection:
  player_detector:
    model: "yolov8n.pt"  # Fastest model (was yolov8x.pt)
```

## Example 6: Troubleshooting

### Problem: CUDA Out of Memory

```bash
# Error:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution:**
```yaml
# config/config.yaml
hardware:
  batch_size: 2  # Reduce from 8

video:
  resize_height: 480  # Reduce from 720

detection:
  player_detector:
    model: "yolov8s.pt"  # Use smaller model
```

### Problem: Poor Stroke Classification

**Solution: Collect more training data and fine-tune**

```bash
# 1. Process matches and collect corrections
python main.py --video match1.mp4 --output predicted1.csv
# (Manually correct predicted1.csv → corrected1.csv)

# Repeat for 20+ matches

# 2. Generate training data
for i in {1..20}; do
  python src/qc_feedback.py \
    --predicted predicted${i}.csv \
    --corrected corrected${i}.csv \
    --update_models
done

# 3. Retrain
python scripts/train_custom.py \
  --task stroke_classification \
  --data_dir data/training/corrections \
  --epochs 30

# 4. Test improved model
python main.py \
  --video test_match.mp4 \
  --output test_improved.csv \
  --model_path models/stroke_classification_best.pt
```

## Example 7: Using the GUI

### Launch GUI
```bash
python src/gui.py
```

### GUI Workflow:
1. Click **"Browse"** next to "Video" field
2. Select your match video
3. Output CSV path is auto-suggested
4. Check **"Use GPU Acceleration"** if available
5. Check **"Generate Annotated Video"** if you want visualization
6. Click **"Process Video"**
7. Monitor progress in log window
8. When complete, find output CSV in specified location

## Example 8: Continuous Improvement Workflow

### Month 1: Initial Setup
```bash
# Download models
python scripts/download_models.py

# Process first 10 matches
python scripts/batch_process.py --input_dir data/inference/month1 --output_dir data/output/month1 --gpu
```

### Month 2: First Training Cycle
```bash
# Collect corrections from month 1
# (10 corrected CSVs in data/training/)

# Train custom model
python scripts/train_custom.py --task stroke_classification --data_dir data/training --epochs 50

# Process month 2 with improved model
python scripts/batch_process.py --input_dir data/inference/month2 --output_dir data/output/month2 --gpu
```

### Month 3+: Iterative Improvement
```bash
# QC check random samples
python src/qc_feedback.py --predicted random_sample.csv --corrected random_sample_qc.csv --report

# Track accuracy trend
python src/qc_feedback.py --report --history_dir data/qc_history

# Retrain quarterly with all accumulated corrections
python scripts/train_custom.py --task stroke_classification --data_dir data/training --epochs 30
```

## Tips & Best Practices

### 1. Video Quality Matters
- Use 720p or higher resolution
- Stable camera angles work best
- Good lighting conditions
- Avoid excessive motion blur

### 2. Start Small
- Begin with 5-10 well-labeled matches
- Fine-tune models incrementally
- Don't expect perfect accuracy initially

### 3. QC Strategy
- Review 10-20% of predictions
- Focus on diverse match types (different surfaces, players, conditions)
- Correct systematically, not randomly

### 4. Model Selection
- Use yolov8x for best accuracy (but slower)
- Use yolov8n for fastest processing (lower accuracy)
- yolov8m is a good balance

### 5. GPU Utilization
- Monitor GPU memory: `nvidia-smi`
- Reduce batch_size if out of memory
- Process multiple videos sequentially, not in parallel

### 6. Backup Strategy
- Save all corrected CSVs
- Keep QC history
- Version control your config changes
- Archive trained models with date stamps
