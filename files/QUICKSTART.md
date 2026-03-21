# Quick Start Guide

Get the Tennis Tagger running in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone or download the project
cd tennis_tagger

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Models (1 minute)

```bash
python scripts/download_models.py
```

This will verify that models are ready. YOLOv8 models download automatically on first use.

## Step 3: Tag Your First Video (2 minutes)

### Option A: Command Line

```bash
# Basic usage
python main.py --video your_match.mp4 --output tags.csv

# With GPU acceleration (faster)
python main.py --video your_match.mp4 --output tags.csv --device cuda
```

### Option B: Web GUI

```bash
# Launch GUI
python gui/app.py

# Then open browser to: http://localhost:7860
```

## Step 4: Review Results

Open `tags.csv` in Excel or any spreadsheet software. You'll see:
- Point-by-point tagging
- Serve and return data
- Stroke classifications
- Placements and depths
- Score progression

## Test with Sample Data

If you don't have a video yet, create a simple test:

```bash
# This will run through the pipeline with placeholder data
python main.py --video test.mp4 --output test_tags.csv
```

## Common Issues

### "CUDA not available"
- System will automatically fall back to CPU
- Processing will be slower but will work
- To enable GPU: Install CUDA toolkit for your GPU

### "Out of memory"
- Reduce batch size in `config/config.yaml`:
  ```yaml
  processing:
    batch_size: 2  # Reduce from 8
  ```

### Slow processing on CPU
- Normal! CPU processing is ~0.5x real-time
- Consider using a smaller model:
  ```bash
  # Edit config/config.yaml
  models:
    detector: "yolov8n"  # Smaller, faster model
  ```

## Next Steps

### Apply QC Feedback
1. Generate initial tags
2. Review and correct in Excel
3. Feed back to system:
```bash
python main.py --video match.mp4 --qc_csv corrected.csv --update_model
```

### Train on Your Data
```bash
# Prepare your labeled videos and CSVs
python scripts/train.py \
    --video_dir training_videos/ \
    --csv_dir training_labels/ \
    --epochs 50
```

### Batch Process
```bash
# Process multiple videos
python main.py --batch --input_dir videos/ --output_dir results/
```

## Performance Expectations

| Hardware | Processing Speed | Example |
|----------|-----------------|---------|
| RTX 3050 | ~3x real-time | 1 hour video = 20 min |
| RTX 2060 | ~2x real-time | 1 hour video = 30 min |
| CPU only | ~0.5x real-time | 1 hour video = 2 hours |

## Getting Help

1. Check README.md for detailed documentation
2. Review config/config.yaml for all options
3. Enable debug mode: `python main.py --video match.mp4 --debug`

## That's It!

You now have a fully functional tennis tagging system running locally on your machine. 🎾
