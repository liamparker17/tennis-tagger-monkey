# Tennis Match Auto-Tagging System

## Overview
Automated tennis match analysis system that detects serves, rallies, strokes, placements, and scores from video footage, outputting Dartfish-compatible CSV files.

## System Requirements

### Minimum Requirements
- CPU: Intel i5 or AMD Ryzen 5 (4+ cores)
- RAM: 16GB
- Storage: 100GB free space
- OS: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+

### Recommended (for GPU acceleration)
- GPU: NVIDIA RTX 3050 or better (6GB+ VRAM)
- RAM: 32GB
- CUDA 11.8+ and cuDNN 8.6+

### Software Dependencies
- Python 3.9-3.11
- CUDA Toolkit (for GPU acceleration)
- FFmpeg

## Installation

### 1. Clone or Extract Project
```bash
cd tennis_tagger
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# For CPU-only
pip install -r requirements.txt

# For GPU (NVIDIA)
pip install -r requirements_gpu.txt
```

### 4. Download Pre-trained Models
```bash
python scripts/download_models.py
```

## Quick Start

### Process a Single Video
```bash
python main.py --video path/to/match.mp4 --output results.csv
```

### With GUI
```bash
python src/gui.py
```

### Batch Processing
```bash
python scripts/batch_process.py --input_dir data/inference --output_dir data/output
```

## Training with Your Historical Data

### 1. Prepare Training Data
Place your labeled videos and corresponding CSVs in `data/training/`

### 2. Fine-tune Models
```bash
python scripts/train_custom.py --data_dir data/training --epochs 50
```

### 3. Apply QC Feedback
```bash
python src/qc_feedback.py --predicted predicted.csv --corrected qc_corrected.csv --update_models
```

## Performance Expectations

### Processing Speed (RTX 3050)
- Real-time factor: 2-4x (process 1 hour in 15-30 minutes)
- CPU only: 0.3-0.5x (process 1 hour in 2-3 hours)

### Accuracy (after fine-tuning)
- Serve detection: 95-98%
- Stroke classification: 85-92%
- Rally segmentation: 90-95%
- Placement estimation: 75-85%

See full documentation in README for detailed instructions.
