# Tennis Tagger System Overview

## Architecture

The Tennis Tagger is a modular video analysis system designed for automated tennis match tagging with continuous improvement via QC feedback.

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     Main Pipeline                        │
│                                                          │
│  Video → Detection → Tracking → Classification          │
│              ↓          ↓            ↓                   │
│           Court    Strokes      Events                   │
│              ↓          ↓            ↓                   │
│         Placement   Scores    → CSV Export               │
└─────────────────────────────────────────────────────────┘
                          ↓
                    QC Feedback
                          ↓
                   Model Updates
```

### Module Descriptions

#### 1. Video Processing (`processing/video_processor.py`)
- **Purpose**: Load and preprocess video files
- **Key Features**:
  - Efficient frame extraction
  - Resolution normalization
  - FPS standardization
- **Input**: Video files (MP4, MOV, etc.)
- **Output**: Preprocessed frame arrays

#### 2. Object Detection (`models/detector.py`)
- **Purpose**: Detect players, ball, and rackets
- **Technology**: YOLOv8 (pre-trained + fine-tuned)
- **Key Features**:
  - Real-time multi-object detection
  - GPU acceleration support
  - Confidence filtering
- **Performance**: ~30-60 FPS on RTX 3050

#### 3. Multi-Object Tracking (`models/tracker.py`)
- **Purpose**: Maintain consistent object IDs across frames
- **Technology**: DeepSORT with Kalman filtering
- **Key Features**:
  - Handles occlusions
  - Re-identification
  - Smooth trajectories
- **Output**: Track sequences with IDs

#### 4. Court Detection (`processing/court_detector.py`)
- **Purpose**: Establish court coordinate system
- **Methods**: 
  - Line detection (Hough transform)
  - Keypoint matching
  - Homography estimation
- **Use**: Shot placement analysis

#### 5. Stroke Classification (`models/stroke_classifier.py`)
- **Purpose**: Classify stroke types
- **Technology**: 3D CNN (X3D architecture)
- **Classes**: 
  - Forehand, Backhand
  - Volleys (forehand/backhand)
  - Serve, Smash, Drop shot, Lob
- **Input**: Temporal video clips (16 frames)

#### 6. Event Detection (`models/event_detector.py`)
- **Purpose**: Identify high-level events
- **Detection**:
  - Serves (pose + trajectory)
  - Rallies (stroke sequences)
  - Points (rally endpoints)
  - Games & Sets (score progression)
- **Methods**: Temporal analysis + rule-based

#### 7. Score Tracking (`processing/score_tracker.py`)
- **Purpose**: Track match score
- **Methods**:
  - OCR on scoreboard region
  - State machine validation
  - Event-based updates
- **Validation**: Tennis scoring rules

#### 8. Placement Analysis (`processing/placement_analyzer.py`)
- **Purpose**: Analyze shot placements
- **Features**:
  - Court zone classification
  - Depth analysis (baseline/mid/net)
  - Trajectory patterns
- **Output**: Placement tags for each shot

#### 9. CSV Generation (`export/csv_generator.py`)
- **Purpose**: Create Dartfish-compatible output
- **Format**: 62 columns covering:
  - Point-level data
  - Serve/return info
  - Stroke sequences
  - Placements & depths
  - Scores & outcomes
  - Metadata

#### 10. QC Feedback (`qc/comparator.py`, `qc/feedback_loop.py`)
- **Purpose**: Continuous improvement from corrections
- **Process**:
  1. Compare predicted vs. corrected CSV
  2. Identify error patterns
  3. Generate training examples
  4. Update model weights
- **Update Strategy**: Incremental learning

## Data Flow

### Forward Pass (Prediction)
```
1. Video File
   ↓
2. Frame Extraction (video_processor)
   ↓
3. Court Detection (court_detector)
   ↓
4. Object Detection (detector)
   ↓  ↓  ↓
   Players  Ball  Rackets
   ↓
5. Object Tracking (tracker)
   ↓
6. Stroke Classification (stroke_classifier)
   ↓
7. Event Detection (event_detector)
   ↓
8. Score Tracking (score_tracker)
   ↓
9. Placement Analysis (placement_analyzer)
   ↓
10. CSV Generation (csv_generator)
   ↓
Result: Dartfish CSV
```

### Feedback Loop (Learning)
```
1. Predicted CSV + Corrected CSV
   ↓
2. Comparison (comparator)
   ↓
3. Error Analysis
   ↓
4. Correction Buffer
   ↓
5. Model Updates (feedback_loop)
   ↓
Improved Models for Next Prediction
```

## Performance Optimization

### GPU Acceleration
- Batch processing for detection
- Mixed precision training (FP16)
- CUDA optimizations

### Memory Management
- Frame-level processing (no full video in RAM)
- Batch size control
- Result streaming

### Speed Optimization
- Model selection (yolov8n = fast, yolov8x = accurate)
- Resolution scaling
- Frame skipping option

## Model Training

### Data Requirements
```
training_data/
├── videos/
│   ├── match001.mp4
│   ├── match002.mp4
│   └── ...
└── labels/
    ├── match001.csv  (Dartfish format)
    ├── match002.csv
    └── ...
```

### Training Process
1. **Data Preparation**: Extract frames + annotations
2. **Detector Training**: Fine-tune YOLOv8 on tennis data
3. **Stroke Classifier**: Train 3D CNN on stroke clips
4. **Event Detector**: Train temporal model
5. **Validation**: Evaluate on held-out data

### Fine-Tuning from QC
- Incremental updates from corrections
- Accumulate N corrections before update
- Save checkpoints after each update

## Configuration

All settings in `config/config.yaml`:

### Video Settings
- `fps`: Processing frame rate
- `resolution`: Frame dimensions
- `skip_frames`: Frame skipping

### Model Settings
- `detector`: YOLOv8 variant (n/s/m/l/x)
- `detector_confidence`: Detection threshold
- `stroke_classifier`: 3D CNN architecture

### Performance Settings
- `batch_size`: Frames per batch
- `num_workers`: Parallel workers
- `mixed_precision`: Enable FP16

### QC Settings
- `learning_rate`: Update learning rate
- `accumulate_corrections`: Update frequency
- `update_strategy`: incremental/batch

## API Usage

### Python API
```python
from main import TennisTagger

# Initialize
tagger = TennisTagger(device='cuda')

# Process video
stats = tagger.process_video(
    video_path='match.mp4',
    output_path='tags.csv'
)

# With QC feedback
tagger.process_video(
    video_path='match.mp4',
    output_path='tags.csv',
    qc_csv_path='corrected.csv',
    update_model=True
)
```

### Command Line
```bash
# Basic tagging
python main.py --video match.mp4 --output tags.csv

# With GPU
python main.py --video match.mp4 --device cuda

# QC feedback
python main.py --video match.mp4 --qc_csv corrected.csv --update_model

# Batch processing
python main.py --batch --input_dir videos/ --output_dir results/
```

### Web GUI
```bash
python gui/app.py
# Then navigate to http://localhost:7860
```

## Extensibility

### Adding New Stroke Types
1. Update `STROKE_CLASSES` in `stroke_classifier.py`
2. Add training examples
3. Retrain classifier

### Adding New Placement Zones
1. Edit `PLACEMENT_ZONES` in `placement_analyzer.py`
2. Update court zone definitions in `config.yaml`

### Custom Event Detection
1. Extend `EventDetector` class
2. Add detection methods
3. Update event types in CSV generator

## Dependencies

### Core
- PyTorch (2.0+): Deep learning framework
- OpenCV (4.8+): Video processing
- NumPy (1.24+): Numerical computing
- Pandas (2.0+): Data manipulation

### Computer Vision
- Ultralytics (8.0+): YOLOv8 models
- MediaPipe (0.10+): Pose estimation
- EasyOCR (1.7+): Score reading

### Tracking
- FilterPy (1.4+): Kalman filtering
- SciPy (1.10+): Optimization

### UI
- Gradio (3.50+): Web interface

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `batch_size` in config
- Use smaller model (yolov8n)
- Lower resolution

**Slow Processing**
- Enable GPU with `--device cuda`
- Use smaller model
- Increase `skip_frames`

**Inaccurate Tags**
- Lower confidence thresholds
- Train on your specific data
- Apply QC feedback

**Model Not Found**
- Run `python scripts/download_models.py`
- Models auto-download on first use

## Future Enhancements

- [ ] Real-time processing mode
- [ ] Multi-camera support
- [ ] Advanced analytics dashboard
- [ ] Cloud deployment option
- [ ] Mobile app integration
- [ ] Player identification
- [ ] Advanced shot prediction

## References

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/nwojke/deep_sort
- X3D: https://arxiv.org/abs/2004.04730
- Dartfish: https://www.dartfish.com

## License

MIT License - Free for commercial and personal use

## Support

For issues, questions, or contributions:
1. Check README.md
2. Review this document
3. Enable debug mode
4. Check configuration settings
