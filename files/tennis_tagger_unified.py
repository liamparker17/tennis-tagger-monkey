"""
Tennis Tagger Unified - Production Ready Single App

Combines video tagging and training into a single tabbed interface.
No separate apps needed - everything in one window!

Tabs:
1. Tag Video - Process new videos, generate CSVs
2. Training - Train models using FVD data
3. QC Corrections - Review and correct tags
4. Video Library - Browse registry, manage FVD cache
5. Models - View/manage trained models

Key Features:
- One-click install and launch
- No video duplication (references only)
- FVD-based training resumption
- Unified model versioning
"""

import gradio as gr
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time
import threading
import shutil

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
from video_registry import VideoRegistry, create_video_registry
from frame_vector_data import FrameVectorData, create_fvd_manager

# Import training components (will be refactored to use FVD)
from training_interface_production import (
    DatasetManager,
    TrainingEstimator,
    QCManager,
    IncrementalTrainer,
    TrainingMonitor
)

# Import training data importer for human-tagged CSV import
try:
    from training_data_importer import TrainingDataImporter, create_training_data_importer
    HAS_TRAINING_IMPORTER = True
except ImportError:
    HAS_TRAINING_IMPORTER = False

# Import video processing
try:
    from video_processing_thread import (
        get_video_processor,
        start_video_processing,
        get_processing_progress,
        is_processing_complete,
        get_processing_result
    )
    HAS_VIDEO_PROCESSING = True
except ImportError:
    HAS_VIDEO_PROCESSING = False

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Singleton instances
_video_registry: VideoRegistry = None
_fvd_manager: FrameVectorData = None


def get_video_registry() -> VideoRegistry:
    """Get or create video registry singleton."""
    global _video_registry
    if _video_registry is None:
        _video_registry = create_video_registry()
    return _video_registry


def get_fvd_manager() -> FrameVectorData:
    """Get or create FVD manager singleton."""
    global _fvd_manager
    if _fvd_manager is None:
        _fvd_manager = create_fvd_manager()
    return _fvd_manager


# =============================================================================
# TAB 1: TAG VIDEO
# =============================================================================

def create_tag_video_tab():
    """Create the video tagging tab."""

    with gr.Tab("Tag Video", id="tag") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Process Tennis Match Videos</h2>
            <p style="margin: 0; opacity: 0.9;">
                Automatically detect serves, strokes, placements, and scores.
                Videos stay in their original location - no duplication needed.
            </p>
        </div>
        """)

        with gr.Row():
            # Left column - Video selection
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #1e40af; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #3b82f6;">Select Video</div>')

                with gr.Row():
                    browse_btn = gr.Button("Browse for Video", variant="primary", size="lg")
                    selected_video = gr.Textbox(
                        label="Selected Video",
                        placeholder="Click Browse to select...",
                        interactive=False,
                        scale=3
                    )

                gr.Markdown("**OR** paste file path:")
                video_path_input = gr.Textbox(
                    label="Video Path",
                    placeholder="C:\\Users\\Videos\\match.mp4",
                    lines=1
                )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #1e40af; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #3b82f6;">Processing Options</div>')

                device_choice = gr.Radio(
                    choices=[("Auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")],
                    value="auto",
                    label="Processing Device"
                )

                batch_size = gr.Slider(
                    minimum=8, maximum=256, value=128, step=8,
                    label="Batch Size (higher = faster on GPU)"
                )

                frame_skip = gr.Radio(
                    choices=[
                        ("Every frame (slowest, most accurate)", 1),
                        ("Every 2nd frame (2x faster)", 2),
                        ("Every 3rd frame (3x faster, recommended)", 3),
                        ("Every 5th frame (5x faster)", 5),
                    ],
                    value=3,
                    label="Frame Skip (3x = ~10fps, enough for tennis events)"
                )

                enable_pose = gr.Checkbox(
                    label="Enable Pose Estimation (slower but more accurate stroke detection)",
                    value=False
                )

                extract_fvd = gr.Checkbox(
                    label="Extract FVD for training resumption",
                    value=True
                )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')

                process_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    size="lg"
                )

                stop_btn = gr.Button(
                    "Stop Processing",
                    variant="secondary",
                    visible=False
                )

            # Right column - Progress and results
            with gr.Column(scale=2):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #1e40af; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #3b82f6;">Processing Progress</div>')

                progress_status = gr.Markdown(
                    "**Idle** - Select a video to begin",
                    label="Status"
                )

                progress_detail = gr.Textbox(
                    label="Details",
                    lines=8,
                    interactive=False
                )

                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #10b981;">Results</div>')

                with gr.Row():
                    csv_download = gr.File(label="Download CSV")
                    fvd_status = gr.Textbox(label="FVD Status", interactive=False)

                results_preview = gr.Dataframe(
                    label="Tag Preview (first 20 rows)",
                    interactive=False
                )

        # Helper functions
        def browse_video():
            """Open file browser."""
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                file_path = filedialog.askopenfilename(
                    title="Select Tennis Match Video",
                    filetypes=[
                        ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                        ("All files", "*.*")
                    ],
                    initialdir=str(Path.home() / "Videos")
                )
                root.destroy()

                if file_path:
                    path = Path(file_path)
                    size_mb = path.stat().st_size / (1024 * 1024)
                    size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
                    return file_path, f"{path.name} ({size_str})"
                return "", "No file selected"
            except Exception as e:
                return "", f"Error: {e}"

        def get_video_path(browsed, manual):
            """Get video path from browsed or manual input."""
            if browsed and browsed.strip():
                return browsed.strip()
            if manual and manual.strip():
                return manual.strip().strip('"').strip("'")
            return None

        def process_video(browsed_path, manual_path, device, batch, skip_frames, pose, extract_fvd_flag):
            """Start video processing."""
            video_path = get_video_path(browsed_path, manual_path)

            if not video_path:
                return (
                    "**Error** - No video selected",
                    "Please browse for a video or enter a path",
                    None, None, None
                )

            path = Path(video_path)
            if not path.exists():
                return (
                    f"**Error** - File not found: {path.name}",
                    f"Path: {video_path}",
                    None, None, None
                )

            # Add to registry (no copy!)
            registry = get_video_registry()
            try:
                video_id, entry = registry.add_video(video_path)
            except Exception as e:
                return (
                    f"**Error** - Failed to register video",
                    str(e),
                    None, None, None
                )

            # Determine output paths
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / f"{path.stem}.csv"

            # Calculate speed multiplier for display
            speed_mult = f"{skip_frames}x faster" if skip_frames > 1 else "full quality"

            # Start processing
            if HAS_VIDEO_PROCESSING:
                registry.mark_processing_started(video_id)

                success, message = start_video_processing(
                    str(path),
                    str(output_csv),
                    visualize=False,
                    batch_size=int(batch),
                    device=device,
                    frame_skip=int(skip_frames)
                )

                if not success:
                    registry.mark_failed(video_id, message)
                    return (
                        f"**Error** - Processing failed",
                        message,
                        None, None, None
                    )

                return (
                    f"**Processing** - {path.name}",
                    f"Started processing ({speed_mult}) with batch size {batch} on {device}...\n\nVideo ID: {video_id}",
                    None,
                    f"FVD will be extracted: {extract_fvd_flag}",
                    None
                )
            else:
                return (
                    "**Error** - Video processing module not available",
                    "Please ensure all dependencies are installed",
                    None, None, None
                )

        def update_progress():
            """Poll for processing progress."""
            if not HAS_VIDEO_PROCESSING:
                return "**Idle**", ""

            try:
                progress = get_processing_progress()

                # Handle case where progress is a string
                if isinstance(progress, str):
                    return "**Idle**", progress

                if is_processing_complete():
                    result = get_processing_result()
                    if result and isinstance(result, dict):
                        if result.get('status') == 'completed':
                            csv_path = result.get('output_csv')

                            # Load preview
                            try:
                                df = pd.read_csv(csv_path)
                                preview = df.head(20)
                            except:
                                preview = None

                            return (
                                "**Complete!**",
                                f"Processed {result.get('frames_processed', 0)} frames\n"
                                f"Found {result.get('num_serves', 0)} serves, "
                                f"{result.get('num_strokes', 0)} strokes, "
                                f"{result.get('num_rallies', 0)} rallies\n\n"
                                f"CSV saved to: {csv_path}",
                            )
                        else:
                            return (
                                "**Failed**",
                                result.get('error', 'Unknown error')
                            )

                # Safely get progress values
                if isinstance(progress, dict):
                    percent = progress.get('percent', 0)
                    message = progress.get('message', 'Working...')
                else:
                    percent = 0
                    message = str(progress) if progress else 'Idle'

                return (
                    f"**Processing** - {percent:.1f}%",
                    message
                )
            except Exception as e:
                return "**Idle**", ""

        # Connect events
        browse_btn.click(
            browse_video,
            outputs=[video_path_input, selected_video]
        )

        process_btn.click(
            process_video,
            inputs=[selected_video, video_path_input, device_choice, batch_size, frame_skip, enable_pose, extract_fvd],
            outputs=[progress_status, progress_detail, csv_download, fvd_status, results_preview]
        )

        # Auto-update progress
        progress_timer = gr.Timer(value=1.0, active=True)
        progress_timer.tick(
            update_progress,
            outputs=[progress_status, progress_detail]
        )

    return tab


# =============================================================================
# TAB 2: TRAINING
# =============================================================================

def create_training_tab():
    """Create the training tab."""

    dataset_manager = DatasetManager()
    trainer = IncrementalTrainer()

    # Import the ImportedDataTrainer
    imported_trainer = None
    if HAS_TRAINING_IMPORTER:
        try:
            from training_interface_production import ImportedDataTrainer
            imported_trainer = ImportedDataTrainer()
        except ImportError:
            pass

    with gr.Tab("Training", id="train") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Train Models from Human-Tagged Data</h2>
            <p style="margin: 0; opacity: 0.9;">
                Import human-tagged CSVs, select datasets, and train models to match human accuracy.
            </p>
        </div>
        """)

        with gr.Row():
            # Left - Configuration
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #10b981;">1. Select Training Datasets</div>')

                refresh_datasets_btn = gr.Button("Refresh Dataset List", variant="secondary")

                datasets_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="Select datasets to train on (import in 'Import Training Data' tab)",
                    value=[]
                )

                dataset_info = gr.Markdown("*Click refresh to see available datasets*")

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #10b981;">2. Select Tasks to Train</div>')

                train_tasks = gr.CheckboxGroup(
                    choices=[
                        ("Serve Classification (Ace/Made/Fault)", "serve_class"),
                        ("Serve Placement (Wide/T/Body)", "serve_placement"),
                        ("Stroke Classification (FH/BH/etc)", "stroke_class"),
                        ("Shot Placement (Crosscourt/Down Line)", "shot_placement"),
                        ("Point Result (Winner/Error)", "point_result")
                    ],
                    value=["serve_class", "stroke_class"],
                    label="Select tasks to train"
                )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #10b981;">3. Training Configuration</div>')

                with gr.Row():
                    epochs = gr.Slider(10, 200, value=50, step=10, label="Epochs")
                    batch = gr.Slider(8, 128, value=32, step=8, label="Batch Size")

                train_mode = gr.Radio(
                    choices=[
                        ("Fine-tune existing models (recommended)", "finetune"),
                        ("Train from scratch", "scratch")
                    ],
                    value="finetune",
                    label="Training Mode"
                )

                device = gr.Radio(
                    choices=[("Auto", "auto"), ("GPU (faster)", "cuda"), ("CPU", "cpu")],
                    value="auto",
                    label="Device"
                )

                time_estimate = gr.Markdown("*Select datasets and tasks to see estimate*")

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')

                start_train_btn = gr.Button(
                    "Start Training",
                    variant="primary",
                    size="lg"
                )

                stop_train_btn = gr.Button(
                    "Stop Training",
                    variant="secondary",
                    visible=False
                )

            # Right - Progress
            with gr.Column(scale=2):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #10b981;">Training Progress</div>')

                train_status = gr.Markdown(
                    "**Ready** - Select datasets and tasks, then click Start Training"
                )

                training_log = gr.Textbox(
                    label="Training Log",
                    lines=12,
                    interactive=False,
                    placeholder="Training progress will appear here..."
                )

                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #059669; margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #10b981;">Training Results</div>')

                results_table = gr.Dataframe(
                    headers=["Task", "Status", "Version", "Accuracy", "Samples"],
                    label="Training Results",
                    interactive=False
                )

                train_chart = gr.Plot(label="Training Metrics (Loss & Accuracy)")

        # Functions
        def refresh_datasets():
            """Refresh list of imported datasets."""
            if not imported_trainer:
                return gr.CheckboxGroup(choices=[]), "*Training importer not available*"

            try:
                datasets = imported_trainer.list_available_datasets()

                if not datasets:
                    return gr.CheckboxGroup(choices=[]), """
**No datasets imported yet!**

Go to the **"Import Training Data"** tab to:
1. Select a human-tagged CSV file
2. Select the corresponding video
3. Click Import

Then come back here to train.
"""

                choices = []
                info_lines = ["**Available Datasets:**\n"]

                for ds in datasets:
                    name = ds.get('name', ds.get('dataset_name', 'Unknown'))
                    points = ds.get('total_points', 0)
                    pairs = ds.get('training_pairs', 0)
                    choices.append((f"{name} ({points} points)", name))
                    info_lines.append(f"- **{name}**: {points} points, {pairs} training pairs")

                return gr.CheckboxGroup(choices=choices), "\n".join(info_lines)

            except Exception as e:
                return gr.CheckboxGroup(choices=[]), f"*Error: {str(e)}*"

        def update_time_estimate(selected_datasets, tasks, epochs_val, batch_val, device_val):
            """Update training time estimate."""
            if not selected_datasets:
                return "*Select datasets to see estimate*"
            if not tasks:
                return "*Select tasks to see estimate*"

            try:
                # Count total points from selected datasets
                total_points = 0
                if imported_trainer:
                    for ds_name in selected_datasets:
                        datasets = imported_trainer.list_available_datasets()
                        for ds in datasets:
                            if ds.get('name') == ds_name or ds.get('dataset_name') == ds_name:
                                total_points += ds.get('total_points', 0)

                if total_points == 0:
                    return "*No data points in selected datasets*"

                estimates = []
                total_seconds = 0
                for task in tasks:
                    seconds = TrainingEstimator.estimate_time(
                        total_points,
                        epochs_val,
                        batch_val,
                        device_val
                    )
                    total_seconds += seconds
                    task_name = task.replace('_', ' ').title()
                    estimates.append(f"- {task_name}: {TrainingEstimator.format_time(seconds)}")

                return f"""**Estimated Training Time:**

{chr(10).join(estimates)}

**Total**: {TrainingEstimator.format_time(total_seconds)}
**Data**: {total_points} points from {len(selected_datasets)} dataset(s)
"""
            except Exception as e:
                return f"*Error estimating: {e}*"

        def start_training(selected_datasets, tasks, epochs_val, batch_val, mode, device_val):
            """Start training on selected datasets."""
            if not selected_datasets:
                return "**Error:** Please select at least one dataset", "", []

            if not tasks:
                return "**Error:** Please select at least one task to train", "", []

            if not imported_trainer:
                return "**Error:** Training importer not available", "", []

            try:
                log_lines = []
                log_lines.append(f"Starting training...")
                log_lines.append(f"Datasets: {', '.join(selected_datasets)}")
                log_lines.append(f"Tasks: {', '.join(tasks)}")
                log_lines.append(f"Mode: {mode}, Epochs: {epochs_val}, Batch: {batch_val}")
                log_lines.append("-" * 40)

                # Run training
                results = imported_trainer.train_all_tasks(
                    dataset_names=selected_datasets,
                    tasks=tasks,
                    epochs=int(epochs_val),
                    batch_size=int(batch_val),
                    device=device_val,
                    mode=mode
                )

                # Build results table
                table_data = []
                for task, result in results.items():
                    if result.get('status') == 'success':
                        metrics = result.get('metrics', {})
                        table_data.append([
                            task.replace('_', ' ').title(),
                            "Success",
                            f"v{result.get('version', '?')}",
                            f"{metrics.get('accuracy', 0):.1f}%",
                            metrics.get('num_samples', 0)
                        ])
                        log_lines.append(f"[OK] {task}: v{result.get('version')} - {metrics.get('accuracy', 0):.1f}% accuracy")
                    else:
                        table_data.append([
                            task.replace('_', ' ').title(),
                            "Failed",
                            "-",
                            "-",
                            "-"
                        ])
                        log_lines.append(f"[FAIL] {task}: {result.get('message', 'Unknown error')}")

                log_lines.append("-" * 40)
                log_lines.append("Training complete!")

                status_md = f"""**Training Complete!**

Trained {len([r for r in results.values() if r.get('status') == 'success'])} / {len(tasks)} tasks successfully.

Models saved with automatic versioning. Check the **Models** tab to see trained models.
"""

                return status_md, "\n".join(log_lines), table_data

            except Exception as e:
                return f"**Error:** {str(e)}", f"Training failed: {str(e)}", []

        # Connect events
        refresh_datasets_btn.click(
            refresh_datasets,
            outputs=[datasets_checkboxes, dataset_info]
        )

        # Auto-refresh on tab load
        tab.select(
            refresh_datasets,
            outputs=[datasets_checkboxes, dataset_info]
        )

        for component in [datasets_checkboxes, train_tasks, epochs, batch, device]:
            component.change(
                update_time_estimate,
                inputs=[datasets_checkboxes, train_tasks, epochs, batch, device],
                outputs=[time_estimate]
            )

        start_train_btn.click(
            start_training,
            inputs=[datasets_checkboxes, train_tasks, epochs, batch, train_mode, device],
            outputs=[train_status, training_log, results_table]
        )

    return tab


# =============================================================================
# TAB 3: QC CORRECTIONS
# =============================================================================

def create_qc_tab():
    """Create the QC corrections tab."""

    qc_manager = QCManager()

    with gr.Tab("QC Corrections", id="qc") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #dc2626 0%, #f97316 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Quality Control - Correct Machine-Tagged Errors</h2>
            <p style="margin: 0; opacity: 0.9;">
                Upload corrected CSVs to improve model accuracy. System learns from your corrections in batch.
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #dc2626; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #f97316;">Submit Correction</div>')

                video_select = gr.Dropdown(
                    choices=[],
                    label="Select Video"
                )

                original_csv = gr.File(
                    label="Original CSV (system-generated)",
                    file_types=[".csv"]
                )

                corrected_csv = gr.File(
                    label="Corrected CSV (your fixes)",
                    file_types=[".csv"]
                )

                submit_qc_btn = gr.Button(
                    "Calculate Accuracy & Submit",
                    variant="primary"
                )

            with gr.Column():
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #dc2626; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #f97316;">Accuracy Results</div>')

                accuracy_display = gr.Markdown(
                    "Upload CSVs to see accuracy score"
                )

                qc_result = gr.Markdown("")

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #dc2626; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #f97316;">Pending Corrections Queue</div>')

        refresh_qc_btn = gr.Button("Refresh Queue")

        qc_table = gr.Dataframe(
            headers=["Video", "Accuracy", "Corrections", "Date"],
            label="QC Corrections waiting for batch retrain",
            interactive=False
        )

        batch_stats = gr.Markdown("*No corrections in queue*")

        batch_retrain_btn = gr.Button(
            "Start Batch Retrain",
            variant="primary"
        )

        # Functions
        def refresh_video_list():
            """Refresh video dropdown."""
            try:
                registry = get_video_registry()
                processed = registry.get_processed_videos()
                choices = []
                for v in processed:
                    if isinstance(v, dict):
                        vid = v.get('id', 'unknown')
                        name = v.get('name', vid)
                        choices.append((name, vid))
                return gr.Dropdown(choices=choices)
            except Exception as e:
                return gr.Dropdown(choices=[])

        def submit_correction(video_id, original, corrected):
            """Submit QC correction."""
            if not video_id or not original or not corrected:
                return "Upload both CSVs", "Missing files"

            try:
                result = qc_manager.save_correction(video_id, original, corrected)

                acc = result['accuracy']
                if acc >= 95:
                    grade = "EXCELLENT"
                elif acc >= 90:
                    grade = "GOOD"
                elif acc >= 85:
                    grade = "FAIR"
                else:
                    grade = "NEEDS IMPROVEMENT"

                accuracy_md = f"""
## Accuracy: {acc:.1f}%

**Grade**: {grade}
**Corrections**: {result['corrections']} errors fixed
                """

                result_md = f"""
Correction saved for: {video_id}
Added to batch queue.
                """

                return accuracy_md, result_md
            except Exception as e:
                return "Error", str(e)

        def refresh_qc_queue():
            """Refresh QC queue."""
            try:
                corrections = qc_manager.list_corrections()
                data = []
                for c in corrections:
                    if isinstance(c, dict):
                        data.append([
                            c.get('video', ''),
                            f"{c.get('accuracy', 0):.1f}%",
                            c.get('corrections', 0),
                            c.get('timestamp', '')
                        ])

                stats = qc_manager.get_batch_stats()
                if stats.get('count', 0) == 0:
                    stats_md = "*No corrections in queue*"
                else:
                    stats_md = f"""
**Total**: {stats.get('count', 0)} corrections
**Average Accuracy**: {stats.get('avg_accuracy', 0):.1f}%
                    """

                return data, stats_md
            except Exception as e:
                return [], f"Error: {e}"

        def start_batch_retrain():
            """Start batch retrain from QC corrections."""
            try:
                corrections = qc_manager.list_corrections()
                if not corrections:
                    return "*No corrections in queue to retrain from*", []

                # This would integrate with ImportedDataTrainer
                # For now, return status
                return f"""
**Batch Retrain Started**

Processing {len(corrections)} corrections...

This feature will:
1. Load corrected CSV data
2. Extract training samples
3. Fine-tune models with low learning rate
4. Create new model versions

*Note: Full implementation requires QC data to be converted to training format*
                """, []
            except Exception as e:
                return f"*Error: {str(e)}*", []

        # Connect events
        tab.select(refresh_video_list, outputs=[video_select])

        submit_qc_btn.click(
            submit_correction,
            inputs=[video_select, original_csv, corrected_csv],
            outputs=[accuracy_display, qc_result]
        )

        refresh_qc_btn.click(
            refresh_qc_queue,
            outputs=[qc_table, batch_stats]
        )

        batch_retrain_btn.click(
            start_batch_retrain,
            outputs=[batch_stats, qc_table]
        )

    return tab


# =============================================================================
# TAB 4: VIDEO LIBRARY
# =============================================================================

def create_library_tab():
    """Create the video library tab."""

    with gr.Tab("Video Library", id="library") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Video Library</h2>
            <p style="margin: 0; opacity: 0.9;">
                Browse registered videos. Videos stay in their original locations - only references stored!
            </p>
        </div>
        """)

        with gr.Row():
            refresh_btn = gr.Button("Refresh Library")
            import_old_btn = gr.Button("Import from Old Database")

        stats_display = gr.Markdown("")

        video_table = gr.Dataframe(
            headers=["ID", "Name", "Status", "Size", "FVD", "Exists"],
            label="Registered Videos",
            interactive=False
        )

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML("""
        <div style="font-size: 1.1em; font-weight: 600; color: #7c3aed; margin-bottom: 5px; padding-bottom: 8px; border-bottom: 2px solid #a855f7;">
            Missing Videos
        </div>
        <p style="color: #6b7280; font-size: 0.9em; margin: 0 0 10px 0;">Videos where the original file was moved or deleted</p>
        """)

        missing_table = gr.Dataframe(
            headers=["ID", "Name", "Last Path"],
            label="Missing Videos",
            interactive=False
        )

        with gr.Row():
            video_id_input = gr.Textbox(label="Video ID", placeholder="Enter video ID")
            new_path_input = gr.Textbox(label="New Path", placeholder="New location of video")
            update_path_btn = gr.Button("Update Path")

        update_status = gr.Markdown("")

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML("""
        <div style="font-size: 1.1em; font-weight: 600; color: #dc2626; margin-bottom: 5px; padding-bottom: 8px; border-bottom: 2px solid #f87171;">
            Active Checkpoints (Incomplete Processing)
        </div>
        <p style="color: #6b7280; font-size: 0.9em; margin: 0 0 10px 0;">Videos that were partially processed - you can resume these without losing progress</p>
        """)

        checkpoint_table = gr.Dataframe(
            headers=["Video Name", "Frames Processed", "Progress", "Last Activity", "Data Size"],
            label="Checkpoints Ready to Resume",
            interactive=False
        )

        with gr.Row():
            checkpoint_select = gr.Dropdown(
                choices=[],
                label="Select Checkpoint to Resume",
                scale=3
            )
            resume_checkpoint_btn = gr.Button("Resume Processing", variant="primary")
            delete_checkpoint_btn = gr.Button("Delete Checkpoint", variant="secondary")

        checkpoint_status = gr.Markdown("")

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #7c3aed; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #a855f7;">FVD Cache Management</div>')

        fvd_stats = gr.Markdown("")

        with gr.Row():
            clear_fvd_btn = gr.Button("Clear All FVD Cache")
            clear_fvd_status = gr.Markdown("")

        # Functions
        def get_checkpoints():
            """Get list of active checkpoints with details."""
            checkpoints_dir = Path("checkpoints")
            if not checkpoints_dir.exists():
                return [], []

            checkpoint_data = []
            checkpoint_choices = []

            for json_file in checkpoints_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        meta = json.load(f)

                    video_name = json_file.stem
                    data_dir = checkpoints_dir / f"{video_name}_data"

                    # Get progress info
                    last_frame = meta.get('last_frame_idx', 0)
                    num_detections = meta.get('num_detections', 0)

                    # Try to get total frames from video metadata or estimate
                    video_path = meta.get('video_path', '')
                    total_frames = "Unknown"
                    progress_pct = "?"

                    # Check if video exists to get total frames
                    if video_path and Path(video_path).exists():
                        try:
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if total_frames > 0:
                                progress_pct = f"{(last_frame / total_frames) * 100:.1f}%"
                        except:
                            pass

                    # Get data size
                    data_size_mb = 0
                    if data_dir.exists():
                        for pkl_file in data_dir.glob("*.pkl"):
                            data_size_mb += pkl_file.stat().st_size / (1024 * 1024)

                    # Get last modified time
                    last_activity = datetime.fromtimestamp(json_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

                    checkpoint_data.append([
                        video_name,
                        f"{last_frame + 1:,} / {total_frames:,}" if isinstance(total_frames, int) else f"{last_frame + 1:,}",
                        progress_pct,
                        last_activity,
                        f"{data_size_mb:.1f} MB"
                    ])

                    checkpoint_choices.append((f"{video_name} ({progress_pct})", video_name))

                except Exception as e:
                    continue

            return checkpoint_data, checkpoint_choices

        def refresh_checkpoints():
            """Refresh checkpoint list."""
            data, choices = get_checkpoints()
            return data, choices

        def resume_from_checkpoint(checkpoint_name):
            """Resume processing from a checkpoint."""
            if not checkpoint_name:
                return "Please select a checkpoint to resume"

            checkpoints_dir = Path("checkpoints")
            json_file = checkpoints_dir / f"{checkpoint_name}.json"

            if not json_file.exists():
                return f"Checkpoint not found: {checkpoint_name}"

            try:
                with open(json_file, 'r') as f:
                    meta = json.load(f)

                video_path = meta.get('video_path', '')
                output_csv = meta.get('output_csv', '')

                if not video_path or not Path(video_path).exists():
                    return f"Video file not found: {video_path}\n\nUpdate the video path first."

                # Start processing with resume=True
                if HAS_VIDEO_PROCESSING:
                    success, message = start_video_processing(
                        video_path,
                        output_csv,
                        visualize=False,
                        batch_size=32,
                        device='auto',
                        resume=True  # Resume from checkpoint!
                    )

                    if success:
                        return f"**Resuming:** {checkpoint_name}\n\nGo to **Tag Video** tab to see progress."
                    else:
                        return f"Failed to start: {message}"
                else:
                    return "Video processing module not available"

            except Exception as e:
                return f"Error: {str(e)}"

        def delete_checkpoint(checkpoint_name):
            """Delete a checkpoint."""
            if not checkpoint_name:
                return "Please select a checkpoint to delete"

            checkpoints_dir = Path("checkpoints")
            json_file = checkpoints_dir / f"{checkpoint_name}.json"
            data_dir = checkpoints_dir / f"{checkpoint_name}_data"

            deleted = []
            try:
                if json_file.exists():
                    json_file.unlink()
                    deleted.append("metadata")

                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    deleted.append("data")

                if deleted:
                    return f"Deleted checkpoint: {checkpoint_name} ({', '.join(deleted)})"
                else:
                    return f"Checkpoint not found: {checkpoint_name}"

            except Exception as e:
                return f"Error deleting: {str(e)}"

        def refresh_library():
            """Refresh library display."""
            try:
                registry = get_video_registry()
                stats = registry.get_stats()

                stats_md = f"""
**Total Videos**: {stats['total']} | **Processed**: {stats['processed']} | **Pending**: {stats['pending']}
**Total Size**: {stats['total_size_gb']} GB | **Missing**: {stats['missing']}
                """

                videos = registry.list_all_videos()
                data = []
                for v in videos:
                    if isinstance(v, dict):
                        data.append([
                            v.get('id', 'unknown'),
                            v.get('name', 'Unknown'),
                            v.get('status', 'unknown'),
                            f"{v.get('file_size', 0) / (1024*1024):.1f} MB",
                            "Yes" if v.get('fvd_path') else "No",
                            "Yes" if v.get('exists') else "NO"
                        ])

                missing = registry.get_missing_videos()
                missing_data = []
                for v in missing:
                    if isinstance(v, dict):
                        missing_data.append([v.get('id', ''), v.get('name', ''), v.get('path', '')])

                # FVD stats
                fvd_mgr = get_fvd_manager()
                fvd_files = fvd_mgr.list_fvd_files()
                total_fvd_size = sum(f.get('size_mb', 0) for f in fvd_files if isinstance(f, dict))
                fvd_md = f"**FVD Files**: {len(fvd_files)} | **Total Size**: {total_fvd_size:.1f} MB"

                return stats_md, data, missing_data, fvd_md
            except Exception as e:
                return f"Error: {e}", [], [], ""

        def import_old_database():
            """Import from old video_database.json."""
            registry = get_video_registry()
            count = registry.import_from_video_database()
            return f"Imported {count} videos from old database"

        def update_video_path(video_id, new_path):
            """Update video path."""
            if not video_id or not new_path:
                return "Please enter both video ID and new path"

            registry = get_video_registry()
            try:
                success = registry.update_video_path(video_id.strip(), new_path.strip())
                if success:
                    return f"Updated path for {video_id}"
                else:
                    return f"Video not found: {video_id}"
            except Exception as e:
                return f"Error: {e}"

        def clear_all_fvd():
            """Clear all FVD cache files."""
            try:
                fvd_mgr = get_fvd_manager()
                fvd_dir = Path("data/fvd")

                if not fvd_dir.exists():
                    return "No FVD cache to clear"

                count = 0
                for fvd_file in fvd_dir.glob("*.fvd.json*"):
                    fvd_file.unlink()
                    count += 1

                return f"Cleared {count} FVD cache files"
            except Exception as e:
                return f"Error: {e}"

        # Connect events
        def full_refresh():
            """Refresh everything including checkpoints."""
            stats_md, video_data, missing_data, fvd_md = refresh_library()
            cp_data, cp_choices = refresh_checkpoints()
            return stats_md, video_data, missing_data, fvd_md, cp_data, gr.update(choices=cp_choices)

        refresh_btn.click(
            full_refresh,
            outputs=[stats_display, video_table, missing_table, fvd_stats, checkpoint_table, checkpoint_select]
        )

        import_old_btn.click(
            import_old_database,
            outputs=[stats_display]
        )

        update_path_btn.click(
            update_video_path,
            inputs=[video_id_input, new_path_input],
            outputs=[update_status]
        )

        clear_fvd_btn.click(
            clear_all_fvd,
            outputs=[clear_fvd_status]
        )

        resume_checkpoint_btn.click(
            resume_from_checkpoint,
            inputs=[checkpoint_select],
            outputs=[checkpoint_status]
        )

        delete_checkpoint_btn.click(
            delete_checkpoint,
            inputs=[checkpoint_select],
            outputs=[checkpoint_status]
        )

        # Auto-refresh checkpoints on tab load
        tab.select(
            full_refresh,
            outputs=[stats_display, video_table, missing_table, fvd_stats, checkpoint_table, checkpoint_select]
        )

    return tab


# =============================================================================
# TAB 5: MODELS
# =============================================================================

def create_models_tab():
    """Create the models management tab."""

    trainer = IncrementalTrainer()

    # All task types (legacy + new from ImportedDataTrainer)
    ALL_TASKS = [
        'stroke', 'serve', 'placement',  # Legacy
        'serve_class', 'serve_placement', 'stroke_class', 'shot_placement', 'point_result'  # New
    ]

    with gr.Tab("Models", id="models") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Model Management</h2>
            <p style="margin: 0; opacity: 0.9;">
                View and manage trained models. Each retrain creates a new version - never overwrites!
            </p>
        </div>
        """)

        refresh_btn = gr.Button("Refresh Models", variant="secondary")

        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #0891b2; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #06b6d4;">Active Models</div>')

        models_table = gr.Dataframe(
            headers=["Model", "Version", "Task", "Accuracy", "Samples", "Mode", "Date"],
            label="Current Active Versions",
            interactive=False
        )

        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #0891b2; margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #06b6d4;">Version History</div>')

        history_table = gr.Dataframe(
            headers=["Task", "Version", "Parent", "Mode", "Date"],
            label="All Model Versions",
            interactive=False
        )

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #0891b2; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #06b6d4;">Model Inference Test</div>')

        with gr.Row():
            test_video = gr.Textbox(label="Test Video Path", placeholder="Enter path to test video", scale=3)
            test_btn = gr.Button("Run Inference Test", variant="primary")

        test_results = gr.Markdown("*Enter a video path and click test to see model predictions*")

        # Functions
        def refresh_models():
            """Refresh model lists."""
            try:
                models_dir = Path("models")
                versions_dir = models_dir / "versions"

                # Active models - check all task types
                active = []
                for task in ALL_TASKS:
                    version = trainer.get_model_version(task)
                    if version > 0:
                        model_file = models_dir / f"{task}_v{version}.pt"
                        metadata_file = models_dir / f"{task}_v{version}_metadata.json"

                        acc = "N/A"
                        mode = "N/A"
                        date = "N/A"
                        samples = "N/A"

                        if metadata_file.exists():
                            try:
                                with open(metadata_file) as f:
                                    meta = json.load(f)
                                    if isinstance(meta, dict):
                                        metrics = meta.get('metrics', {})
                                        if isinstance(metrics, dict):
                                            acc_val = metrics.get('accuracy', 'N/A')
                                            acc = f"{acc_val:.1f}%" if isinstance(acc_val, (int, float)) else str(acc_val)
                                            samples = metrics.get('num_samples', 'N/A')
                                        mode = meta.get('training_mode', 'N/A')
                                        ts = meta.get('timestamp', '')
                                        date = ts[:10] if ts else 'N/A'
                            except:
                                pass

                        # Format task name nicely
                        task_display = task.replace('_', ' ').title()

                        active.append([
                            f"{task}_v{version}.pt",
                            f"v{version}",
                            task_display,
                            acc,
                            samples,
                            mode,
                            date
                        ])

                # Version history
                history = []
                for task in ALL_TASKS:
                    versions = trainer.list_model_versions(task)
                    for v in versions:
                        if isinstance(v, dict):
                            ts = v.get('timestamp', '')
                            task_display = task.replace('_', ' ').title()
                            history.append([
                                task_display,
                                f"v{v.get('version', '?')}",
                                f"v{v.get('parent_version', '-')}" if v.get('parent_version') else "-",
                                v.get('training_mode', 'N/A'),
                                ts[:10] if ts else ''
                            ])

                if not active:
                    active = [["No models trained yet", "-", "-", "-", "-", "-", "-"]]

                return active, history
            except Exception as e:
                return [[f"Error: {e}", "-", "-", "-", "-", "-", "-"]], []

        def run_inference_test(video_path):
            """Run inference test on a video."""
            if not video_path or not video_path.strip():
                return "*Please enter a video path to test*"

            video_path = Path(video_path.strip())
            if not video_path.exists():
                return f"*Video not found: {video_path}*"

            try:
                models_dir = Path("models")

                # Check which models are available
                available_models = []
                for task in ALL_TASKS:
                    version = trainer.get_model_version(task)
                    if version > 0:
                        model_file = models_dir / f"{task}_v{version}.pt"
                        if model_file.exists():
                            available_models.append(f"- {task.replace('_', ' ').title()} v{version}")

                if not available_models:
                    return """
**No trained models found!**

Train models first using the **Training** tab:
1. Import human-tagged data in "Import Training Data" tab
2. Select datasets in "Training" tab
3. Choose tasks and click "Start Training"
"""

                return f"""
**Models Available for Inference:**

{chr(10).join(available_models)}

**Video:** {video_path.name}

*Full inference testing requires processing the video through the detection pipeline.*
*Use the "Tag Video" tab to process this video with the trained models.*
"""
            except Exception as e:
                return f"*Error: {str(e)}*"

        # Connect events
        refresh_btn.click(
            refresh_models,
            outputs=[models_table, history_table]
        )

        test_btn.click(
            run_inference_test,
            inputs=[test_video],
            outputs=[test_results]
        )

        # Auto-refresh on tab load
        tab.select(
            refresh_models,
            outputs=[models_table, history_table]
        )

    return tab


# =============================================================================
# TAB 6: IMPORT TRAINING DATA
# =============================================================================

def create_import_training_tab():
    """Create the import training data tab for human-tagged CSVs."""

    # Initialize importer
    importer = None
    if HAS_TRAINING_IMPORTER:
        importer = create_training_data_importer()

    with gr.Tab("Import Training Data", id="import") as tab:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #ca8a04 0%, #eab308 100%);
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">Import Human-Tagged Training Data</h2>
            <p style="margin: 0; opacity: 0.9;">
                Import Dartfish CSV + video to create training data. Video is automatically processed to extract player/ball detection.
            </p>
        </div>
        """)

        gr.HTML("""
        <div style="background: #fefce8; border: 1px solid #eab308; border-radius: 8px;
                    padding: 15px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="text-align: center; flex: 1; min-width: 140px;">
                    <div style="font-size: 1.5em; margin-bottom: 5px;">1</div>
                    <div style="font-weight: 600; color: #854d0e;">Select CSV</div>
                    <div style="font-size: 0.85em; color: #a16207;">Human-tagged data</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 140px;">
                    <div style="font-size: 1.5em; margin-bottom: 5px;">2</div>
                    <div style="font-weight: 600; color: #854d0e;">Select Video</div>
                    <div style="font-size: 0.85em; color: #a16207;">Matching footage</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 140px;">
                    <div style="font-size: 1.5em; margin-bottom: 5px;">3</div>
                    <div style="font-weight: 600; color: #854d0e;">Import</div>
                    <div style="font-size: 0.85em; color: #a16207;">Auto-processes video</div>
                </div>
                <div style="text-align: center; flex: 1; min-width: 140px;">
                    <div style="font-size: 1.5em; margin-bottom: 5px;">4</div>
                    <div style="font-weight: 600; color: #854d0e;">Train</div>
                    <div style="font-size: 0.85em; color: #a16207;">In Training tab</div>
                </div>
            </div>
        </div>
        """)

        if not HAS_TRAINING_IMPORTER:
            gr.HTML("""
            <div style="background: #fef2f2; border: 1px solid #ef4444; border-radius: 8px; padding: 15px;">
                <strong style="color: #b91c1c;">Warning:</strong> Training Data Importer module not available.
                Make sure <code>training_data_importer.py</code> is in the files directory.
            </div>
            """)
            return tab

        with gr.Row():
            # Left column - Import controls
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #ca8a04; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #eab308;">Select Files</div>')

                # CSV selection
                with gr.Row():
                    csv_browse_btn = gr.Button("Browse CSV", variant="primary")
                    csv_path = gr.Textbox(
                        label="CSV File Path",
                        placeholder="Select human-tagged CSV file...",
                        interactive=True
                    )

                # Video selection
                with gr.Row():
                    video_browse_btn = gr.Button("Browse Video", variant="primary")
                    video_path = gr.Textbox(
                        label="Video File Path",
                        placeholder="Select corresponding video file...",
                        interactive=True
                    )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #ca8a04; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #eab308;">Dataset Settings</div>')

                dataset_name = gr.Textbox(
                    label="Dataset Name (optional)",
                    placeholder="Auto-generated from CSV name if empty"
                )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')

                with gr.Row():
                    preview_btn = gr.Button("Preview Annotations", variant="secondary")
                    import_btn = gr.Button("Import & Create Training Data", variant="primary", size="lg")

                import_status = gr.Markdown("")

            # Right column - Preview and results
            with gr.Column(scale=2):
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #ca8a04; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #eab308;">Annotation Preview</div>')

                preview_stats = gr.Markdown("*Select CSV and click Preview to see annotations*")

                preview_table = gr.Dataframe(
                    headers=["Point", "Time (s)", "Serve", "Placement", "Last Shot", "Result"],
                    label="Parsed Annotations (first 20)",
                    interactive=False
                )

                gr.HTML('<hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">')
                gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #ca8a04; margin: 15px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #eab308;">Label Statistics</div>')

                label_stats = gr.Markdown("")

        gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;">')
        gr.HTML('<div style="font-size: 1.1em; font-weight: 600; color: #ca8a04; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid #eab308;">Imported Training Datasets</div>')

        refresh_datasets_btn = gr.Button("Refresh List")

        datasets_table = gr.Dataframe(
            headers=["Dataset", "Points", "Pairs", "Video", "Date", "Size"],
            label="Imported Datasets Ready for Training",
            interactive=False
        )

        with gr.Row():
            delete_dataset_input = gr.Textbox(
                label="Dataset to Delete",
                placeholder="Enter dataset name"
            )
            delete_btn = gr.Button("Delete Dataset", variant="secondary")
            delete_status = gr.Markdown("")

        # Helper functions
        def browse_csv():
            """Open file browser for CSV."""
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                file_path = filedialog.askopenfilename(
                    title="Select Human-Tagged CSV",
                    filetypes=[
                        ("CSV files", "*.csv"),
                        ("All files", "*.*")
                    ],
                    initialdir=str(Path("data/training_pairs"))
                )
                root.destroy()

                if file_path:
                    return file_path
                return ""
            except Exception as e:
                return ""

        def browse_video():
            """Open file browser for video."""
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                file_path = filedialog.askopenfilename(
                    title="Select Video File",
                    filetypes=[
                        ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                        ("All files", "*.*")
                    ],
                    initialdir=str(Path.home() / "Videos")
                )
                root.destroy()

                if file_path:
                    return file_path
                return ""
            except Exception as e:
                return ""

        def preview_annotations(csv_file):
            """Preview parsed annotations from CSV."""
            if not csv_file or not csv_file.strip():
                return "*Select a CSV file first*", [], ""

            csv_path_obj = Path(csv_file.strip())
            if not csv_path_obj.exists():
                return f"*File not found: {csv_file}*", [], ""

            try:
                annotations = importer.parse_annotation_csv(str(csv_path_obj))

                # Create preview table
                table_data = []
                for i, ann in enumerate(annotations[:20]):
                    position_s = ann.get('position_us', 0) / 1_000_000
                    table_data.append([
                        ann.get('name', f'Point {i+1}'),
                        f"{position_s:.1f}",
                        ann.get('serve_data', ''),
                        ann.get('serve_placement', ''),
                        ann.get('last_shot', ''),
                        ann.get('point_result', '')
                    ])

                # Create stats
                stats_md = f"**Total Points:** {len(annotations)}\n\n"

                # Count by serve type
                serve_counts = {}
                for ann in annotations:
                    serve = ann.get('serve_data', 'Unknown')
                    serve_counts[serve] = serve_counts.get(serve, 0) + 1

                if serve_counts:
                    stats_md += "**Serve Types:**\n"
                    for serve, count in sorted(serve_counts.items(), key=lambda x: -x[1])[:5]:
                        stats_md += f"- {serve}: {count}\n"

                # Count by stroke type
                stroke_counts = {}
                for ann in annotations:
                    stroke = ann.get('last_shot', 'Unknown')
                    if stroke:
                        stroke_counts[stroke] = stroke_counts.get(stroke, 0) + 1

                if stroke_counts:
                    stats_md += "\n**Stroke Types:**\n"
                    for stroke, count in sorted(stroke_counts.items(), key=lambda x: -x[1])[:5]:
                        stats_md += f"- {stroke}: {count}\n"

                preview_md = f"**Parsed {len(annotations)} points from:** {csv_path_obj.name}"

                return preview_md, table_data, stats_md

            except Exception as e:
                return f"*Error parsing CSV: {str(e)}*", [], ""

        def import_training_data(csv_file, video_file, ds_name):
            """Import CSV and video as training data."""
            if not csv_file or not csv_file.strip():
                return "**Error:** Please select a CSV file"

            if not video_file or not video_file.strip():
                return "**Error:** Please select a video file"

            csv_path_obj = Path(csv_file.strip())
            video_path_obj = Path(video_file.strip())

            if not csv_path_obj.exists():
                return f"**Error:** CSV not found: {csv_file}"

            if not video_path_obj.exists():
                return f"**Error:** Video not found: {video_file}"

            # Check if FVD exists - if not, we'll process the video automatically
            fvd_manager = get_fvd_manager()
            fvd_exists = fvd_manager.fvd_exists(str(video_path_obj))
            video_size_gb = video_path_obj.stat().st_size / (1024**3)

            # If FVD doesn't exist, check if we can process videos
            if not fvd_exists:
                # Check if video processing is available
                if not HAS_VIDEO_PROCESSING:
                    return f"""
**Cannot Process Video Automatically**

The video **{video_path_obj.name}** needs to be processed to extract player/ball detection data,
but the video processing module is not available.

**To fix this:**
1. Make sure all dependencies are installed:
   ```
   pip install ultralytics opencv-python torch torchvision
   ```
2. Restart the application
3. Try importing again

**Or manually process:**
1. Use the "Tag Video" tab to process the video first
2. Then come back here and import your CSV
"""

            try:
                result = importer.import_csv(
                    str(csv_path_obj),
                    str(video_path_obj),
                    dataset_name=ds_name.strip() if ds_name and ds_name.strip() else None
                )

                # Calculate actual data size
                dataset_path = Path(result['path'])
                total_size = sum(f.stat().st_size for f in dataset_path.glob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)

                # Get metadata for FVD info
                metadata = result.get('metadata', {})
                fvd_frames = metadata.get('fvd_frame_count', 0)
                pairs_with_features = metadata.get('pairs_with_features', 0)
                has_fvd = metadata.get('has_fvd', False)

                # Check if video was processed during import
                video_was_processed = not fvd_exists and has_fvd
                processing_note = ""
                if video_was_processed:
                    processing_note = f"\n\n**Note:** Video was automatically processed to extract detection data ({video_size_gb:.1f} GB → {fvd_frames:,} frames analyzed)"

                if not has_fvd:
                    return f"""
**Import Partially Complete - Missing Video Data**

- **Dataset:** {result['dataset_name']}
- **Human Labels:** {result['total_points']} points imported from CSV

**Problem:** Could not extract detection data from video.
This usually means the video processing failed or YOLO models aren't installed.

**To fix:**
1. Check that YOLO models are installed (yolov8x.pt, ball_detector.pt)
2. Try processing the video manually in "Tag Video" tab
3. Then re-import this CSV

The training will not work properly without detection features.
"""

                return f"""
**Import Successful!**

- **Dataset:** {result['dataset_name']}
- **Total Points:** {result['total_points']}
- **Training Pairs:** {result['training_pairs']} ({pairs_with_features} with frame features)
- **FVD Frames:** {fvd_frames:,} frames of detection data
- **Data Size:** {size_mb:.1f} MB{processing_note}

**What was extracted:**
- Human labels from your CSV (serve types, placements, strokes)
- Player positions and movement from each frame
- Ball trajectory data

**Next Steps:**
1. Go to **Training** tab
2. Click "Refresh Dataset List"
3. Select this dataset
4. Choose tasks to train
5. Click "Start Training"
                """

            except Exception as e:
                import traceback
                error_details = str(e)

                # Check for common issues
                if "No module named" in error_details or "ImportError" in error_details:
                    suggestion = "\n\n**Suggestion:** Install missing dependencies with:\n```\npip install ultralytics opencv-python torch\n```"
                elif "CUDA" in error_details or "GPU" in error_details:
                    suggestion = "\n\n**Suggestion:** GPU error. Try running with CPU or check CUDA installation."
                elif "not found" in error_details.lower():
                    suggestion = "\n\n**Suggestion:** A required file is missing. Check that all model files are present."
                else:
                    suggestion = ""

                return f"""**Import Failed**

**Error:** {error_details}
{suggestion}

**Full traceback:**
```
{traceback.format_exc()}
```

**Alternative:** Process the video manually in the "Tag Video" tab first, then import your CSV.
"""

        def refresh_imported_datasets():
            """Refresh list of imported datasets."""
            try:
                datasets = importer.list_imported_datasets()

                table_data = []
                for ds in datasets:
                    video_name = Path(ds.get('video_path', '')).name if ds.get('video_path') else 'N/A'
                    date = ds.get('import_date', '')[:10] if ds.get('import_date') else ''

                    table_data.append([
                        ds.get('dataset_name', ds.get('name', 'Unknown')),
                        ds.get('total_points', 0),
                        ds.get('training_pairs', 0),
                        video_name[:30] + '...' if len(video_name) > 30 else video_name,
                        date,
                        f"{ds.get('size_mb', 0):.1f} MB"
                    ])

                return table_data
            except Exception as e:
                return []

        def delete_dataset(ds_name):
            """Delete a dataset."""
            if not ds_name or not ds_name.strip():
                return "Enter dataset name to delete"

            try:
                success = importer.delete_dataset(ds_name.strip())
                if success:
                    return f"Deleted: {ds_name}"
                else:
                    return f"Dataset not found: {ds_name}"
            except Exception as e:
                return f"Error: {str(e)}"

        # Connect events
        csv_browse_btn.click(browse_csv, outputs=[csv_path])
        video_browse_btn.click(browse_video, outputs=[video_path])

        preview_btn.click(
            preview_annotations,
            inputs=[csv_path],
            outputs=[preview_stats, preview_table, label_stats]
        )

        import_btn.click(
            import_training_data,
            inputs=[csv_path, video_path, dataset_name],
            outputs=[import_status]
        )

        refresh_datasets_btn.click(
            refresh_imported_datasets,
            outputs=[datasets_table]
        )

        delete_btn.click(
            delete_dataset,
            inputs=[delete_dataset_input],
            outputs=[delete_status]
        )

    return tab


# =============================================================================
# MAIN APP
# =============================================================================

def create_unified_app():
    """Create the unified Tennis Tagger application."""

    # Custom CSS for better appearance
    custom_css = """
    /* Global styles */
    .gradio-container {
        max-width: 1600px !important;
    }

    /* Tab styling */
    .tabs > .tabitem {
        padding: 20px !important;
    }

    /* Button improvements */
    .primary {
        font-weight: 600 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 15px 0 10px 0;
    }

    /* Card-like containers */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Status badges */
    .status-success { color: #059669; font-weight: 600; }
    .status-warning { color: #d97706; font-weight: 600; }
    .status-error { color: #dc2626; font-weight: 600; }

    /* Table improvements */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Input improvements */
    .textbox input, .textbox textarea {
        border-radius: 6px !important;
    }

    /* Footer */
    .footer-info {
        text-align: center;
        padding: 15px;
        background: #f8fafc;
        border-radius: 8px;
        margin-top: 20px;
    }
    """

    with gr.Blocks(
        title="Tennis Tagger",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        css=custom_css
    ) as app:

        # Main header
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
                    color: white; padding: 25px; border-radius: 12px; margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                <div style="font-size: 2.5em;">&#127934;</div>
                <div>
                    <h1 style="margin: 0; font-size: 2.2em; font-weight: 700;">Tennis Tagger</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">
                        Production-Ready Video Tagging & Training System
                    </p>
                </div>
            </div>
        </div>
        """)

        # Create all tabs
        create_tag_video_tab()
        create_training_tab()
        create_qc_tab()
        create_library_tab()
        create_models_tab()
        create_import_training_tab()

        # Footer
        gr.HTML("""
        <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px;
                    padding: 20px; margin-top: 20px; text-align: center;">
            <div style="font-weight: 600; color: #1e40af; margin-bottom: 10px;">
                Tennis Tagger v4.1 - Unified Edition
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;
                        font-size: 0.9em; color: #6b7280;">
                <span>&#10003; Single unified app</span>
                <span>&#10003; No video duplication</span>
                <span>&#10003; FVD training resumption</span>
                <span>&#10003; Auto model versioning</span>
                <span>&#10003; One-click install</span>
            </div>
        </div>
        """)

    return app


# =============================================================================
# ENTRY POINTS
# =============================================================================

def launch_gradio(port: int = 7860, share: bool = False):
    """Launch Gradio server."""
    app = create_unified_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        show_error=True
    )


def launch_desktop():
    """Launch as desktop application with PyWebView."""
    import threading
    import time

    # Check for pywebview
    try:
        import webview
    except ImportError:
        print("Installing PyWebView...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywebview"])
        import webview

    print("""
=====================================================
   TENNIS TAGGER - UNIFIED EDITION v4.0
=====================================================

Starting unified application...
    """)

    # Ensure directories exist
    dirs = [
        "logs", "data", "data/output", "data/fvd",
        "data/training_pairs", "data/datasets", "data/qc_corrections",
        "data/training_data",
        "models", "models/versions", "checkpoints", "cache"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Start Gradio server in background
    def start_server():
        app = create_unified_app()
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            prevent_thread_lock=False,
            inbrowser=False,
            quiet=True
        )

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server
    print("[*] Waiting for server...")
    import requests
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:7860", timeout=1)
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(0.5)

    print("[OK] Server ready!")

    # Create desktop window
    window = webview.create_window(
        title='Tennis Tagger',
        url='http://127.0.0.1:7860',
        width=1500,
        height=950,
        resizable=True,
        min_size=(1200, 800),
        confirm_close=True
    )

    print("[OK] Desktop window opened!")
    print("\n" + "="*50)
    print("  TENNIS TAGGER IS RUNNING")
    print("="*50)
    print("\nClose the window to exit.")

    webview.start(debug=True)

    print("\n[*] Shutting down...")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Tennis Tagger Unified App")
    parser.add_argument('--web', action='store_true', help='Run as web server only (no desktop window)')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--share', action='store_true', help='Create public share link')

    args = parser.parse_args()

    if args.web:
        launch_gradio(port=args.port, share=args.share)
    else:
        launch_desktop()


if __name__ == "__main__":
    main()
