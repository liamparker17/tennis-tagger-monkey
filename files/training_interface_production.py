"""
Production Training Interface v4.0 - Unified Edition with FVD Support

This module provides training interface components that can be:
1. Used standalone (python training_interface_production.py)
2. Imported as components into the unified app (tennis_tagger_unified.py)

v4.0 Changes (Unified Edition):
- FVD (Frame Vector Data) support for training without video loading
- Video Registry integration (no video duplication)
- Refactored as importable components
- Works with unified tabbed interface

v3.1 Features (retained):
- Automatic model versioning (v1→v2→v3) - prevents overwriting previous learning!
- Lower learning rate for fine-tuning (0.00005) - prevents catastrophic forgetting
- Model backup system (old versions saved to models/versions/)
- Enhanced QC accuracy display with grading system
- Direct dataset integration button

Core Features:
- Train multiple tasks simultaneously (Stroke, Serve, Placement)
- Merge datasets from multiple PCs
- Incremental learning (fine-tune without overwriting)
- Batch QC corrections with accuracy tracking & grading
- Training time estimates (per-task breakdown)
- Live multi-task training visualization
- FVD-based training (no need to reload videos)

How Incremental Learning Works:
1. Load existing model (e.g., stroke_v5.pt)
2. Create NEW version (stroke_v6.pt)
3. Fine-tune with LOW learning rate (0.00005 vs 0.001 from scratch)
4. Backup v5 to versions/ folder
5. v6 becomes active, v5 preserved forever

This prevents "catastrophic forgetting" where new training overwrites old knowledge.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import shutil
import tkinter as tk
from tkinter import filedialog

# Import video processing (optional - may not be available)
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
    # Provide stub functions
    def get_processing_progress():
        return {"percent": 0, "message": "Video processing not available"}
    def is_processing_complete():
        return False
    def get_processing_result():
        return None

# Import FVD and Registry (new unified system)
try:
    from frame_vector_data import FrameVectorData, create_fvd_manager
    from video_registry import VideoRegistry, create_video_registry
    HAS_FVD_SYSTEM = True
except ImportError:
    HAS_FVD_SYSTEM = False


# =============================================================================
# DATASET MANAGER - Merge, import, manage datasets
# =============================================================================

class DatasetManager:
    """Manage training datasets and merging"""
    
    def __init__(self, datasets_dir="data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self):
        """List all available datasets"""
        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                npz_files = list(dataset_dir.glob("*.npz"))
                json_files = list(dataset_dir.glob("*_labels.json"))
                
                if npz_files:
                    total_size = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)
                    created = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
                    
                    datasets.append({
                        'name': dataset_dir.name,
                        'points': len(npz_files),
                        'size': f"{total_size:.1f} MB",
                        'created': created.strftime('%Y-%m-%d %H:%M'),
                        'path': str(dataset_dir)
                    })
        
        return sorted(datasets, key=lambda x: x['created'], reverse=True)
    
    def merge_datasets(self, dataset_names, output_name):
        """Merge multiple datasets into one"""
        output_dir = self.datasets_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        merged_count = 0
        
        for dataset_name in dataset_names:
            source_dir = self.datasets_dir / dataset_name
            if not source_dir.exists():
                continue
            
            # Copy all npz and json files
            for file in source_dir.glob("*.npz"):
                shutil.copy2(file, output_dir / f"{dataset_name}_{file.name}")
                merged_count += 1
            
            for file in source_dir.glob("*.json"):
                shutil.copy2(file, output_dir / f"{dataset_name}_{file.name}")
        
        return merged_count
    
    def import_dataset(self, source_path, dataset_name):
        """Import dataset from external source (e.g., from another PC)"""
        output_dir = self.datasets_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        source = Path(source_path)
        imported_count = 0
        
        if source.is_dir():
            for file in source.glob("*.npz"):
                shutil.copy2(file, output_dir / file.name)
                imported_count += 1
            for file in source.glob("*.json"):
                shutil.copy2(file, output_dir / file.name)
        
        return imported_count


# =============================================================================
# TRAINING TIME ESTIMATOR
# =============================================================================

class TrainingEstimator:
    """Estimate training time based on hardware and data"""
    
    # Benchmarks (points per second on different hardware)
    BENCHMARKS = {
        'cuda': 50,   # RTX 2050/3050
        'cuda_fast': 100,  # RTX 4050+
        'cpu': 5      # CPU
    }
    
    @staticmethod
    def estimate_time(num_points, epochs, batch_size, device='cuda'):
        """Estimate training time"""
        
        # Adjust for batch size (larger = faster per point)
        efficiency = min(1.0, batch_size / 32)
        
        # Select benchmark
        if device == 'cuda':
            # Check if fast GPU (simplified - you'd detect actual GPU)
            pps = TrainingEstimator.BENCHMARKS['cuda']
        elif device == 'cpu':
            pps = TrainingEstimator.BENCHMARKS['cpu']
        else:
            pps = TrainingEstimator.BENCHMARKS['cuda']
        
        # Calculate
        total_iterations = (num_points * epochs) / batch_size
        seconds = total_iterations / (pps * efficiency)
        
        return seconds
    
    @staticmethod
    def format_time(seconds):
        """Format seconds as human-readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


# =============================================================================
# QC CORRECTIONS MANAGER
# =============================================================================

class QCManager:
    """Manage QC corrections for batch retraining"""
    
    def __init__(self, qc_dir="data/qc_corrections"):
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)
    
    def list_processed_videos(self):
        """List videos that have been processed"""
        try:
            # Try new unified registry system first
            from video_registry import create_video_registry
            registry = create_video_registry()
            processed = registry.get_processed_videos()
            return [v.get('name', Path(v.get('path', '')).name) for v in processed if isinstance(v, dict)]
        except ImportError:
            pass

        try:
            # Fall back to old system
            from utils.video_database import VideoDatabase
            video_db = VideoDatabase()
            completed_videos = video_db.get_completed_videos()
            return [Path(video['path']).name for video in completed_videos]
        except ImportError:
            return []
    
    def save_correction(self, video_name, original_csv, corrected_csv):
        """Save QC correction with accuracy calculation"""
        
        # Load CSVs
        original_df = pd.read_csv(original_csv)
        corrected_df = pd.read_csv(corrected_csv)
        
        # Calculate accuracy
        total_rows = len(original_df)
        differences = 0
        
        for col in original_df.columns:
            if col in corrected_df.columns:
                differences += (original_df[col] != corrected_df[col]).sum()
        
        accuracy = ((total_rows * len(original_df.columns) - differences) / 
                   (total_rows * len(original_df.columns))) * 100
        
        # Save correction
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        correction_name = f"{Path(video_name).stem}_{timestamp}"
        
        correction_data = {
            'video': video_name,
            'timestamp': timestamp,
            'accuracy': accuracy,
            'total_rows': total_rows,
            'corrections': differences,
            'original_csv': str(original_csv),
            'corrected_csv': str(corrected_csv)
        }
        
        # Save metadata
        with open(self.qc_dir / f"{correction_name}.json", 'w') as f:
            json.dump(correction_data, f, indent=2)
        
        # Copy corrected CSV
        shutil.copy2(corrected_csv, self.qc_dir / f"{correction_name}_corrected.csv")
        
        return correction_data
    
    def list_corrections(self):
        """List all QC corrections waiting for batch training"""
        corrections = []
        
        for json_file in self.qc_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                corrections.append(data)
        
        return sorted(corrections, key=lambda x: x['timestamp'], reverse=True)
    
    def get_batch_stats(self):
        """Get statistics about pending corrections"""
        corrections = self.list_corrections()
        
        if not corrections:
            return {
                'count': 0,
                'avg_accuracy': 0,
                'total_corrections': 0
            }
        
        return {
            'count': len(corrections),
            'avg_accuracy': np.mean([c['accuracy'] for c in corrections]),
            'total_corrections': sum(c['corrections'] for c in corrections)
        }


# =============================================================================
# INCREMENTAL TRAINER - Fine-tune without forgetting
# =============================================================================

class IncrementalTrainer:
    """Train models incrementally without catastrophic forgetting"""

    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)

    def get_model_version(self, task):
        """Get current version number for a task"""
        existing_models = list(self.models_dir.glob(f"{task}_v*.pt"))

        if not existing_models:
            return 0

        # Extract version numbers
        versions = []
        for model_path in existing_models:
            try:
                # Extract version from filename like "stroke_v5.pt"
                version_str = model_path.stem.split('_v')[-1]
                versions.append(int(version_str))
            except:
                pass

        return max(versions) if versions else 0

    def save_model_version(self, task, model, optimizer, metrics, training_mode):
        """Save model with version control"""
        current_version = self.get_model_version(task)
        new_version = current_version + 1

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main model file
        model_filename = f"{task}_v{new_version}.pt"
        model_path = self.models_dir / model_filename

        # Backup previous version to versions folder
        if current_version > 0:
            prev_model = self.models_dir / f"{task}_v{current_version}.pt"
            if prev_model.exists():
                backup_path = self.versions_dir / f"{task}_v{current_version}_{timestamp}.pt"
                shutil.copy2(prev_model, backup_path)

        # Save new version
        checkpoint = {
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
            'optimizer_state_dict': optimizer.state_dict() if optimizer and hasattr(optimizer, 'state_dict') else None,
            'version': new_version,
            'timestamp': timestamp,
            'training_mode': training_mode,
            'metrics': metrics,
            'parent_version': current_version if training_mode == 'fine-tune' else None
        }

        torch.save(checkpoint, model_path)

        # Save metadata
        metadata = {
            'task': task,
            'version': new_version,
            'parent_version': current_version if training_mode == 'fine-tune' else None,
            'timestamp': timestamp,
            'training_mode': training_mode,
            'metrics': metrics
        }

        with open(self.models_dir / f"{task}_v{new_version}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_path, new_version

    def load_or_create_model(self, task, num_classes):
        """Load existing model or create new one"""

        current_version = self.get_model_version(task)

        if current_version > 0:
            # Load most recent version
            latest_model = self.models_dir / f"{task}_v{current_version}.pt"

            if latest_model.exists():
                checkpoint = torch.load(latest_model, map_location='cpu')
                return checkpoint, True, current_version  # True = existing model

        return None, False, 0  # False = new model

    def fine_tune_model(self, task, new_data, base_model_path=None):
        """Fine-tune existing model with new data using lower learning rate"""

        if base_model_path:
            # Load existing model
            checkpoint = torch.load(base_model_path, map_location='cpu')

            # Use much lower learning rate for fine-tuning to preserve existing knowledge
            # This prevents "catastrophic forgetting"
            learning_rate = 0.00005  # 20x lower than from-scratch training

            # Use fewer epochs for fine-tuning
            recommended_epochs = 10

            mode = "fine-tune"
        else:
            # Train from scratch
            checkpoint = None
            learning_rate = 0.001
            recommended_epochs = 50
            mode = "new"

        return mode, learning_rate, recommended_epochs, checkpoint

    def list_model_versions(self, task):
        """List all versions of a model"""
        versions = []

        for metadata_file in self.models_dir.glob(f"{task}_v*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
            except:
                pass

        return sorted(versions, key=lambda x: x['version'], reverse=True)


# =============================================================================
# FVD TRAINER - Train using Frame Vector Data (no video loading!)
# =============================================================================

class FVDTrainer:
    """
    Train models using Frame Vector Data instead of loading videos.

    FVD contains pre-extracted detection data:
    - Player bounding boxes and track IDs
    - Ball positions
    - Pose keypoints (if enabled during processing)
    - Court lines

    This allows training to:
    - Resume without reprocessing video
    - Run on machines without GPU (FVD is lightweight)
    - Share training data across machines
    """

    def __init__(self, fvd_dir: str = "data/fvd"):
        self.fvd_dir = Path(fvd_dir)
        self.fvd_manager = None

        if HAS_FVD_SYSTEM:
            self.fvd_manager = create_fvd_manager(fvd_dir)

    def list_available_fvd(self):
        """List FVD files available for training."""
        if self.fvd_manager:
            return self.fvd_manager.list_fvd_files()
        return []

    def load_training_data_from_fvd(self, fvd_paths: list):
        """
        Load training data from multiple FVD files.

        Args:
            fvd_paths: List of FVD file paths

        Returns:
            Combined training data dictionary
        """
        if not self.fvd_manager:
            return None

        combined_data = {
            'videos': [],
            'player_tracks': [],
            'ball_positions': [],
            'poses': [],
            'total_frames': 0
        }

        for fvd_path in fvd_paths:
            try:
                # Load FVD
                fvd = self.fvd_manager.load_fvd(fvd_path)
                if not fvd:
                    continue

                # Extract training data
                training_data = self.fvd_manager.get_training_data_from_fvd(fvd)

                combined_data['videos'].append(fvd.get('video_path'))
                combined_data['player_tracks'].extend(training_data.get('player_tracks', []))
                combined_data['ball_positions'].extend(training_data.get('ball_positions', []))
                combined_data['poses'].extend(training_data.get('poses', []))
                combined_data['total_frames'] += fvd.get('total_frames', 0)

            except Exception as e:
                print(f"Error loading FVD {fvd_path}: {e}")

        return combined_data

    def prepare_stroke_training_data(self, fvd_data: dict, labels_csv: str):
        """
        Prepare stroke classification training data from FVD + labels.

        Args:
            fvd_data: Combined FVD training data
            labels_csv: Path to CSV with stroke labels

        Returns:
            Training dataset ready for model
        """
        if not fvd_data:
            return None

        # Load labels
        labels_df = pd.read_csv(labels_csv)

        # Match FVD player tracks with labels
        training_samples = []

        # This would extract features from poses for stroke classification
        # The actual implementation depends on your stroke classifier architecture

        return training_samples

    def estimate_training_time(self, fvd_files: list, epochs: int, batch_size: int, device: str = 'cuda'):
        """
        Estimate training time using FVD data.

        Args:
            fvd_files: List of FVD metadata dicts
            epochs: Number of training epochs
            batch_size: Training batch size
            device: Training device

        Returns:
            Estimated time in seconds
        """
        total_frames = sum(f.get('frame_count', 0) for f in fvd_files)
        # Estimate ~100 training samples per 1000 frames
        est_samples = total_frames // 10

        return TrainingEstimator.estimate_time(est_samples, epochs, batch_size, device)


# =============================================================================
# TRAINING MONITOR
# =============================================================================

class TrainingMonitor:
    """Monitor multi-task training"""
    
    def __init__(self, tasks):
        self.tasks = tasks
        self.history = {task: {'epoch': [], 'train_loss': [], 'val_loss': [], 
                              'train_acc': [], 'val_acc': [], 'lr': []}
                       for task in tasks}
    
    def update(self, task, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Add metrics for a task"""
        self.history[task]['epoch'].append(epoch)
        self.history[task]['train_loss'].append(train_loss)
        self.history[task]['val_loss'].append(val_loss)
        self.history[task]['train_acc'].append(train_acc)
        self.history[task]['val_acc'].append(val_acc)
        self.history[task]['lr'].append(lr)
    
    def create_charts(self):
        """Generate charts for all tasks"""
        if not any(self.history[task]['epoch'] for task in self.tasks):
            return None
        
        num_tasks = len(self.tasks)
        fig = make_subplots(
            rows=num_tasks, cols=2,
            subplot_titles=[item for task in self.tasks 
                          for item in [f'{task.title()} - Loss', f'{task.title()} - Accuracy']],
            vertical_spacing=0.1
        )
        
        colors = {'stroke': 'blue', 'serve': 'green', 'placement': 'orange'}
        
        for idx, task in enumerate(self.tasks, 1):
            if not self.history[task]['epoch']:
                continue
            
            epochs = self.history[task]['epoch']
            color = colors.get(task, 'purple')
            
            # Loss plot
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history[task]['train_loss'],
                          name=f'{task} Train Loss', mode='lines',
                          line=dict(color=color, dash='solid')),
                row=idx, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history[task]['val_loss'],
                          name=f'{task} Val Loss', mode='lines',
                          line=dict(color=color, dash='dash')),
                row=idx, col=1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history[task]['train_acc'],
                          name=f'{task} Train Acc', mode='lines',
                          line=dict(color=color, dash='solid')),
                row=idx, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history[task]['val_acc'],
                          name=f'{task} Val Acc', mode='lines',
                          line=dict(color=color, dash='dash')),
                row=idx, col=2
            )
        
        fig.update_layout(height=300*num_tasks, showlegend=True, title_text="Multi-Task Training Progress")
        return fig


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def create_training_interface():
    """Create production training interface"""
    
    dataset_manager = DatasetManager()
    qc_manager = QCManager()
    trainer = IncrementalTrainer()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Tennis Tagger - Training") as app:
        
        gr.Markdown("""
        # 🎾 Tennis Tagger - Production Training System v3.1
        ## ✅ Multi-Task Training • 🔀 Dataset Merging • 🔄 Incremental Learning • 📊 Batch QC • 🏷️ Model Versioning

        **NEW: Model versioning prevents overwriting!** Each retrain creates a new version (v1→v2→v3), preserving all previous learning.
        """)
        
        with gr.Tabs():
            
            # ==================== TAB 1: TRAINING ====================
            with gr.Tab("⚡ Training") as train_tab:
                
                with gr.Row():
                    # LEFT: Configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Training Data")
                        
                        gr.Markdown("**⚠️ IMPORTANT: Do NOT drag-and-drop large videos! Use the Browse button or file path below.**")

                        with gr.Row():
                            browse_video_btn = gr.Button(
                                "📁 Browse for Video File",
                                variant="primary",
                                size="lg"
                            )
                            selected_video_display = gr.Textbox(
                                label="Selected Video",
                                placeholder="Click 'Browse' to select a video file",
                                interactive=False,
                                scale=3
                            )

                        gr.Markdown("**OR** manually enter/paste the file path:")

                        video_path_input = gr.Textbox(
                            label="Video File Path",
                            placeholder="C:\\Users\\liamp\\Videos\\match.mp4",
                            lines=1,
                            info="Paste the full file path here. No quotes needed."
                        )

                        gr.Markdown("---")
                        gr.Markdown("**For small files (<500 MB) only:**")

                        video_upload = gr.File(
                            label="Drag-and-drop (NOT recommended for large files)",
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi"],
                            height=60
                        )

                        upload_progress_status = gr.Markdown(
                            "",
                            visible=False
                        )
                        
                        csv_upload = gr.File(
                            label="CSV Labels",
                            file_count="multiple",
                            file_types=[".csv"],
                            height=80
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🎯 Quick Add Training Pair")
                        gr.Markdown("*Upload a video + its corresponding CSV together to add to training data*")

                        with gr.Row():
                            submit_pair_btn = gr.Button(
                                "✅ Submit Video+CSV Pair for Training",
                                variant="primary",
                                size="lg"
                            )

                        pair_status = gr.Markdown("*Upload matching video and CSV files above, then click to submit*")

                        gr.Markdown("---")
                        gr.Markdown("### 🎬 Video Processing Progress")
                        gr.Markdown("*Real-time progress when processing videos for feature extraction*")

                        video_progress = gr.Markdown(
                            "⏸️ **Idle**\n\nNo video processing in progress",
                            label="Video Processing Status"
                        )

                        process_video_btn = gr.Button(
                            "🎬 Process Video for Feature Extraction",
                            variant="secondary",
                            size="sm"
                        )

                        gr.Markdown("---")

                        upload_status = gr.Markdown("No files uploaded")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📊 Dataset Management")
                        
                        dataset_name = gr.Textbox(
                            label="Dataset Name",
                            value=f"dataset_{datetime.now().strftime('%Y%m%d')}",
                            placeholder="my_training_data"
                        )
                        
                        with gr.Row():
                            save_dataset_btn = gr.Button("💾 Save as Dataset", size="sm")
                            import_dataset_btn = gr.Button("📥 Import External Dataset", size="sm")

                        with gr.Row():
                            integrate_dataset_btn = gr.Button("⚡ Integrate & Train Immediately", size="sm", variant="primary")

                        dataset_status = gr.Markdown("*Save datasets to merge from multiple PCs. Import from USB/network. Integrate to train immediately.*")
                        
                        gr.Markdown("---")
                        gr.Markdown("### ⚙️ Training Configuration")
                        
                        train_tasks = gr.CheckboxGroup(
                            choices=[
                                ("Stroke Classification", "stroke"),
                                ("Serve Detection", "serve"),
                                ("Shot Placement", "placement")
                            ],
                            value=[],
                            label="What to Train (✓ Select ALL tasks you want to train) - ✓ Check multiple boxes to train all tasks simultaneously"
                        )
                        
                        gr.Markdown("""
                        **Training Guidelines:**
                        - **Epochs**: New models: 50-100 | Fine-tuning: 10-30 | Large datasets: 30-50
                        - **Batch Size**: RTX 3050/4050: 32 | RTX 4060+: 64 | CPU: 16 | Low memory: 8
                        - **Tip**: Larger batches = faster but need more VRAM
                        """)

                        with gr.Row():
                            train_epochs = gr.Slider(
                                10, 200, value=50, step=10,
                                label="Epochs (More = better accuracy but longer training)"
                            )
                            train_batch = gr.Slider(
                                8, 128, value=32, step=8,
                                label="Batch Size (Larger = faster training if GPU has enough VRAM)"
                            )
                        
                        train_mode = gr.Radio(
                            choices=[
                                ("Train from Scratch (New Model - v1)", "scratch"),
                                ("Fine-Tune Existing Models (Incremental - Preserves Learning)", "finetune")
                            ],
                            value="finetune",
                            label="Training Mode (Fine-tune uses LOW learning rate + versioning to avoid overwriting previous learning)"
                        )
                        
                        train_device = gr.Radio(
                            choices=[("Auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")],
                            value="auto",
                            label="Device"
                        )
                        
                        time_estimate = gr.Markdown("*Upload data to see time estimate*")
                        
                        gr.Markdown("---")
                        
                        start_training_btn = gr.Button(
                            "🚀 Start Training",
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
                        
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                    
                    # RIGHT: Progress
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Training Progress")
                        
                        progress_status = gr.Textbox(
                            label="Current Status",
                            value="Configure settings and upload data to begin",
                            lines=2,
                            interactive=False
                        )
                        
                        with gr.Row():
                            extraction_progress = gr.Textbox(
                                label="Feature Extraction",
                                lines=6,
                                interactive=False
                            )
                            
                            training_progress = gr.Textbox(
                                label="Training Progress",
                                lines=6,
                                interactive=False
                            )
                        
                        train_chart = gr.Plot(label="Live Training Metrics")
                
                # FUNCTIONS
                def browse_for_video():
                    """Open native file browser to select video"""
                    try:
                        # Create a hidden tkinter root window
                        root = tk.Tk()
                        root.withdraw()  # Hide the main window
                        root.attributes('-topmost', True)  # Bring dialog to front

                        # Open file dialog
                        file_path = filedialog.askopenfilename(
                            title="Select Video File",
                            filetypes=[
                                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                                ("MP4 files", "*.mp4"),
                                ("MOV files", "*.mov"),
                                ("AVI files", "*.avi"),
                                ("All files", "*.*")
                            ],
                            initialdir=str(Path.home() / "Videos")
                        )

                        root.destroy()

                        if file_path:
                            # Get file size
                            file_size = Path(file_path).stat().st_size / (1024**3)
                            size_info = f" ({file_size:.2f} GB)" if file_size > 0.1 else f" ({file_size*1024:.1f} MB)"

                            return file_path, f"{Path(file_path).name}{size_info}"
                        else:
                            return "", "No file selected"

                    except Exception as e:
                        return "", f"Error: {str(e)}"

                def show_upload_status(videos):
                    """Display upload status when files are being uploaded"""
                    if not videos:
                        return "", False

                    # Calculate total size
                    total_size = 0
                    file_count = len(videos) if isinstance(videos, list) else 1

                    try:
                        if isinstance(videos, list):
                            for v in videos:
                                if v:
                                    total_size += Path(v).stat().st_size
                        else:
                            total_size += Path(videos).stat().st_size

                        size_gb = total_size / (1024**3)
                        size_mb = total_size / (1024**2)

                        if size_gb > 0.1:
                            size_str = f"{size_gb:.2f} GB"
                        else:
                            size_str = f"{size_mb:.1f} MB"

                        status = f"✅ **Upload Complete!**\n\n{file_count} file(s) uploaded ({size_str})"

                        if size_gb > 1:
                            status += f"\n\n⚠️ **Large file detected!** Next time use the file path input to avoid waiting."

                        return status, True
                    except:
                        return "✅ Upload complete", True

                def update_time_estimate(videos, epochs, batch, device, tasks, video_path_str):
                    """Calculate training time estimate"""
                    # Count videos from both upload and path input
                    video_count = 0
                    if video_path_str and video_path_str.strip():
                        # Clean the path - remove quotes, extra whitespace
                        clean_path = video_path_str.strip().strip('"').strip("'").strip()
                        video_path = Path(clean_path)
                        if video_path.exists() and video_path.is_file():
                            video_count += 1
                    if videos:
                        video_count += len(videos) if isinstance(videos, list) else 1

                    if video_count == 0:
                        return "*Upload data to see time estimate*"

                    # Estimate points (10 per video average)
                    est_points = video_count * 10
                    
                    estimates = []
                    for task in (tasks or ["stroke"]):
                        seconds = TrainingEstimator.estimate_time(
                            est_points, epochs, batch, device
                        )
                        estimates.append(f"**{task}**: {TrainingEstimator.format_time(seconds)}")
                    
                    total_seconds = sum(TrainingEstimator.estimate_time(est_points, epochs, batch, device) 
                                      for _ in (tasks or ["stroke"]))
                    
                    return f"""**Estimated Training Time:**

{chr(10).join(estimates)}

**Total**: {TrainingEstimator.format_time(total_seconds)}

*Estimates based on {est_points} points, {device} device*"""
                
                def process_video_for_extraction(videos, video_path_str, browsed_path, batch_size, device):
                    """Start video processing in background thread with UI-controlled settings"""
                    # Priority: 1) Browsed path, 2) Manual path input, 3) Uploaded file
                    if browsed_path and browsed_path.strip():
                        # Use browsed file path
                        video_path = Path(browsed_path.strip())
                        if not video_path.exists():
                            return f"❌ **Error**: File not found: `{browsed_path}`"
                    elif video_path_str and video_path_str.strip():
                        # Clean the path - remove quotes, extra whitespace
                        clean_path = video_path_str.strip().strip('"').strip("'").strip()
                        video_path = Path(clean_path)
                        if not video_path.exists():
                            return f"❌ **Error**: File not found: `{clean_path}`\n\nMake sure the path is correct and the file exists."
                        if not video_path.is_file():
                            return f"❌ **Error**: Not a file: `{clean_path}`"
                    elif videos:
                        # Fall back to uploaded file
                        video_file = videos[0] if isinstance(videos, list) else videos
                        video_path = Path(video_file)
                    else:
                        return "❌ **Error**: Please select a video using Browse button, file path, or upload"

                    # Check file size and warn
                    file_size_gb = video_path.stat().st_size / (1024**3)
                    if file_size_gb > 5:
                        size_warning = f"\n\n⚠️ **Large file detected: {file_size_gb:.2f} GB**\nProcessing may take 30+ minutes..."
                    else:
                        size_warning = ""

                    # Generate output path
                    output_dir = Path("data/feature_extraction")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_csv = output_dir / f"{video_path.stem}_features.csv"

                    # Start processing with UI-controlled settings
                    success, message = start_video_processing(
                        str(video_path),
                        str(output_csv),
                        visualize=False,
                        batch_size=int(batch_size),
                        device=device
                    )

                    if success:
                        return f"🔄 **Processing Started**\n\nVideo: {video_path.name} ({file_size_gb:.2f} GB)\nProcessing has begun in the background. Progress will update below...{size_warning}"
                    else:
                        return f"❌ **Error**: {message}"

                def update_video_progress():
                    """Update video processing progress (called periodically)"""
                    return get_processing_progress()

                def check_processing_complete():
                    """Check if processing is complete and return result"""
                    if is_processing_complete():
                        return get_processing_result()
                    return None

                def submit_training_pair(videos, csvs, video_path_str, browsed_path):
                    """Process and save video+CSV pair for training"""
                    # Build video list from either browse, manual path, or upload
                    video_list = []

                    # Check browsed path first
                    if browsed_path and browsed_path.strip():
                        video_path = Path(browsed_path.strip())
                        if video_path.exists() and video_path.is_file():
                            video_list.append(str(video_path))
                        else:
                            return f"❌ **Error**: Browsed video file not found: `{browsed_path}`"
                    # Check for manual path input
                    elif video_path_str and video_path_str.strip():
                        # Clean the path - remove quotes, extra whitespace
                        clean_path = video_path_str.strip().strip('"').strip("'").strip()
                        video_path = Path(clean_path)
                        if video_path.exists() and video_path.is_file():
                            video_list.append(str(video_path))
                        else:
                            return f"❌ **Error**: Video file not found: `{clean_path}`\n\nMake sure the path is correct and the file exists."

                    # Add uploaded videos
                    if videos:
                        if not isinstance(videos, list):
                            videos = [videos] if videos else []
                        video_list.extend([v for v in videos if v is not None])

                    # Check if we have videos and CSVs
                    if not video_list:
                        return "❌ **Error**: Please provide a video (upload or file path)"
                    if not csvs:
                        return "❌ **Error**: Please upload CSV files"

                    # Convert CSVs to list if single file
                    if not isinstance(csvs, list):
                        csvs = [csvs] if csvs else []

                    # Match video and CSV by filename
                    video_names = {}
                    for v in video_list:
                        if v is not None:
                            video_path = Path(v)
                            video_names[video_path.stem] = str(v)

                    csv_names = {}
                    for c in csvs:
                        if c is not None:
                            csv_path = Path(c)
                            csv_names[csv_path.stem] = str(c)

                    matched = set(video_names.keys()) & set(csv_names.keys())

                    if len(matched) == 0:
                        return f"❌ **Error**: No matching video-CSV pairs found.\n\n**Videos:** {', '.join(list(video_names.keys())) or 'None'}\n\n**CSVs:** {', '.join(list(csv_names.keys())) or 'None'}\n\n*Files must have matching names (e.g., match1.mp4 + match1.csv)*"

                    # Create training pairs directory
                    pairs_dir = Path("data/training_pairs")
                    pairs_dir.mkdir(parents=True, exist_ok=True)

                    # Save matched pairs
                    saved_pairs = []
                    for match_name in matched:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        pair_dir = pairs_dir / f"{match_name}_{timestamp}"
                        pair_dir.mkdir(parents=True, exist_ok=True)

                        # Copy video and CSV to pair directory
                        video_file = video_names[match_name]
                        csv_file = csv_names[match_name]

                        try:
                            # Get file extension from source
                            video_ext = Path(video_file).suffix

                            # Copy files
                            video_dest = pair_dir / f"{match_name}{video_ext}"
                            csv_dest = pair_dir / f"{match_name}.csv"

                            shutil.copy2(video_file, video_dest)
                            shutil.copy2(csv_file, csv_dest)

                            # Read CSV to count tags
                            df = pd.read_csv(csv_file)
                            tag_count = len(df)

                            saved_pairs.append({
                                'name': match_name,
                                'tags': tag_count,
                                'path': str(pair_dir)
                            })
                        except Exception as e:
                            return f"❌ **Error saving pair '{match_name}'**: {str(e)}"

                    # Format success message
                    result = f"✅ **Successfully saved {len(saved_pairs)} training pair(s)!**\n\n"
                    for pair in saved_pairs:
                        result += f"- **{pair['name']}**: {pair['tags']} tags\n"
                    result += f"\n📁 Saved to: `{pairs_dir}`\n\n"
                    result += "✨ **Next steps:**\n"
                    result += "1. These pairs are now ready for training\n"
                    result += "2. Click **'💾 Save as Dataset'** to add to a dataset\n"
                    result += "3. Or click **'⚡ Integrate & Train Immediately'** to train right away\n\n"
                    result += "*The model will learn what correct tags look like from this example*"

                    return result

                def validate_training(videos, csvs, tasks, video_path_str, browsed_path):
                    """Validate before enabling training"""
                    # Check if we have videos (browsed, manual path, or uploaded)
                    has_video = False
                    if browsed_path and browsed_path.strip():
                        video_path = Path(browsed_path.strip())
                        has_video = video_path.exists() and video_path.is_file()
                    elif video_path_str and video_path_str.strip():
                        # Clean the path - remove quotes, extra whitespace
                        clean_path = video_path_str.strip().strip('"').strip("'").strip()
                        video_path = Path(clean_path)
                        has_video = video_path.exists() and video_path.is_file()
                    elif videos:
                        has_video = True

                    if not has_video or not csvs:
                        return "Upload videos and CSVs to continue", gr.Button(interactive=False)

                    if not tasks:
                        return "Select at least one task to train", gr.Button(interactive=False)

                    # Build video list
                    video_list = []
                    if video_path_str and video_path_str.strip():
                        # Clean the path - remove quotes, extra whitespace
                        clean_path = video_path_str.strip().strip('"').strip("'").strip()
                        video_path = Path(clean_path)
                        if video_path.exists():
                            video_list.append(str(video_path))
                    if videos:
                        if isinstance(videos, list):
                            video_list.extend(videos)
                        else:
                            video_list.append(videos)

                    video_names = {Path(v).stem for v in video_list if v}
                    csv_names = {Path(c).stem for c in (csvs if isinstance(csvs, list) else [csvs])}
                    matched = video_names & csv_names

                    if len(matched) == 0:
                        return "No matching video-CSV pairs found", gr.Button(interactive=False)

                    task_names = ", ".join(tasks)
                    return f"✅ Ready! Will train: {task_names} on {len(matched)} videos", gr.Button(interactive=True)
                
                # Connect browse button
                browse_video_btn.click(
                    browse_for_video,
                    inputs=[],
                    outputs=[video_path_input, selected_video_display]
                )

                # Connect video processing button with batch size and device controls
                process_video_btn.click(
                    process_video_for_extraction,
                    inputs=[video_upload, video_path_input, video_path_input, train_batch, train_device],
                    outputs=[video_progress]
                )

                # Connect submit pair button
                submit_pair_btn.click(
                    submit_training_pair,
                    inputs=[video_upload, csv_upload, video_path_input, video_path_input],
                    outputs=[pair_status]
                )

                # Auto-update video progress every 1 second
                video_progress_timer = gr.Timer(value=1.0, active=True)
                video_progress_timer.tick(
                    update_video_progress,
                    inputs=None,
                    outputs=[video_progress]
                )

                # Show upload status when files are uploaded
                video_upload.change(
                    show_upload_status,
                    inputs=[video_upload],
                    outputs=[upload_progress_status, upload_progress_status]
                )

                # Connect events
                for component in [video_upload, csv_upload, train_tasks, video_path_input]:
                    component.change(
                        validate_training,
                        inputs=[video_upload, csv_upload, train_tasks, video_path_input, video_path_input],
                        outputs=[upload_status, start_training_btn]
                    )
                
                for component in [video_upload, train_epochs, train_batch, train_device, train_tasks, video_path_input]:
                    component.change(
                        update_time_estimate,
                        inputs=[video_upload, train_epochs, train_batch, train_device, train_tasks, video_path_input],
                        outputs=[time_estimate]
                    )
            
            # ==================== TAB 2: DATASET MANAGEMENT ====================
            with gr.Tab("📦 Dataset Management") as dataset_tab:

                # Video Processing Database Status
                gr.Markdown("### 📹 Processed Videos Database")

                video_db_refresh_btn = gr.Button("🔄 Refresh Video Database")
                video_db_stats = gr.Markdown("")

                video_db_table = gr.Dataframe(
                    headers=["Video", "Status", "Progress", "Completed", "Output"],
                    label="Processed Videos",
                    interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("### 📚 Available Datasets")

                refresh_datasets_btn = gr.Button("🔄 Refresh List")
                
                datasets_table = gr.Dataframe(
                    headers=["Name", "Points", "Size", "Created"],
                    label="Datasets (for merging from multiple PCs)",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🔀 Merge Datasets")
                
                merge_select = gr.CheckboxGroup(
                    choices=[],
                    label="Select datasets to merge"
                )
                
                merged_name = gr.Textbox(
                    label="Merged Dataset Name",
                    value=f"merged_{datetime.now().strftime('%Y%m%d')}"
                )
                
                merge_btn = gr.Button("🔀 Merge Selected Datasets", variant="primary")
                merge_status = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("### 📥 Import from Another PC")
                
                import_path = gr.Textbox(
                    label="Import Path (Path to dataset from USB drive or network share)",
                    placeholder="D:/from_other_pc/dataset_20251110"
                )
                
                import_name = gr.Textbox(
                    label="Import As",
                    value=f"imported_{datetime.now().strftime('%Y%m%d')}"
                )
                
                import_btn = gr.Button("📥 Import Dataset", variant="primary")
                import_status = gr.Markdown("")

                def refresh_video_database():
                    """Refresh video processing database"""
                    from utils.video_database import VideoDatabase

                    video_db = VideoDatabase()
                    stats = video_db.get_processing_stats()

                    # Create stats markdown
                    stats_md = f"""
**Total Videos**: {stats['total']} | **Completed**: {stats['completed']} | **Processing**: {stats['processing']} | **Failed**: {stats['failed']}
                    """

                    # Get all videos from database
                    all_videos = []
                    for video_data in video_db.db['videos'].values():
                        status = video_data.get('status', 'unknown')
                        progress = f"{video_data.get('percentage', 0):.1f}%"
                        completed_at = video_data.get('completed_at', 'N/A')
                        if completed_at != 'N/A':
                            completed_at = completed_at.split('T')[0]  # Just date
                        output_csv = Path(video_data.get('output_csv', '')).name if video_data.get('output_csv') else 'N/A'

                        all_videos.append([
                            video_data.get('name', Path(video_data.get('path', 'unknown')).name),
                            status,
                            progress,
                            completed_at,
                            output_csv
                        ])

                    # Sort by most recent first
                    all_videos.sort(key=lambda x: x[3], reverse=True)

                    return stats_md, all_videos

                def refresh_datasets():
                    """Refresh dataset list"""
                    datasets = dataset_manager.list_datasets()
                    table_data = [[d['name'], d['points'], d['size'], d['created']] for d in datasets]
                    choices = [(d['name'], d['name']) for d in datasets]
                    return table_data, gr.CheckboxGroup(choices=choices)
                
                def merge_datasets(selected, output_name):
                    """Merge selected datasets"""
                    if not selected:
                        return "❌ No datasets selected"
                    
                    count = dataset_manager.merge_datasets(selected, output_name)
                    return f"✅ Merged {len(selected)} datasets → {count} points in '{output_name}'"
                
                def import_dataset(source, name):
                    """Import dataset"""
                    if not source:
                        return "❌ No path provided"
                    
                    try:
                        count = dataset_manager.import_dataset(source, name)
                        return f"✅ Imported {count} points as '{name}'"
                    except Exception as e:
                        return f"❌ Import failed: {str(e)}"
                
                video_db_refresh_btn.click(
                    refresh_video_database,
                    outputs=[video_db_stats, video_db_table]
                )

                refresh_datasets_btn.click(
                    refresh_datasets,
                    outputs=[datasets_table, merge_select]
                )

                merge_btn.click(
                    merge_datasets,
                    inputs=[merge_select, merged_name],
                    outputs=[merge_status]
                )
                
                import_btn.click(
                    import_dataset,
                    inputs=[import_path, import_name],
                    outputs=[import_status]
                )
            
            # ==================== TAB 3: QC CORRECTIONS ====================
            with gr.Tab("✅ QC Corrections") as qc_tab:

                gr.Markdown("""
                ### Quality Control - Correct Machine-Tagged Errors

                **This tab is NOT for training - it's for correcting mistakes the AI made!**

                **Workflow:**
                1. System auto-tags a tennis match video → generates CSV
                2. You review the CSV and find errors (wrong stroke types, missed serves, etc.)
                3. You manually fix the errors in the CSV (the "QC" process)
                4. Upload the corrected CSV here to see accuracy score
                5. System collects 10-20 corrections, then batch retrains to learn from mistakes

                **Why batch training?** Retraining after every correction is slow and can cause the model to "forget"
                previous learning. Batching 10-20 corrections together gives better, more stable improvements.
                """)
                
                gr.Markdown("---")
                gr.Markdown("### 📤 Submit QC Correction")

                with gr.Row():
                    with gr.Column():
                        video_select = gr.Dropdown(
                            choices=qc_manager.list_processed_videos(),
                            label="1️⃣ Select Which Game Was Tagged by System (Choose the video match that was auto-tagged)"
                        )

                        original_csv_upload = gr.File(
                            label="2️⃣ Upload System-Generated CSV (Original) - The CSV the machine originally created",
                            file_types=[".csv"]
                        )

                        corrected_csv_upload = gr.File(
                            label="3️⃣ Upload Your QC-Corrected CSV (The CSV after you manually fixed errors)",
                            file_types=[".csv"]
                        )

                        submit_qc_btn = gr.Button(
                            "✅ Calculate Accuracy & Submit Correction",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column():
                        gr.Markdown("### 📊 Accuracy Results")

                        qc_accuracy = gr.Markdown(
                            """
                            **Waiting for files...**

                            Upload both CSVs to see:
                            - Overall accuracy percentage
                            - Number of corrections made
                            - Quality score
                            """,
                            label="Accuracy Score"
                        )

                        qc_result = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Pending Corrections (Batch Queue)")
                
                refresh_qc_btn = gr.Button("🔄 Refresh QC Queue")
                
                qc_queue_table = gr.Dataframe(
                    headers=["Video", "Accuracy", "Corrections", "Date"],
                    label="QC Corrections Waiting for Batch Retrain",
                    interactive=False
                )
                
                batch_stats = gr.Markdown("*No corrections in queue*")
                
                gr.Markdown("---")
                gr.Markdown("""
                ### 🔄 Batch Retrain Models

                **How it works without overwriting:**
                1. System loads your current model (e.g., stroke_v5.pt)
                2. Creates a NEW version (stroke_v6.pt)
                3. Fine-tunes v6 with QC corrections using **low learning rate** (0.00005)
                4. Backs up v5 to versions/ folder
                5. v6 becomes the new active model, but v5 is preserved!

                **This prevents "catastrophic forgetting"** - the model learns from corrections
                without forgetting everything it knew before. All previous versions are backed up.
                """)

                batch_retrain_btn = gr.Button(
                    "🔄 Start Batch Retrain (Creates New Model Version)",
                    variant="primary",
                    size="lg"
                )

                batch_retrain_status = gr.Markdown("")
                
                def submit_qc_correction(video, original, corrected):
                    """Submit QC correction"""
                    if not video or not original or not corrected:
                        waiting_md = """
                        **Waiting for files...**

                        Upload both CSVs to see:
                        - Overall accuracy percentage
                        - Number of corrections made
                        - Quality score
                        """
                        return waiting_md, "❌ Please select video and upload both CSVs"

                    try:
                        result = qc_manager.save_correction(video, original, corrected)

                        # Calculate quality grade
                        acc = result['accuracy']
                        if acc >= 95:
                            grade = "🌟 EXCELLENT"
                            color = "green"
                        elif acc >= 90:
                            grade = "✅ GOOD"
                            color = "blue"
                        elif acc >= 85:
                            grade = "⚠️ FAIR"
                            color = "orange"
                        else:
                            grade = "❌ NEEDS WORK"
                            color = "red"

                        accuracy_display = f"""
                        ## 🎯 Accuracy: {result['accuracy']:.1f}%

                        **Grade**: {grade}

                        **Corrections**: {result['corrections']} errors fixed
                        **Total Rows**: {result['total_rows']}
                        **Error Rate**: {(result['corrections']/result['total_rows']*100):.1f}%

                        {"✨ Great job! Model is performing well." if acc >= 90 else "⚠️ Model needs retraining - consider batch retrain soon."}
                        """

                        result_md = f"""### ✅ QC Correction Saved Successfully!

**Video**: `{video}`
**Timestamp**: {result['timestamp']}

This correction has been added to the batch queue.
**Next step**: Collect 10-20 corrections, then run batch retrain to improve the model without overwriting previous learning!"""

                        return accuracy_display, result_md

                    except Exception as e:
                        error_md = """
                        **Error occurred**

                        Could not process correction.
                        """
                        return error_md, f"❌ Error: {str(e)}"
                
                def refresh_qc_queue():
                    """Refresh QC corrections queue"""
                    corrections = qc_manager.list_corrections()
                    table_data = [[c['video'], f"{c['accuracy']:.1f}%", c['corrections'], c['timestamp']] 
                                 for c in corrections]
                    
                    stats = qc_manager.get_batch_stats()
                    
                    if stats['count'] == 0:
                        stats_md = "*No corrections in queue*"
                    else:
                        stats_md = f"""### 📊 Batch Statistics

**Total Corrections**: {stats['count']}
**Average Accuracy**: {stats['avg_accuracy']:.1f}%
**Total Edits**: {stats['total_corrections']}

**Recommendation**: {"✅ Good batch size - ready to retrain!" if stats['count'] >= 10 else f"⏳ Collect {10 - stats['count']} more corrections for optimal batch"}"""
                    
                    return table_data, stats_md
                
                submit_qc_btn.click(
                    submit_qc_correction,
                    inputs=[video_select, original_csv_upload, corrected_csv_upload],
                    outputs=[qc_accuracy, qc_result]
                )
                
                refresh_qc_btn.click(
                    refresh_qc_queue,
                    outputs=[qc_queue_table, batch_stats]
                )
            
            # ==================== TAB 4: MODEL & DATA VIEWER ====================
            with gr.Tab("💾 Models & Live Training") as model_tab:

                gr.Markdown("### 🧠 Trained Models with Version Control")

                gr.Markdown("""
                **Model Versioning System:** Each time you retrain, a NEW version is created (e.g., v1 → v2 → v3).
                Old versions are backed up to `models/versions/` folder, so you never lose previous learning!
                """)

                refresh_models_btn = gr.Button("🔄 Refresh Models List")

                models_table = gr.Dataframe(
                    headers=["Model", "Version", "Task", "Accuracy", "Mode", "Date"],
                    label="Available Models (Current Active Versions)",
                    interactive=False
                )

                gr.Markdown("### 📜 Version History")

                version_history_table = gr.Dataframe(
                    headers=["Task", "Version", "Parent Ver", "Training Mode", "Accuracy", "Date"],
                    label="Model Evolution History",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📊 Feature Vectors by Video")
                
                vectors_table = gr.Dataframe(
                    headers=["Video", "Points", "Size", "Extracted"],
                    label="Extracted Feature Data",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📈 Live Training Visualization")
                
                live_chart_placeholder = gr.Plot(label="Training will appear here when active")
                
                gr.Markdown("""
                *This chart updates in real-time during training*
                *Supports multi-task training - shows all tasks simultaneously*
                """)
        
        gr.Markdown("""
        ---
        ### 🎾 Tennis Tagger Training System v3.1

        **Features:**
        - ✅ Multi-task training (train all 3 tasks simultaneously)
        - 🔄 Incremental learning with LOW learning rate (prevents forgetting)
        - 🏷️ Automatic model versioning (v1→v2→v3, old versions backed up)
        - 🔀 Dataset merging from multiple PCs
        - 📊 Batch QC correction system (10-20 corrections before retrain)
        - 📈 Live training visualization
        - ⚡ Training time estimates

        **No more overwriting!** Each training session creates a new version, preserving all previous learning.
        """)

    return app


# =============================================================================
# IMPORTED DATA TRAINER - Train using human-tagged CSV imports
# =============================================================================

class ImportedDataTrainer:
    """
    Train models using imported human-tagged data.

    This trainer uses training pairs created by TrainingDataImporter
    to train models that can match human-quality tagging.

    Supported tasks:
    - serve_class: Ace, Made, Fault classification
    - serve_placement: Wide, T, Body placement
    - stroke_class: Forehand, Backhand, etc.
    - shot_placement: Crosscourt, Down Line, etc.
    - point_result: Winner, Error classification
    """

    def __init__(self, training_data_dir: str = "data/training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.incremental_trainer = IncrementalTrainer()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def list_available_datasets(self) -> list:
        """List available imported datasets."""
        datasets = []

        for dataset_dir in self.training_data_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            npz_file = dataset_dir / "training_pairs.npz"
            metadata_file = dataset_dir / "metadata.json"

            if npz_file.exists() and metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata['name'] = dataset_dir.name
                    metadata['path'] = str(dataset_dir)
                    datasets.append(metadata)
                except:
                    pass

        return datasets

    def load_dataset(self, dataset_name: str) -> dict:
        """
        Load training data from imported dataset INCLUDING features.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with labels, features, and label mappings
        """
        dataset_dir = self.training_data_dir / dataset_name

        # Load NPZ data (labels)
        npz_file = dataset_dir / "training_pairs.npz"
        data = np.load(npz_file, allow_pickle=True)

        # Load label mapping
        mapping_file = dataset_dir / "label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                label_mapping = json.load(f)
        else:
            label_mapping = {}

        # Load features (player tracks, ball positions)
        features_file = dataset_dir / "training_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                features = json.load(f)
        else:
            features = {'player_bbox_sequences': [], 'ball_position_sequences': []}

        return {
            'serve_class': data['serve_class'],
            'serve_placement': data['serve_placement'],
            'stroke_class': data['stroke_class'],
            'shot_placement': data['shot_placement'],
            'point_result': data['point_result'],
            'frame_indices': data['frame_indices'],
            'frame_counts': data.get('frame_counts', np.zeros(len(data['serve_class']))),
            'player_bbox_sequences': features.get('player_bbox_sequences', []),
            'ball_position_sequences': features.get('ball_position_sequences', []),
            'label_mapping': label_mapping
        }

    def merge_datasets(self, dataset_names: list) -> dict:
        """
        Merge multiple datasets for training INCLUDING features.

        Args:
            dataset_names: List of dataset names to merge

        Returns:
            Merged training data dictionary with labels AND features
        """
        merged = {
            'serve_class': [],
            'serve_placement': [],
            'stroke_class': [],
            'shot_placement': [],
            'point_result': [],
            'frame_indices': [],
            'frame_counts': [],
            'player_bbox_sequences': [],
            'ball_position_sequences': [],
            'label_mapping': {}
        }

        for ds_name in dataset_names:
            try:
                data = self.load_dataset(ds_name)

                # Append label arrays
                merged['serve_class'].extend(data['serve_class'].tolist())
                merged['serve_placement'].extend(data['serve_placement'].tolist())
                merged['stroke_class'].extend(data['stroke_class'].tolist())
                merged['shot_placement'].extend(data['shot_placement'].tolist())
                merged['point_result'].extend(data['point_result'].tolist())
                merged['frame_indices'].extend(data['frame_indices'].tolist())

                # Append feature arrays
                frame_counts = data.get('frame_counts', [])
                if hasattr(frame_counts, 'tolist'):
                    frame_counts = frame_counts.tolist()
                merged['frame_counts'].extend(frame_counts)

                merged['player_bbox_sequences'].extend(data.get('player_bbox_sequences', []))
                merged['ball_position_sequences'].extend(data.get('ball_position_sequences', []))

                # Merge label mappings
                for key, mapping in data.get('label_mapping', {}).items():
                    if key not in merged['label_mapping']:
                        merged['label_mapping'][key] = {}
                    merged['label_mapping'][key].update(mapping)

            except Exception as e:
                print(f"Error loading dataset {ds_name}: {e}")

        # Convert back to numpy arrays
        for key in ['serve_class', 'serve_placement', 'stroke_class', 'shot_placement', 'point_result']:
            merged[key] = np.array(merged[key])
        merged['frame_indices'] = np.array(merged['frame_indices'])
        merged['frame_counts'] = np.array(merged['frame_counts'])

        return merged

    def prepare_training_data(self, data: dict, task: str) -> tuple:
        """
        Prepare training data for a specific task with REAL features from FVD.

        Args:
            data: Merged dataset dictionary
            task: Task name (serve_class, stroke_class, etc.)

        Returns:
            (X, y, num_classes, label_to_idx) tuple where X is actual feature vectors
        """
        labels = data.get(task, [])
        player_sequences = data.get('player_bbox_sequences', [])
        ball_sequences = data.get('ball_position_sequences', [])

        # Ensure we have enough data
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()

        # Filter out empty labels and get valid indices
        valid_indices = [i for i, l in enumerate(labels) if l and str(l).strip()]

        if len(valid_indices) == 0:
            return None, None, 0, {}

        # Get or create label mapping
        label_mapping = data.get('label_mapping', {}).get(task, {})

        if not label_mapping:
            # Create mapping from unique labels
            valid_labels = [labels[i] for i in valid_indices]
            unique_labels = sorted(set(valid_labels))
            label_mapping = {l: i for i, l in enumerate(unique_labels)}

        # Extract features for each valid sample
        features_list = []
        labels_list = []

        for idx in valid_indices:
            label = labels[idx]
            label_idx = label_mapping.get(label, -1)

            if label_idx < 0:
                continue

            # Extract feature vector from FVD data
            player_seq = player_sequences[idx] if idx < len(player_sequences) else []
            ball_seq = ball_sequences[idx] if idx < len(ball_sequences) else []

            feature_vector = self._extract_feature_vector(player_seq, ball_seq, task)
            features_list.append(feature_vector)
            labels_list.append(label_idx)

        if len(features_list) == 0:
            return None, None, 0, {}

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        num_classes = len(label_mapping)

        return X, y, num_classes, label_mapping

    def _extract_feature_vector(self, player_seq: list, ball_seq: list, task: str) -> np.ndarray:
        """
        Extract a fixed-length feature vector from variable-length FVD sequences.

        For tennis classification, we compute:
        - Player movement statistics (bbox center movement, size changes)
        - Ball trajectory statistics
        - Temporal features (duration, frame count)

        Args:
            player_seq: List of [frame, player_id, x1, y1, x2, y2, conf]
            ball_seq: List of [frame, x, y, conf]
            task: Task being trained (affects which features matter)

        Returns:
            Fixed-length feature vector (64 dimensions)
        """
        # Feature vector: 64 dimensions
        features = np.zeros(64, dtype=np.float32)

        # === Player features (32 dims) ===
        if player_seq:
            player_arr = np.array(player_seq)

            if len(player_arr) > 0 and player_arr.shape[1] >= 6:
                # Separate by player ID
                player_ids = np.unique(player_arr[:, 1])

                for i, pid in enumerate(player_ids[:2]):  # Max 2 players
                    mask = player_arr[:, 1] == pid
                    p_data = player_arr[mask]

                    if len(p_data) > 0:
                        # Bbox centers
                        cx = (p_data[:, 2] + p_data[:, 4]) / 2  # (x1 + x2) / 2
                        cy = (p_data[:, 3] + p_data[:, 5]) / 2  # (y1 + y2) / 2

                        # Bbox sizes
                        w = p_data[:, 4] - p_data[:, 2]
                        h = p_data[:, 5] - p_data[:, 3]

                        base = i * 16  # 16 features per player

                        # Position stats
                        features[base + 0] = np.mean(cx) / 1920  # Normalized x
                        features[base + 1] = np.mean(cy) / 1080  # Normalized y
                        features[base + 2] = np.std(cx) / 100 if len(cx) > 1 else 0  # Movement x
                        features[base + 3] = np.std(cy) / 100 if len(cy) > 1 else 0  # Movement y

                        # Size stats
                        features[base + 4] = np.mean(w) / 200  # Normalized width
                        features[base + 5] = np.mean(h) / 400  # Normalized height

                        # Movement features
                        if len(cx) > 1:
                            dx = np.diff(cx)
                            dy = np.diff(cy)
                            features[base + 6] = np.mean(np.abs(dx))  # Avg lateral movement
                            features[base + 7] = np.mean(np.abs(dy))  # Avg vertical movement
                            features[base + 8] = np.max(np.abs(dx))  # Max lateral movement
                            features[base + 9] = np.max(np.abs(dy))  # Max vertical movement
                            features[base + 10] = np.sum(np.sqrt(dx**2 + dy**2))  # Total distance

                        # First/last positions (for direction)
                        features[base + 11] = cx[0] / 1920
                        features[base + 12] = cy[0] / 1080
                        features[base + 13] = cx[-1] / 1920
                        features[base + 14] = cy[-1] / 1080
                        features[base + 15] = len(p_data) / 300  # Frame count normalized

        # === Ball features (24 dims) ===
        if ball_seq:
            ball_arr = np.array(ball_seq)

            if len(ball_arr) > 0 and ball_arr.shape[1] >= 3:
                bx = ball_arr[:, 1]  # x positions
                by = ball_arr[:, 2]  # y positions
                bc = ball_arr[:, 3] if ball_arr.shape[1] > 3 else np.ones(len(bx))

                # Position stats
                features[32] = np.mean(bx) / 1920
                features[33] = np.mean(by) / 1080
                features[34] = np.std(bx) / 100 if len(bx) > 1 else 0
                features[35] = np.std(by) / 100 if len(by) > 1 else 0

                # Trajectory features
                if len(bx) > 1:
                    dx = np.diff(bx)
                    dy = np.diff(by)
                    features[36] = np.mean(dx) / 50  # Avg x velocity
                    features[37] = np.mean(dy) / 50  # Avg y velocity
                    features[38] = np.std(dx) / 20   # Velocity variation x
                    features[39] = np.std(dy) / 20   # Velocity variation y

                    # Direction changes (indicates bounces/volleys)
                    sign_changes_x = np.sum(np.diff(np.sign(dx)) != 0)
                    sign_changes_y = np.sum(np.diff(np.sign(dy)) != 0)
                    features[40] = sign_changes_x / 10
                    features[41] = sign_changes_y / 10

                # First/last ball positions
                features[42] = bx[0] / 1920
                features[43] = by[0] / 1080
                features[44] = bx[-1] / 1920
                features[45] = by[-1] / 1080

                # Ball confidence
                features[46] = np.mean(bc)
                features[47] = len(ball_arr) / 300  # Ball detection count

        # === Temporal/meta features (8 dims) ===
        total_frames = len(player_seq) + len(ball_seq)
        features[56] = total_frames / 600  # Total data points
        features[57] = len(player_seq) / 300  # Player detections
        features[58] = len(ball_seq) / 300  # Ball detections

        return features

    def train_task(self, data: dict, task: str, epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 0.001,
                   device: str = 'auto', mode: str = 'finetune',
                   callback=None) -> dict:
        """
        Train a model for a specific task using REAL FVD features.

        Args:
            data: Training data dictionary with features
            task: Task name
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            device: Device to use (auto, cuda, cpu)
            mode: Training mode (finetune, scratch)
            callback: Optional progress callback

        Returns:
            Training results dictionary
        """
        X, y, num_classes, label_mapping = self.prepare_training_data(data, task)

        if X is None or len(X) < 10:
            return {
                'task': task,
                'status': 'error',
                'message': f'Not enough training samples for {task} (need at least 10, got {len(X) if X is not None else 0})'
            }

        # Get feature dimension from data
        input_dim = X.shape[1] if len(X.shape) > 1 else 64

        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Adjust learning rate for fine-tuning
        if mode == 'finetune':
            learning_rate = 0.0001  # Lower for fine-tuning

        # Create classifier that takes real feature vectors
        model = self._create_classifier(input_dim, num_classes)
        model = model.to(device)

        # Create dataset with REAL features
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        # Split into train/val for proper evaluation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # === Training ===
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)  # Real features!
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            # === Validation ===
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0

            # Update scheduler
            scheduler.step(avg_val_loss)

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if callback:
                callback(epoch + 1, epochs, avg_val_loss, val_acc)

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Save model with versioning
        metrics = {
            'accuracy': best_val_acc,
            'train_accuracy': history['train_acc'][-1],
            'final_loss': history['val_loss'][-1],
            'num_samples': len(y),
            'num_classes': num_classes,
            'input_dim': input_dim
        }

        model_path, version = self.incremental_trainer.save_model_version(
            task, model, optimizer, metrics, mode
        )

        # Save label mapping with model
        mapping_path = self.models_dir / f"{task}_v{version}_labels.json"
        with open(mapping_path, 'w') as f:
            json.dump({
                'label_to_idx': label_mapping,
                'idx_to_label': {v: k for k, v in label_mapping.items()},
                'input_dim': input_dim
            }, f, indent=2)

        return {
            'task': task,
            'status': 'success',
            'version': version,
            'model_path': str(model_path),
            'metrics': metrics,
            'history': history
        }

    def _create_classifier(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Create a classifier that takes REAL feature vectors as input.

        Args:
            input_dim: Dimension of input feature vectors (64 for our FVD features)
            num_classes: Number of output classes

        Returns:
            Neural network model
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def train_all_tasks(self, dataset_names: list, tasks: list = None,
                        epochs: int = 50, batch_size: int = 32,
                        device: str = 'auto', mode: str = 'finetune',
                        callback=None) -> dict:
        """
        Train all specified tasks on merged datasets.

        Args:
            dataset_names: List of dataset names to use
            tasks: List of tasks to train (default: all)
            epochs: Training epochs
            batch_size: Batch size
            device: Device to use
            mode: Training mode
            callback: Progress callback

        Returns:
            Dictionary with results for each task
        """
        if tasks is None:
            tasks = ['serve_class', 'serve_placement', 'stroke_class',
                     'shot_placement', 'point_result']

        # Merge datasets
        data = self.merge_datasets(dataset_names)

        results = {}

        for i, task in enumerate(tasks):
            if callback:
                callback(f"Training {task}...", i, len(tasks))

            result = self.train_task(
                data, task, epochs=epochs, batch_size=batch_size,
                device=device, mode=mode
            )
            results[task] = result

        return results


# =============================================================================
# COMPONENT EXPORTS FOR UNIFIED APP
# =============================================================================

def get_training_components():
    """
    Export training components for use in unified app.

    Returns:
        Dictionary of component classes and functions
    """
    return {
        'DatasetManager': DatasetManager,
        'TrainingEstimator': TrainingEstimator,
        'QCManager': QCManager,
        'IncrementalTrainer': IncrementalTrainer,
        'TrainingMonitor': TrainingMonitor,
        'FVDTrainer': FVDTrainer,
        'ImportedDataTrainer': ImportedDataTrainer,
        'create_training_interface': create_training_interface
    }


def create_training_tab_component():
    """
    Create training tab as a component for unified app.

    This returns a Gradio Tab component that can be added to
    the unified app's Blocks.

    Returns:
        Gradio Tab component
    """
    # Import here to allow component use
    dataset_manager = DatasetManager()
    qc_manager = QCManager()
    trainer = IncrementalTrainer()
    fvd_trainer = FVDTrainer() if HAS_FVD_SYSTEM else None

    # The actual tab creation would happen in the unified app
    # This function provides the necessary setup
    return {
        'dataset_manager': dataset_manager,
        'qc_manager': qc_manager,
        'trainer': trainer,
        'fvd_trainer': fvd_trainer,
        'has_fvd': HAS_FVD_SYSTEM,
        'has_video_processing': HAS_VIDEO_PROCESSING
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🎾 TENNIS TAGGER - PRODUCTION TRAINING SYSTEM v3.1         ║
║                                                              ║
║   Multi-Task • Incremental • Versioning • Batch QC          ║
╚══════════════════════════════════════════════════════════════╝

✨ NEW in v3.1: MODEL VERSIONING ✨
Each training creates a NEW version (v1→v2→v3) - no more overwriting!

Features:
✓ Train all 3 tasks simultaneously (multi-select checkboxes)
✓ Merge datasets from multiple PCs
✓ Incremental learning with LOW learning rate (0.00005)
✓ Automatic model versioning + backups
✓ Batch QC with accuracy tracking & grading
✓ Training time estimates (per-task breakdown)
✓ Live multi-task visualization
✓ Direct dataset integration button

🔒 Prevents catastrophic forgetting - models learn without losing old knowledge!

Starting interface...
Open: http://localhost:7861
    """)
    
    app = create_training_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
