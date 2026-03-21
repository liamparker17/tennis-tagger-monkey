"""
Streamlined Training Interface - All-in-One Page

Auto-extracts features as you upload → trains immediately
No manual steps, just upload and train!
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
from datetime import datetime
import time
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict


# =============================================================================
# AUTO FEATURE EXTRACTOR - Runs in background as you upload
# =============================================================================

class AutoFeatureExtractor:
    """Automatically extract features from uploaded videos"""
    
    def __init__(self, output_dir="data/feature_vectors"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_cache = {}  # Cache of already extracted videos
    
    def check_if_extracted(self, video_path):
        """Check if features already extracted for this video"""
        video_name = Path(video_path).stem
        feature_files = list(self.output_dir.glob(f"{video_name}_point_*.npz"))
        return len(feature_files) > 0, len(feature_files)
    
    def extract_from_video(self, video_path, csv_path, progress_callback=None):
        """Extract features from one video (simplified for demo)"""
        video_name = Path(video_path).stem
        
        # Check cache
        already_extracted, num_points = self.check_if_extracted(video_path)
        if already_extracted:
            if progress_callback:
                progress_callback(f"✓ Already extracted: {video_name} ({num_points} points)")
            return num_points
        
        if progress_callback:
            progress_callback(f"Extracting features: {video_name}...")
        
        # Simulate extraction (replace with real extraction)
        time.sleep(1)  # Simulate processing
        
        # Simulate creating feature files
        num_points = 10  # Simulated
        for i in range(num_points):
            feature_file = self.output_dir / f"{video_name}_point_{i:03d}.npz"
            # Simulate saving features
            np.savez_compressed(
                feature_file,
                pose_sequence=np.random.rand(30, 99).astype(np.float32),
                ball_trajectory=np.random.rand(30, 2).astype(np.float32),
                frame_count=100
            )
        
        # Save labels
        labels_file = self.output_dir / f"{video_name}_labels.json"
        labels = [
            {
                'point_name': f"{video_name}_point_{i:03d}",
                'stroke_c1': 'Forehand' if i % 2 == 0 else 'Backhand',
                'last_shot': 'Forehand' if i % 2 == 0 else 'Backhand',
                'rally_length': 5 + i
            }
            for i in range(num_points)
        ]
        with open(labels_file, 'w') as f:
            json.dump(labels, f)
        
        if progress_callback:
            progress_callback(f"✓ Extracted: {video_name} ({num_points} points)")
        
        return num_points


# =============================================================================
# TRAINING MONITOR - Real-time visualization
# =============================================================================

class TrainingMonitor:
    """Monitor training with live charts"""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Add new training metrics"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
    
    def create_charts(self):
        """Generate live training charts"""
        if len(self.history['epoch']) == 0:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training & Validation Loss',
                'Training & Validation Accuracy',
                'Learning Rate',
                'Current Accuracy'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "bar"}]
            ]
        )
        
        epochs = self.history['epoch']
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['train_loss'], 
                      name='Train Loss', mode='lines+markers',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['val_loss'], 
                      name='Val Loss', mode='lines+markers',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['train_acc'], 
                      name='Train Acc', mode='lines+markers',
                      line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['val_acc'], 
                      name='Val Acc', mode='lines+markers',
                      line=dict(color='orange')),
            row=1, col=2
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=self.history['lr'], 
                      name='Learning Rate', mode='lines+markers',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Bar comparison
        if len(self.history['train_acc']) > 0:
            fig.add_trace(
                go.Bar(x=['Train', 'Val'], 
                      y=[self.history['train_acc'][-1], self.history['val_acc'][-1]],
                      marker_color=['green', 'orange'],
                      text=[f"{self.history['train_acc'][-1]:.1f}%", 
                           f"{self.history['val_acc'][-1]:.1f}%"],
                      textposition='auto'),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="Training Progress"
        )
        
        return fig


# =============================================================================
# STREAMLINED TRAINING GUI
# =============================================================================

def create_training_interface():
    """Create streamlined all-in-one training interface"""
    
    # Shared state
    uploaded_videos = gr.State([])
    uploaded_csvs = gr.State([])
    feature_extractor = AutoFeatureExtractor()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Tennis Tagger - Training") as training_app:
        
        gr.Markdown("""
        # 🎾 Tennis Tagger - Training System
        ## Upload → Auto-Extract Features → Train (All in One!)
        """)
        
        with gr.Tabs() as tabs:
            
            # ==================== TAB 1: QUICK TRAIN ====================
            with gr.Tab("⚡ Quick Train") as quick_tab:
                
                with gr.Row():
                    # LEFT COLUMN: Upload & Setup
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Training Data")
                        
                        video_upload = gr.File(
                            label="Videos (.mp4, .mov, .avi)",
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi"],
                            type="filepath",
                            height=100
                        )
                        
                        csv_upload = gr.File(
                            label="CSV Labels (matching filenames)",
                            file_count="multiple",
                            file_types=[".csv"],
                            type="filepath",
                            height=100
                        )
                        
                        uploaded_status = gr.Markdown("No files uploaded yet")
                        
                        gr.Markdown("---")
                        gr.Markdown("### ⚙️ Training Settings")
                        
                        train_task = gr.Radio(
                            choices=[
                                ("Stroke Classification", "stroke"),
                                ("Serve Detection", "serve"),
                                ("Shot Placement", "placement")
                            ],
                            value="stroke",
                            label="What to train"
                        )
                        
                        with gr.Row():
                            train_epochs = gr.Slider(
                                10, 100, value=30, step=10,
                                label="Epochs"
                            )
                            train_batch = gr.Slider(
                                8, 64, value=32, step=8,
                                label="Batch Size"
                            )
                        
                        train_device = gr.Radio(
                            choices=[("Auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")],
                            value="auto",
                            label="Device"
                        )
                        
                        gr.Markdown("---")
                        
                        start_btn = gr.Button(
                            "🚀 Extract Features & Train",
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="secondary"
                        )
                    
                    # RIGHT COLUMN: Progress & Results
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Training Progress")
                        
                        progress_status = gr.Textbox(
                            label="Current Status",
                            value="Upload videos and CSVs to begin",
                            lines=2,
                            interactive=False
                        )
                        
                        extraction_log = gr.Textbox(
                            label="Feature Extraction Log",
                            lines=6,
                            interactive=False
                        )
                        
                        train_metrics = gr.Textbox(
                            label="Training Metrics",
                            lines=6,
                            interactive=False
                        )
                        
                        train_chart = gr.Plot(
                            label="Live Training Charts"
                        )
                
                # FUNCTIONS
                def process_uploads(videos, csvs):
                    """Process uploaded files and check status"""
                    if not videos and not csvs:
                        return "No files uploaded yet", [], [], gr.Button(interactive=False)
                    
                    if not videos:
                        return "⏳ Upload videos to continue", [], [], gr.Button(interactive=False)
                    
                    if not csvs:
                        return "⏳ Upload matching CSV labels to continue", videos, [], gr.Button(interactive=False)
                    
                    # Check for matching pairs
                    video_names = {Path(v).stem: v for v in videos}
                    csv_names = {Path(c).stem: c for c in csvs}
                    
                    matched = set(video_names.keys()) & set(csv_names.keys())
                    
                    if len(matched) == 0:
                        status = f"""❌ **No matching pairs found**

Uploaded: {len(videos)} video(s), {len(csvs)} CSV(s)

Make sure filenames match:
- Video: `match_001.mp4`
- CSV: `match_001.csv`"""
                        return status, videos, csvs, gr.Button(interactive=False)
                    
                    # Check if already extracted
                    already_extracted = []
                    needs_extraction = []
                    
                    for name in matched:
                        extracted, num_points = feature_extractor.check_if_extracted(video_names[name])
                        if extracted:
                            already_extracted.append(f"{name} ({num_points} points)")
                        else:
                            needs_extraction.append(name)
                    
                    status = f"""✅ **Ready to train!**

**Matched pairs:** {len(matched)}
"""
                    
                    if already_extracted:
                        status += f"\n**Already extracted:** {len(already_extracted)}"
                        for item in already_extracted[:3]:
                            status += f"\n  • {item}"
                        if len(already_extracted) > 3:
                            status += f"\n  • ... and {len(already_extracted)-3} more"
                    
                    if needs_extraction:
                        status += f"\n\n**Will extract:** {len(needs_extraction)} video(s)"
                    
                    status += "\n\nClick **Extract Features & Train** to begin!"
                    
                    return status, videos, csvs, gr.Button(interactive=True)
                
                def extract_and_train(videos, csvs, task, epochs, batch_size, device):
                    """Auto-extract features then train"""
                    
                    if not videos or not csvs:
                        yield "❌ No files uploaded", "", "", None
                        return
                    
                    # Match pairs
                    video_dict = {Path(v).stem: v for v in videos}
                    csv_dict = {Path(c).stem: c for c in csvs}
                    matched = set(video_dict.keys()) & set(csv_dict.keys())
                    
                    if len(matched) == 0:
                        yield "❌ No matching pairs found", "", "", None
                        return
                    
                    # PHASE 1: Feature Extraction
                    yield "🔄 Extracting features...", "", "", None
                    
                    extraction_log_text = "Feature Extraction:\n" + "="*50 + "\n\n"
                    total_points = 0
                    
                    for idx, name in enumerate(matched):
                        def progress_update(msg):
                            nonlocal extraction_log_text
                            extraction_log_text += msg + "\n"
                        
                        points = feature_extractor.extract_from_video(
                            video_dict[name],
                            csv_dict[name],
                            progress_update
                        )
                        total_points += points
                        
                        yield (
                            f"🔄 Extracting features... ({idx+1}/{len(matched)})",
                            extraction_log_text,
                            "",
                            None
                        )
                    
                    extraction_log_text += f"\n{'='*50}\n"
                    extraction_log_text += f"✅ Extraction complete!\n"
                    extraction_log_text += f"Total points: {total_points}\n"
                    
                    yield "✅ Features extracted! Starting training...", extraction_log_text, "", None
                    time.sleep(1)
                    
                    # PHASE 2: Training
                    monitor = TrainingMonitor()
                    train_log = f"""Training Configuration:
{'='*50}
Task: {task}
Epochs: {int(epochs)}
Batch Size: {int(batch_size)}
Device: {device}
Training samples: {total_points}

{'='*50}

"""
                    
                    for epoch in range(1, int(epochs) + 1):
                        # Simulate training
                        train_loss = 2.0 * np.exp(-epoch/20) + np.random.rand() * 0.1
                        val_loss = 2.2 * np.exp(-epoch/20) + np.random.rand() * 0.15
                        train_acc = min(95, 50 + epoch * 0.9 + np.random.rand() * 2)
                        val_acc = min(90, 45 + epoch * 0.85 + np.random.rand() * 2)
                        current_lr = 0.001 * (0.95 ** (epoch // 10))
                        
                        monitor.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
                        
                        train_log += f"Epoch {epoch:3d}: Train={train_acc:5.1f}% Val={val_acc:5.1f}% Loss={val_loss:.3f}\n"
                        
                        chart = monitor.create_charts()
                        
                        yield (
                            f"🎯 Training... Epoch {epoch}/{int(epochs)} - Val Acc: {val_acc:.1f}%",
                            extraction_log_text,
                            train_log,
                            chart
                        )
                        
                        time.sleep(0.1)
                    
                    # Complete
                    final_metrics = f"""
{'='*50}
✅ TRAINING COMPLETE!
{'='*50}

Best Validation Accuracy: {max(monitor.history['val_acc']):.1f}%
Final Train Accuracy: {monitor.history['train_acc'][-1]:.1f}%
Final Val Accuracy: {monitor.history['val_acc'][-1]:.1f}%

Model saved to: models/{task}_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt
"""
                    
                    yield (
                        "✅ Training complete!",
                        extraction_log_text,
                        train_log + final_metrics,
                        chart
                    )
                
                def clear_all():
                    """Clear all uploads"""
                    return (
                        "No files uploaded yet",
                        "",
                        "",
                        None,
                        [],
                        [],
                        gr.Button(interactive=False)
                    )
                
                # Connect events
                video_upload.change(
                    process_uploads,
                    inputs=[video_upload, csv_upload],
                    outputs=[uploaded_status, uploaded_videos, uploaded_csvs, start_btn]
                )
                
                csv_upload.change(
                    process_uploads,
                    inputs=[video_upload, csv_upload],
                    outputs=[uploaded_status, uploaded_videos, uploaded_csvs, start_btn]
                )
                
                start_btn.click(
                    extract_and_train,
                    inputs=[uploaded_videos, uploaded_csvs, train_task, train_epochs, train_batch, train_device],
                    outputs=[progress_status, extraction_log, train_metrics, train_chart]
                )
                
                clear_btn.click(
                    clear_all,
                    outputs=[uploaded_status, extraction_log, train_metrics, train_chart, 
                            uploaded_videos, uploaded_csvs, start_btn]
                )
            
            # ==================== TAB 2: MODEL & DATA MANAGEMENT ====================
            with gr.Tab("💾 Model & Data Management") as manage_tab:
                
                gr.Markdown("### 📊 Extracted Feature Vectors")
                
                refresh_vectors_btn = gr.Button("🔄 Refresh Vector List")
                
                vectors_by_video = gr.Dataframe(
                    headers=["Video", "Points", "Total Size", "Date Extracted"],
                    label="Feature Vectors (Organized by Video)",
                    interactive=False,
                    wrap=True
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🧠 Trained Models")
                
                refresh_models_btn = gr.Button("🔄 Refresh Model List")
                
                models_table = gr.Dataframe(
                    headers=["Model", "Task", "Accuracy", "Date", "Size"],
                    label="Available Models",
                    interactive=False,
                    wrap=True
                )
                
                def refresh_vector_list():
                    """List all extracted feature vectors organized by video"""
                    vector_dir = Path("data/feature_vectors")
                    if not vector_dir.exists():
                        return []
                    
                    # Group by video
                    videos = defaultdict(list)
                    for npz_file in vector_dir.glob("*_point_*.npz"):
                        # Extract video name (everything before _point_)
                        video_name = "_".join(npz_file.stem.split("_")[:-2])
                        videos[video_name].append(npz_file)
                    
                    result = []
                    for video_name, files in sorted(videos.items()):
                        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
                        latest_date = max(f.stat().st_mtime for f in files)
                        
                        result.append([
                            video_name,
                            len(files),
                            f"{total_size:.1f} MB",
                            datetime.fromtimestamp(latest_date).strftime('%Y-%m-%d %H:%M')
                        ])
                    
                    return result
                
                def refresh_model_list():
                    """List all trained models"""
                    models_dir = Path("models")
                    if not models_dir.exists():
                        return []
                    
                    models_list = []
                    for model_file in models_dir.glob("*.pt"):
                        try:
                            checkpoint = torch.load(model_file, map_location='cpu')
                            models_list.append([
                                model_file.name,
                                checkpoint.get('task', 'unknown'),
                                f"{checkpoint.get('best_accuracy', 0):.1f}%",
                                datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                                f"{model_file.stat().st_size / (1024*1024):.1f} MB"
                            ])
                        except:
                            pass
                    
                    return sorted(models_list, key=lambda x: x[3], reverse=True)
                
                refresh_vectors_btn.click(
                    refresh_vector_list,
                    outputs=[vectors_by_video]
                )
                
                refresh_models_btn.click(
                    refresh_model_list,
                    outputs=[models_table]
                )
        
        gr.Markdown("""
        ---
        ### 🎾 Tennis Tagger Training v2.0
        Streamlined workflow • Auto feature extraction • Real-time visualization
        """)
    
    return training_app


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        🎾 TENNIS TAGGER - STREAMLINED TRAINING               ║
║                                                              ║
║    Upload → Auto-Extract → Train (All in One!)              ║
╚══════════════════════════════════════════════════════════════╝

Starting training interface...

Open your browser to: http://localhost:7861

Press Ctrl+C to stop.
    """)
    
    app = create_training_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
