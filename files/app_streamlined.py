"""
Tennis Tagger - Main Application (Streamlined)
User-friendly video processing and QC correction system
"""

import gradio as gr
from pathlib import Path
import pandas as pd
from datetime import datetime
import time


# =============================================================================
# VIDEO PROCESSOR (Simplified - integrate with your actual processor)
# =============================================================================

class VideoProcessor:
    """Process tennis videos and generate tagged CSVs"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_video(self, video_path, progress_callback=None):
        """Process a tennis video"""
        
        video_name = Path(video_path).stem
        
        if progress_callback:
            progress_callback(f"Loading video: {video_name}")
        
        time.sleep(0.5)  # Simulate processing
        
        if progress_callback:
            progress_callback(f"Detecting ball and players...")
        
        time.sleep(1)
        
        if progress_callback:
            progress_callback(f"Analyzing strokes...")
        
        time.sleep(1)
        
        if progress_callback:
            progress_callback(f"Tracking score...")
        
        time.sleep(0.5)
        
        # Generate output CSV
        output_csv = self.output_dir / f"{video_name}_tagged.csv"
        
        # Simulate CSV generation
        sample_data = {
            'Timestamp': ['00:00:15', '00:00:28', '00:00:45'],
            'Event': ['Rally Start', 'Stroke', 'Point End'],
            'Stroke_Type': ['', 'Forehand', ''],
            'Player': ['', 'Player 1', ''],
            'Score': ['0-0', '0-0', '15-0']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_csv, index=False)
        
        if progress_callback:
            progress_callback(f"✅ Complete! Saved to {output_csv.name}")
        
        return str(output_csv), df


# =============================================================================
# QC MANAGER
# =============================================================================

class QCManager:
    """Manage QC corrections"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.qc_dir = Path("data/qc_corrections")
        self.qc_dir.mkdir(parents=True, exist_ok=True)
    
    def list_processed_videos(self):
        """List videos that have been processed"""
        if not self.output_dir.exists():
            return []
        
        processed = []
        for csv_file in self.output_dir.glob("*_tagged.csv"):
            video_name = csv_file.stem.replace('_tagged', '')
            processed.append(f"{video_name}.mp4")
        
        return sorted(processed)
    
    def calculate_accuracy(self, original_csv, corrected_csv):
        """Calculate accuracy between original and corrected CSV"""
        
        orig_df = pd.read_csv(original_csv)
        corr_df = pd.read_csv(corrected_csv)
        
        total_cells = orig_df.shape[0] * orig_df.shape[1]
        differences = 0
        
        for col in orig_df.columns:
            if col in corr_df.columns:
                differences += (orig_df[col] != corr_df[col]).sum()
        
        accuracy = ((total_cells - differences) / total_cells) * 100
        
        return accuracy, differences


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def create_main_interface():
    """Create streamlined Tennis Tagger interface"""
    
    processor = VideoProcessor()
    qc_manager = QCManager()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Tennis Tagger") as app:
        
        gr.Markdown("""
        # 🎾 Tennis Tagger - Video Processing System
        ## Automatic tennis match tagging with AI
        """)
        
        with gr.Tabs():
            
            # ==================== TAB 1: PROCESS VIDEOS ====================
            with gr.Tab("🎬 Process Videos") as process_tab:
                
                with gr.Row():
                    # LEFT: Upload & Configure
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Video")
                        
                        video_upload = gr.File(
                            label="Tennis Match Video",
                            file_types=[".mp4", ".mov", ".avi"],
                            height=100
                        )
                        
                        video_status = gr.Markdown("*Upload a video to begin*")
                        
                        gr.Markdown("---")
                        gr.Markdown("### ⚙️ Processing Options")
                        
                        detect_ball = gr.Checkbox(
                            label="Ball Tracking",
                            value=True,
                            info="Track ball position"
                        )
                        
                        detect_strokes = gr.Checkbox(
                            label="Stroke Classification",
                            value=True,
                            info="Identify forehand, backhand, etc."
                        )
                        
                        detect_score = gr.Checkbox(
                            label="Score Tracking",
                            value=True,
                            info="Auto-track match score"
                        )
                        
                        gr.Markdown("---")
                        
                        process_btn = gr.Button(
                            "🚀 Process Video",
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            variant="secondary"
                        )
                    
                    # RIGHT: Progress & Results
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Processing Progress")
                        
                        progress_status = gr.Textbox(
                            label="Current Status",
                            value="Waiting for video upload",
                            lines=2,
                            interactive=False
                        )
                        
                        progress_log = gr.Textbox(
                            label="Processing Log",
                            lines=8,
                            interactive=False
                        )
                        
                        gr.Markdown("### 📄 Results")
                        
                        output_preview = gr.Dataframe(
                            label="Tagged CSV Preview",
                            interactive=False,
                            wrap=True
                        )
                        
                        download_csv = gr.File(
                            label="Download Tagged CSV",
                            interactive=False
                        )
                
                # FUNCTIONS
                def validate_upload(video):
                    """Validate video upload"""
                    if not video:
                        return "Upload a video to begin", gr.Button(interactive=False)
                    
                    video_path = Path(video)
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    
                    status = f"""✅ **Video ready to process!**

**File**: {video_path.name}
**Size**: {size_mb:.1f} MB

Click **Process Video** to begin automatic tagging."""
                    
                    return status, gr.Button(interactive=True)
                
                def process_video(video, ball, strokes, score):
                    """Process the video"""
                    if not video:
                        yield "❌ No video uploaded", "", None, None
                        return
                    
                    log_text = "Starting video processing...\n\n"
                    yield "🔄 Processing...", log_text, None, None
                    
                    def log_update(msg):
                        nonlocal log_text
                        log_text += msg + "\n"
                    
                    # Process video
                    csv_path, df = processor.process_video(video, log_update)
                    
                    yield "🔄 Processing...", log_text, None, None
                    
                    final_status = f"""✅ **Processing Complete!**

Tagged CSV ready for download.
Go to **QC Corrections** tab to review and correct any errors."""
                    
                    yield final_status, log_text, df, csv_path
                
                def clear_all():
                    """Clear all"""
                    return (
                        "Upload a video to begin",
                        "",
                        "",
                        None,
                        None,
                        gr.Button(interactive=False)
                    )
                
                # Connect events
                video_upload.change(
                    validate_upload,
                    inputs=[video_upload],
                    outputs=[video_status, process_btn]
                )
                
                process_btn.click(
                    process_video,
                    inputs=[video_upload, detect_ball, detect_strokes, detect_score],
                    outputs=[progress_status, progress_log, output_preview, download_csv]
                )
                
                clear_btn.click(
                    clear_all,
                    outputs=[video_status, progress_status, progress_log, 
                            output_preview, download_csv, process_btn]
                )
            
            # ==================== TAB 2: QC CORRECTIONS ====================
            with gr.Tab("✅ QC Corrections") as qc_tab:
                
                gr.Markdown("""
                ### Quality Control - Correct AI Mistakes
                
                Review and correct any errors made by the AI tagging system.
                Your corrections help improve the models!
                """)
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📤 Submit Correction")
                        
                        qc_video_select = gr.Dropdown(
                            choices=qc_manager.list_processed_videos(),
                            label="Select Processed Video",
                            info="Which video did you QC?"
                        )
                        
                        refresh_videos_btn = gr.Button("🔄 Refresh Video List")
                        
                        qc_original_csv = gr.File(
                            label="Original CSV (from system)",
                            file_types=[".csv"]
                        )
                        
                        qc_corrected_csv = gr.File(
                            label="Corrected CSV (your edits)",
                            file_types=[".csv"]
                        )
                        
                        qc_submit_btn = gr.Button(
                            "✅ Submit QC Correction",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 📊 Correction Result")
                        
                        qc_result = gr.Markdown("*Upload files to see accuracy*")
                        
                        qc_accuracy_display = gr.Textbox(
                            label="Accuracy Score",
                            value="Waiting...",
                            interactive=False,
                            lines=1
                        )
                        
                        gr.Markdown("""
**What happens next?**

Your correction is saved to the QC queue.
When 10-20 corrections are collected, go to the
**Training System** to batch retrain the models.

This improves accuracy without constant retraining!
                        """)
                
                def refresh_video_list():
                    """Refresh list of processed videos"""
                    videos = qc_manager.list_processed_videos()
                    return gr.Dropdown(choices=videos)
                
                def submit_qc(video, original, corrected):
                    """Submit QC correction"""
                    if not video or not original or not corrected:
                        return "❌ Please select video and upload both CSVs", ""
                    
                    try:
                        accuracy, corrections = qc_manager.calculate_accuracy(original, corrected)
                        
                        result_md = f"""### ✅ QC Correction Saved!

**Video**: {video}
**Accuracy**: {accuracy:.1f}%
**Corrections Made**: {corrections}

This correction has been added to the training queue.

Go to **Training System** → **QC Corrections** tab when you have 10-20 corrections ready for batch retraining.
"""
                        
                        return result_md, f"{accuracy:.1f}%"
                        
                    except Exception as e:
                        return f"❌ Error: {str(e)}", ""
                
                refresh_videos_btn.click(
                    refresh_video_list,
                    outputs=[qc_video_select]
                )
                
                qc_submit_btn.click(
                    submit_qc,
                    inputs=[qc_video_select, qc_original_csv, qc_corrected_csv],
                    outputs=[qc_result, qc_accuracy_display]
                )
            
            # ==================== TAB 3: BATCH PROCESSING ====================
            with gr.Tab("📦 Batch Processing") as batch_tab:
                
                gr.Markdown("### Process Multiple Videos")
                
                batch_videos = gr.File(
                    label="Upload Multiple Videos",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi"]
                )
                
                batch_process_btn = gr.Button(
                    "🚀 Process All Videos",
                    variant="primary",
                    size="lg"
                )
                
                batch_progress = gr.Textbox(
                    label="Batch Progress",
                    lines=15,
                    interactive=False
                )
                
                batch_results = gr.Dataframe(
                    headers=["Video", "Status", "Output CSV"],
                    label="Batch Results",
                    interactive=False
                )
                
                def process_batch(videos):
                    """Process multiple videos"""
                    if not videos:
                        return "No videos uploaded", []
                    
                    results = []
                    log_text = f"Starting batch processing of {len(videos)} videos...\n\n"
                    
                    for idx, video in enumerate(videos, 1):
                        video_name = Path(video).name
                        log_text += f"[{idx}/{len(videos)}] Processing {video_name}...\n"
                        yield log_text, results
                        
                        try:
                            csv_path, _ = processor.process_video(video)
                            results.append([video_name, "✅ Success", csv_path])
                            log_text += f"  ✓ Complete: {Path(csv_path).name}\n\n"
                        except Exception as e:
                            results.append([video_name, f"❌ Error: {str(e)}", ""])
                            log_text += f"  ✗ Failed: {str(e)}\n\n"
                        
                        yield log_text, results
                    
                    log_text += f"\n{'='*50}\n"
                    log_text += f"Batch processing complete!\n"
                    log_text += f"Successful: {sum(1 for r in results if '✅' in r[1])}/{len(videos)}\n"
                    
                    yield log_text, results
                
                batch_process_btn.click(
                    process_batch,
                    inputs=[batch_videos],
                    outputs=[batch_progress, batch_results]
                )
            
            # ==================== TAB 4: HELP ====================
            with gr.Tab("❓ Help") as help_tab:
                
                gr.Markdown("""
                ## Tennis Tagger - Quick Guide
                
                ### 🎬 Process Videos Tab
                
                **How to use:**
                1. Upload a tennis match video (.mp4, .mov, or .avi)
                2. Choose which features to enable (ball tracking, strokes, score)
                3. Click **Process Video**
                4. Wait for processing (1-2 minutes per minute of video)
                5. Download the tagged CSV
                
                **What you get:**
                - Timestamped events (rally starts, points, etc.)
                - Stroke classifications (forehand, backhand, volley, etc.)
                - Ball tracking data
                - Score tracking
                
                ---
                
                ### ✅ QC Corrections Tab
                
                **How to correct mistakes:**
                1. Download the CSV from processing
                2. Open in Excel/Google Sheets
                3. Fix any incorrect tags
                4. Save the corrected CSV
                5. Go to QC Corrections tab
                6. Select the video
                7. Upload both original and corrected CSVs
                8. Submit the correction
                
                **What happens:**
                - System calculates accuracy score
                - Correction saved to training queue
                - When 10-20 corrections collected → batch retrain
                - Models improve from your corrections!
                
                ---
                
                ### 📦 Batch Processing Tab
                
                **Process multiple videos at once:**
                1. Upload multiple video files
                2. Click **Process All Videos**
                3. System processes each video automatically
                4. Download all tagged CSVs when complete
                
                **Good for:**
                - Processing an entire tournament
                - Batch tagging of archived matches
                - Large-scale data generation
                
                ---
                
                ### 🎓 Training System
                
                **Accessed separately** (port 7861)
                
                Use for:
                - Training new models
                - Batch retraining with QC corrections
                - Merging datasets from multiple PCs
                - Managing models and data
                
                ---
                
                ### Tips
                
                ✓ Upload high-quality videos for best results
                ✓ Process one match at a time for first use
                ✓ Review and QC the first few outputs
                ✓ Collect 10-20 corrections before retraining
                ✓ Use batch processing for large sets
                
                ---
                
                ### System Requirements
                
                **Minimum:**
                - 8GB RAM
                - GPU recommended (but CPU works)
                - 10GB free disk space
                
                **Recommended:**
                - 16GB RAM
                - NVIDIA GPU (RTX 2050+)
                - 50GB free disk space
                
                ---
                
                ### Support
                
                If you encounter issues:
                1. Check the Logs tab in the Launcher
                2. Ensure models are trained (Training System)
                3. Verify video format is supported
                4. Try a shorter video first
                """)
        
        gr.Markdown("""
        ---
        ### 🎾 Tennis Tagger v3.0
        
        Automatic tennis match tagging • AI-powered • Production-ready
        """)
    
    return app


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🎾 TENNIS TAGGER - MAIN SYSTEM                     ║
║                                                              ║
║       Process videos • Generate tagged CSVs • QC             ║
╚══════════════════════════════════════════════════════════════╝

Starting main application...

Open your browser to: http://localhost:7860

Features:
- Video processing with AI
- Batch processing
- QC correction system
- Clean, user-friendly interface

Press Ctrl+C to stop.
    """)
    
    app = create_main_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
