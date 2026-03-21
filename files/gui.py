"""
Graphical User Interface
Simple GUI for tennis tagging system
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QTextEdit, QProgressBar, QCheckBox,
                             QComboBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import TennisTagger, setup_logging, load_config


class ProcessingThread(QThread):
    """Thread for video processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, tagger, video_path, output_csv, visualize):
        super().__init__()
        self.tagger = tagger
        self.video_path = video_path
        self.output_csv = output_csv
        self.visualize = visualize
    
    def run(self):
        try:
            self.progress.emit("Processing video...")
            stats = self.tagger.process_video(
                self.video_path,
                self.output_csv,
                visualize=self.visualize
            )
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class TennisTaggerGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tennis Match Auto-Tagger")
        self.setGeometry(100, 100, 800, 600)
        
        # Load config
        try:
            self.config = load_config('config/config.yaml')
        except Exception as e:
            self.config = {}
            logging.error(f"Could not load config: {e}")
        
        self.tagger = None
        self.processing_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Video input section
        video_group = QGroupBox("Video Input")
        video_layout = QVBoxLayout()
        
        video_input_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Select video file...")
        video_browse_btn = QPushButton("Browse")
        video_browse_btn.clicked.connect(self.browse_video)
        video_input_layout.addWidget(QLabel("Video:"))
        video_input_layout.addWidget(self.video_path_edit)
        video_input_layout.addWidget(video_browse_btn)
        video_layout.addLayout(video_input_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Output section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        output_input_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output CSV file...")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output)
        output_input_layout.addWidget(QLabel("CSV:"))
        output_input_layout.addWidget(self.output_path_edit)
        output_input_layout.addWidget(output_browse_btn)
        output_layout.addLayout(output_input_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Options section
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        self.gpu_checkbox = QCheckBox("Use GPU Acceleration")
        self.gpu_checkbox.setChecked(self.config.get('hardware', {}).get('use_gpu', False))
        options_layout.addWidget(self.gpu_checkbox)
        
        self.visualize_checkbox = QCheckBox("Generate Annotated Video")
        options_layout.addWidget(self.visualize_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; }")
        layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        central_widget.setLayout(layout)
    
    def browse_video(self):
        """Open file dialog for video selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path_edit.setText(file_path)
            
            # Auto-suggest output path
            video_path = Path(file_path)
            output_path = video_path.parent / f"{video_path.stem}_tagged.csv"
            self.output_path_edit.setText(str(output_path))
    
    def browse_output(self):
        """Open file dialog for output selection"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def process_video(self):
        """Start video processing"""
        video_path = self.video_path_edit.text()
        output_path = self.output_path_edit.text()
        
        if not video_path or not output_path:
            self.log("Error: Please select video and output paths")
            return
        
        # Update config
        self.config['hardware']['use_gpu'] = self.gpu_checkbox.isChecked()
        
        # Initialize tagger
        try:
            self.log("Initializing tagger...")
            logger = setup_logging(self.config)
            self.tagger = TennisTagger(self.config, logger)
        except Exception as e:
            self.log(f"Error initializing tagger: {e}")
            return
        
        # Start processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.processing_thread = ProcessingThread(
            self.tagger,
            video_path,
            output_path,
            self.visualize_checkbox.isChecked()
        )
        
        self.processing_thread.progress.connect(self.log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        
        self.processing_thread.start()
    
    def processing_finished(self, stats):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log("\n" + "=" * 50)
        self.log("Processing Complete!")
        self.log("=" * 50)
        self.log(f"Frames processed: {stats['frames_processed']}")
        self.log(f"Duration: {stats['duration_seconds']:.1f} seconds")
        self.log(f"Serves: {stats['num_serves']}")
        self.log(f"Strokes: {stats['num_strokes']}")
        self.log(f"Rallies: {stats['num_rallies']}")
        self.log(f"Output: {stats['output_csv']}")
    
    def processing_error(self, error_msg):
        """Handle processing error"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log(f"\nError: {error_msg}")
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)


def main():
    """Run GUI application"""
    app = QApplication(sys.argv)
    window = TennisTaggerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
