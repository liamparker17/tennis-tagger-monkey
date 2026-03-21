"""
Video Processing Module
Handles video loading, frame extraction, and preprocessing
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path


class VideoProcessor:
    """Process tennis match videos"""
    
    def __init__(self, config: dict):
        self.config = config
        self.video_config = config.get('video', {})
        self.logger = logging.getLogger('VideoProcessor')
        
        self.target_fps = self.video_config.get('target_fps', 30)
        self.frame_skip = self.video_config.get('frame_skip', 1)
        self.resize_height = self.video_config.get('resize_height', 720)
    
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Load video and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        self.logger.info(f"Loading video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate target dimensions
        if self.resize_height and self.resize_height != height:
            aspect_ratio = width / height
            target_width = int(self.resize_height * aspect_ratio)
            target_height = self.resize_height
        else:
            target_width = width
            target_height = height
        
        self.logger.info(f"Video properties: {width}x{height} @ {original_fps:.2f}fps, "
                        f"{total_frames} frames, {duration:.2f}s")
        self.logger.info(f"Target resolution: {target_width}x{target_height}")
        
        # Extract frames
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame skip
            if frame_idx % self.frame_skip == 0:
                # Resize if needed
                if (target_width != width) or (target_height != height):
                    frame = cv2.resize(frame, (target_width, target_height))
                
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        metadata = {
            'path': video_path,
            'original_fps': original_fps,
            'target_fps': self.target_fps,
            'original_resolution': (width, height),
            'target_resolution': (target_width, target_height),
            'total_frames': total_frames,
            'extracted_frames': len(frames),
            'duration': duration,
            'frame_skip': self.frame_skip
        }
        
        self.logger.info(f"Extracted {len(frames)} frames")
        
        return frames, metadata
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for detection
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame
        """
        # Convert to RGB (most models expect RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def generate_visualization(self, frames: List[np.ndarray],
                             detections: List[Dict],
                             tracks: List[Dict],
                             strokes: List[Dict],
                             output_path: str):
        """
        Generate annotated video with detections and analysis
        
        Args:
            frames: List of video frames
            detections: Detection results per frame
            tracks: Tracking results per frame
            strokes: Detected strokes
            output_path: Path to save annotated video
        """
        self.logger.info(f"Generating visualization: {output_path}")
        
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.video_config.get('visualization', {}).get('fps', 30)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_idx, frame in enumerate(frames):
            frame_vis = frame.copy()
            
            # Draw detections
            if frame_idx < len(detections):
                det = detections[frame_idx]
                
                # Draw players
                for player in det.get('players', []):
                    x1, y1, x2, y2 = player['bbox']
                    cv2.rectangle(frame_vis, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame_vis, f"Player {player.get('id', '?')}", 
                              (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw ball
                for ball in det.get('ball', []):
                    x1, y1, x2, y2 = ball['bbox']
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.circle(frame_vis, center, 5, (0, 0, 255), -1)
            
            # Draw frame number
            cv2.putText(frame_vis, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame_vis)
        
        out.release()
        self.logger.info(f"Visualization saved: {output_path}")
