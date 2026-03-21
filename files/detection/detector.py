"""
Player and Ball Detector Module

Uses YOLOv8 for real-time detection of:
- Players (both opponents)
- Tennis ball
- Rackets (optional)

Optimized for tennis court scenarios with custom fine-tuning support.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import cv2


class PlayerBallDetector:
    """
    YOLOv8-based detector for tennis players and ball.
    
    Features:
    - Multi-object detection (players, ball, rackets)
    - Batch processing for efficiency
    - Confidence filtering
    - GPU acceleration
    """
    
    # Class mapping for tennis-specific objects
    TENNIS_CLASSES = {
        0: 'person',  # Players
        32: 'sports_ball',  # Tennis ball
        38: 'tennis_racket'  # Racket (if using full COCO model)
    }
    
    def __init__(
        self,
        model_name: str = "yolov8l",
        device: torch.device = torch.device('cpu'),
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize detector.
        
        Args:
            model_name: YOLOv8 model variant (yolov8n/s/m/l/x)
            device: Torch device (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
        """
        self.device = device
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLOv8 model
        print(f"   Loading {model_name} model...")
        self.model = YOLO(f'{model_name}.pt')
        
        # Move to device
        if str(device) != 'cpu':
            self.model.to(device)
        
        print(f"   Detector ready on {device}")
    
    def detect_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[Dict[str, List]]:
        """
        Detect objects in batch of frames.
        
        Args:
            frames: List of frames (numpy arrays in RGB format)
        
        Returns:
            List of detection dictionaries, one per frame
            Each dict contains:
                - 'players': List of player bboxes
                - 'ball': List of ball detections (usually 0 or 1)
                - 'rackets': List of racket detections
        """
        # Run inference
        results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        batch_detections = []
        
        for result in results:
            frame_detections = {
                'players': [],
                'ball': [],
                'rackets': []
            }
            
            # Extract boxes
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box data
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detection = {
                        'bbox': xyxy.tolist(),
                        'confidence': conf,
                        'class': cls
                    }
                    
                    # Categorize detection
                    if cls == 0:  # person
                        frame_detections['players'].append(detection)
                    elif cls == 32:  # sports ball
                        frame_detections['ball'].append(detection)
                    elif cls == 38:  # tennis racket
                        frame_detections['rackets'].append(detection)
            
            # Filter to top 2 players (assumes only 2 players in frame)
            if len(frame_detections['players']) > 2:
                frame_detections['players'] = sorted(
                    frame_detections['players'],
                    key=lambda x: x['confidence'],
                    reverse=True
                )[:2]
            
            # Keep only highest confidence ball
            if len(frame_detections['ball']) > 1:
                frame_detections['ball'] = [max(
                    frame_detections['ball'],
                    key=lambda x: x['confidence']
                )]
            
            batch_detections.append(frame_detections)
        
        return batch_detections
    
    def detect_single(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect objects in single frame.
        
        Args:
            frame: Single frame (numpy array)
        
        Returns:
            Detection dictionary
        """
        return self.detect_batch([frame])[0]
    
    def identify_players(
        self,
        detections_sequence: List[Dict],
        court_info: Dict
    ) -> Dict[str, str]:
        """
        Identify which player is which based on court position.
        
        Args:
            detections_sequence: Sequence of frame detections
            court_info: Court detection information
        
        Returns:
            Mapping of player_id to position ('near'/'far')
        """
        # Collect all player positions
        player_positions = []
        
        for frame_det in detections_sequence[:30]:  # Use first 30 frames
            if len(frame_det['players']) == 2:
                # Get y-coordinates (vertical position)
                p1_y = (frame_det['players'][0]['bbox'][1] + 
                       frame_det['players'][0]['bbox'][3]) / 2
                p2_y = (frame_det['players'][1]['bbox'][1] + 
                       frame_det['players'][1]['bbox'][3]) / 2
                
                player_positions.append({
                    'player_0_y': p1_y,
                    'player_1_y': p2_y
                })
        
        if not player_positions:
            return {'0': 'near', '1': 'far'}
        
        # Average positions
        avg_p0_y = np.mean([p['player_0_y'] for p in player_positions])
        avg_p1_y = np.mean([p['player_1_y'] for p in player_positions])
        
        # Player closer to bottom of frame is "near", other is "far"
        if avg_p0_y > avg_p1_y:
            return {'0': 'near', '1': 'far'}
        else:
            return {'0': 'far', '1': 'near'}
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Dict[str, List]
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: Detection dictionary
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw players (blue boxes)
        for i, player in enumerate(detections['players']):
            x1, y1, x2, y2 = [int(c) for c in player['bbox']]
            conf = player['confidence']
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Player {i+1}: {conf:.2f}"
            cv2.putText(
                annotated, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )
        
        # Draw ball (yellow circles)
        for ball in detections['ball']:
            x1, y1, x2, y2 = [int(c) for c in ball['bbox']]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max(5, (x2 - x1) // 2)
            
            cv2.circle(annotated, (center_x, center_y), radius, (0, 255, 255), -1)
            cv2.circle(annotated, (center_x, center_y), radius+2, (0, 0, 0), 2)
        
        # Draw rackets (green boxes)
        for racket in detections['rackets']:
            x1, y1, x2, y2 = [int(c) for c in racket['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return annotated
    
    def fine_tune(
        self,
        train_images: List[str],
        train_labels: List[str],
        epochs: int = 50,
        batch_size: int = 16
    ):
        """
        Fine-tune detector on custom tennis dataset.
        
        Args:
            train_images: List of training image paths
            train_labels: List of label file paths (YOLO format)
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Create YAML config for training
        data_yaml = {
            'train': train_images,
            'val': train_labels,
            'nc': 3,  # number of classes
            'names': ['player', 'ball', 'racket']
        }
        
        # Fine-tune model
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            device=str(self.device)
        )
        
        print("✅ Fine-tuning complete")
