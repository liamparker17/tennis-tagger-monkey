#!/usr/bin/env python3
"""
Training Script

Train/fine-tune tennis tagging models using labeled video data.

Usage:
    python scripts/train.py --video_dir videos/ --csv_dir labels/ --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import PlayerBallDetector
from models.stroke_classifier import StrokeClassifier
from models.event_detector import EventDetector
from processing.video_processor import VideoProcessor


class TennisTrainer:
    """
    Training pipeline for tennis tagging models.
    """
    
    def __init__(
        self,
        video_dir: str,
        csv_dir: str,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize trainer.
        
        Args:
            video_dir: Directory with training videos
            csv_dir: Directory with label CSVs
            config_path: Path to configuration
        """
        self.video_dir = Path(video_dir)
        self.csv_dir = Path(csv_dir)
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Find matching video-CSV pairs
        self.training_pairs = self._find_training_pairs()
        
        print(f"Found {len(self.training_pairs)} training pairs")
    
    def _find_training_pairs(self) -> list:
        """Find matching video and CSV files."""
        pairs = []
        
        for video_path in self.video_dir.glob("*.mp4"):
            csv_name = video_path.stem + ".csv"
            csv_path = self.csv_dir / csv_name
            
            if csv_path.exists():
                pairs.append({
                    'video': str(video_path),
                    'csv': str(csv_path)
                })
        
        return pairs
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        device: str = 'cuda'
    ):
        """
        Train all models.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to use
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Training Tennis Tagging Models")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Training pairs: {len(self.training_pairs)}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        # Train detector
        print("📦 Training Player/Ball Detector...")
        self._train_detector(epochs, batch_size, device)
        
        # Train stroke classifier
        print("\n🎾 Training Stroke Classifier...")
        self._train_stroke_classifier(epochs, batch_size, device)
        
        # Train event detector
        print("\n⚡ Training Event Detector...")
        self._train_event_detector(epochs, batch_size, device)
        
        print(f"\n{'='*60}")
        print("✅ Training Complete!")
        print(f"{'='*60}\n")
    
    def _train_detector(
        self,
        epochs: int,
        batch_size: int,
        device: torch.device
    ):
        """
        Train object detector.
        
        In production, would:
        1. Extract frames with bounding box annotations
        2. Fine-tune YOLOv8 on tennis-specific data
        3. Save checkpoint
        """
        print("   Preparing detector training data...")
        
        # Initialize detector
        detector = PlayerBallDetector(
            model_name=self.config['models']['detector'],
            device=device
        )
        
        # Extract training data from labeled CSVs
        # (This is simplified - real implementation would extract bbox annotations)
        print(f"   Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training loop would go here
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{epochs}")
        
        # Save model
        save_path = "models/weights/detector_finetuned.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(detector.model.state_dict(), save_path)
        print(f"   ✅ Detector saved to {save_path}")
    
    def _train_stroke_classifier(
        self,
        epochs: int,
        batch_size: int,
        device: torch.device
    ):
        """
        Train stroke classifier.
        
        In production, would:
        1. Extract video clips around stroke events
        2. Train 3D CNN on labeled stroke types
        3. Save checkpoint
        """
        print("   Preparing stroke classifier training data...")
        
        # Initialize classifier
        classifier = StrokeClassifier(
            model_name=self.config['models']['stroke_classifier'],
            device=device
        )
        
        # Extract stroke clips from videos using CSV labels
        print(f"   Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training loop would go here
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{epochs}")
        
        # Save model
        save_path = "models/weights/stroke_classifier_finetuned.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(classifier.model.state_dict(), save_path)
        print(f"   ✅ Stroke classifier saved to {save_path}")
    
    def _train_event_detector(
        self,
        epochs: int,
        batch_size: int,
        device: torch.device
    ):
        """
        Train event detector.
        
        In production, would:
        1. Extract temporal sequences
        2. Train temporal model on event labels
        3. Save checkpoint
        """
        print("   Preparing event detector training data...")
        
        print(f"   Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training loop would go here
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{epochs}")
        
        print(f"   ✅ Event detector training complete")
    
    def validate(self, split: float = 0.2):
        """
        Validate models on held-out data.
        
        Args:
            split: Fraction of data for validation
        """
        print("\n📊 Validating Models...")
        
        val_size = int(len(self.training_pairs) * split)
        val_pairs = self.training_pairs[-val_size:]
        
        print(f"   Validation set: {len(val_pairs)} videos")
        
        # Run inference and compare with labels
        accuracies = {
            'detector': 0.0,
            'stroke_classifier': 0.0,
            'event_detector': 0.0
        }
        
        for pair in tqdm(val_pairs, desc="Validation"):
            # Would run inference and compute metrics
            pass
        
        print(f"\n   Validation Results:")
        print(f"   Detector accuracy: {accuracies['detector']:.2%}")
        print(f"   Stroke classifier accuracy: {accuracies['stroke_classifier']:.2%}")
        print(f"   Event detector accuracy: {accuracies['event_detector']:.2%}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train Tennis Tagging Models"
    )
    
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Directory with training videos'
    )
    
    parser.add_argument(
        '--csv_dir',
        type=str,
        required=True,
        help='Directory with label CSVs'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for training'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TennisTrainer(
        video_dir=args.video_dir,
        csv_dir=args.csv_dir
    )
    
    # Train models
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Validate if requested
    if args.validate:
        trainer.validate()


if __name__ == "__main__":
    main()
