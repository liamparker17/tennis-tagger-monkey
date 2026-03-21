"""
Model Training Script
Train custom models on historical tennis data
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class TennisDataset(Dataset):
    """Dataset for tennis training data"""
    
    def __init__(self, data_dir: str, task: str):
        self.data_dir = Path(data_dir)
        self.task = task
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load training samples"""
        samples = []
        
        # Find all video/CSV pairs
        for csv_file in self.data_dir.glob('*.csv'):
            video_file = csv_file.with_suffix('.mp4')
            if not video_file.exists():
                video_file = csv_file.with_suffix('.mov')
            
            if video_file.exists():
                samples.append({
                    'video': str(video_file),
                    'csv': str(csv_file)
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Placeholder - would load and process actual data
        return torch.randn(10, 16), torch.tensor(0)


class StrokeClassificationModel(nn.Module):
    """Neural network for stroke classification"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


def train_model(config: dict, task: str, data_dir: str, 
                epochs: int, batch_size: int):
    """Train a model"""
    logger = logging.getLogger('Training')
    logger.info(f"Training {task} model...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and 
                         config.get('hardware', {}).get('use_gpu', False)
                         else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = TennisDataset(data_dir, task)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    if task == 'stroke_classification':
        num_classes = 7  # forehand, backhand, volley, smash, drop, lob, slice
        model = StrokeClassificationModel(160, num_classes)
    else:
        logger.error(f"Unknown task: {task}")
        return
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.get('training', {}).get('learning_rate', 0.001))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.view(inputs.size(0), -1))
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, '
                   f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      f'models/{task}_best.pt')
            logger.info('Saved best model')
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train custom tennis models')
    
    parser.add_argument('--task', type=str, required=True,
                       choices=['stroke_classification', 'serve_detection', 
                               'placement_prediction'],
                       help='Training task')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with training data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train_model(config, args.task, args.data_dir, 
               args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
