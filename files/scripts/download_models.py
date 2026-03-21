"""
Download Pre-trained Models
Downloads required models for tennis tagging
"""

import os
from pathlib import Path
import logging
import urllib.request
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ModelDownloader')


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def main():
    """Download all required models"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    logger.info("Downloading pre-trained models...")
    
    # YOLOv8 models
    yolo_models = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    }
    
    for model_name, url in yolo_models.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists, skipping")
            continue
        
        logger.info(f"Downloading {model_name}...")
        try:
            download_url(url, str(model_path))
            logger.info(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
    
    logger.info("Model download complete!")
    logger.info(f"Models saved to: {models_dir.absolute()}")


if __name__ == "__main__":
    main()
