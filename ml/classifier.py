"""
Stroke Classifier — 3D CNN for tennis stroke classification.

Transplanted from files/analysis/stroke_classifier.py.
Provides Simple3DCNN architecture and StrokeClassifier wrapper.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 8 stroke classes
STROKE_CLASSES: list[str] = [
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "serve",
    "smash",
    "drop_shot",
    "lob",
]

# Clip length expected by the model
CLIP_LENGTH = 16

# Minimum confidence to report a classification
DEFAULT_CONF_THRESHOLD = 0.7


class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for stroke classification.

    Architecture: Conv3d 3->64->128->256 with BatchNorm, ReLU,
    AdaptiveAvgPool3d, and a Linear(256, num_classes) head.
    """

    def __init__(self, num_classes: int = 8, temporal_depth: int = CLIP_LENGTH):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, C=3, T, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VideoSwinTransformer(nn.Module):
    """Video Swin Transformer for stroke classification.

    Uses a timm Swin-Tiny backbone for per-frame feature extraction,
    followed by temporal pooling and a classification head.
    Falls back to random features if timm is not installed.
    """

    def __init__(self, num_classes: int = 8, temporal_depth: int = CLIP_LENGTH):
        super().__init__()
        self.temporal_depth = temporal_depth

        try:
            import timm

            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                num_classes=0,
                pretrained=False,
            )
            backbone_dim = self.backbone.num_features
        except ImportError:
            logger.warning("timm not installed — VideoSwinTransformer using random features")
            self.backbone = None
            backbone_dim = 768

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, C=3, T, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        B, C, T, H, W = x.shape

        if self.backbone is not None:
            # Reshape to (B*T, C, H, W) for per-frame feature extraction
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            # Resize to 224x224 if needed
            if H != 224 or W != 224:
                frames = nn.functional.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
            features = self.backbone(frames)  # (B*T, D)
            features = features.reshape(B, T, -1)  # (B, T, D)
        else:
            # Random features fallback
            features = torch.randn(B, T, 768, device=x.device)

        # Temporal pool: (B, T, D) -> (B, D, T) -> pool -> (B, D, 1) -> (B, D)
        features = features.permute(0, 2, 1)
        features = self.temporal_pool(features).squeeze(-1)

        return self.classifier(features)


class StrokeClassifier:
    """Classify player clips into one of 8 stroke types.

    Usage::

        clf = StrokeClassifier("models/stroke_3dcnn.pt")
        results = clf.classify(clips)
        # results: [{"stroke": "forehand", "confidence": 0.92}, ...]
    """

    MODEL_TYPES = ("3dcnn", "swin")

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        confidence_threshold: float = DEFAULT_CONF_THRESHOLD,
        model_type: str = "3dcnn",
    ):
        """
        Args:
            model_path: Path to saved model weights (``.pt``).
                        If the file does not exist, a randomly-initialised
                        model is used (useful for integration tests).
            device: ``"auto"`` picks CUDA when available.
            confidence_threshold: Suppress predictions below this value.
            model_type: ``"3dcnn"`` or ``"swin"``.
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model_type {model_type!r}, expected one of {self.MODEL_TYPES}")

        self.model_type = model_type
        self.conf_threshold = confidence_threshold

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model based on type
        if model_type == "swin":
            self._model = VideoSwinTransformer(num_classes=len(STROKE_CLASSES), temporal_depth=CLIP_LENGTH)
        else:
            self._model = Simple3DCNN(num_classes=len(STROKE_CLASSES), temporal_depth=CLIP_LENGTH)

        # Load weights — fail loudly if missing so callers don't get random predictions
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Stroke classifier weights not found at {model_path}. "
                f"Cannot produce meaningful predictions without trained weights."
            )
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Loaded stroke classifier weights from %s", model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load weights from {model_path}") from exc

        self._model.to(self.device)
        self._model.eval()
        logger.info("StrokeClassifier ready (device=%s, threshold=%.2f)", self.device, self.conf_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, clips: np.ndarray) -> list[dict]:
        """Classify one or more player clips.

        Args:
            clips: Array of shape ``(N, T=16, H, W, C=3)`` with uint8 BGR
                   frames.  Each clip must have exactly ``CLIP_LENGTH``
                   frames.

        Returns:
            One dict per clip::

                {"stroke": str, "confidence": float}

            If the confidence is below the threshold the stroke is
            reported as ``"unknown"``.
        """
        if len(clips) == 0:
            return []

        # Normalise to float32 [0,1] and reshape to (N, C, T, H, W)
        tensor = torch.from_numpy(clips.astype(np.float32) / 255.0)
        tensor = tensor.permute(0, 4, 1, 2, 3)  # NTHWC -> NCTHW
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            confs, idxs = torch.max(probs, dim=1)

        results: list[dict] = []
        for conf_val, idx_val in zip(confs.cpu().numpy(), idxs.cpu().numpy()):
            conf = float(conf_val)
            if conf >= self.conf_threshold:
                stroke = STROKE_CLASSES[int(idx_val)]
            else:
                stroke = "unknown"
            results.append({"stroke": stroke, "confidence": round(conf, 4)})

        return results
