"""
Trainer — stub for Phase 5 model training / fine-tuning.

Provides the API surface that the Go bridge will call once
training is implemented.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer:
    """Model training and fine-tuning (Phase 5 — not yet implemented)."""

    def __init__(self, models_dir: str, device: str = "auto"):
        """
        Args:
            models_dir: Root directory where model version subdirs live.
            device: ``"auto"`` picks CUDA when available.
        """
        self.models_dir = Path(models_dir)
        self.device = device
        logger.info("Trainer stub initialised (models_dir=%s)", self.models_dir)

    def train(self, pairs: list[dict], config: dict) -> dict:
        """Train a new model version from labelled pairs.

        Args:
            pairs: Training data (video path + annotations).
            config: Hyperparameters and training settings.

        Raises:
            NotImplementedError: Always — training is Phase 5.
        """
        raise NotImplementedError("Phase 5")

    def fine_tune(self, corrections: list[dict], config: dict) -> dict:
        """Fine-tune an existing model with user corrections.

        Args:
            corrections: List of corrections (frame, old label,
                         new label).
            config: Fine-tuning settings.

        Raises:
            NotImplementedError: Always — fine-tuning is Phase 5.
        """
        raise NotImplementedError("Phase 5")

    def get_versions(self) -> list[dict]:
        """List saved model version directories.

        Returns:
            List of dicts with ``name`` and ``path`` for each version
            subdirectory found under ``models_dir``.
        """
        versions: list[dict] = []

        if not self.models_dir.exists():
            return versions

        for entry in sorted(self.models_dir.iterdir()):
            if entry.is_dir():
                versions.append({
                    "name": entry.name,
                    "path": str(entry),
                })

        return versions
