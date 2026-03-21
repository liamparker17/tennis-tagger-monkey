"""
Trainer — model training and fine-tuning with PyTorch.

Provides training from scratch (with synthetic data for now) and
fine-tuning from user corrections. Manages versioned model storage.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ml.classifier import Simple3DCNN, STROKE_CLASSES

logger = logging.getLogger(__name__)


class Trainer:
    """Model training and fine-tuning with versioned storage."""

    def __init__(self, models_dir: str, device: str = "auto"):
        """
        Args:
            models_dir: Root directory where model version subdirs live.
            device: ``"auto"`` picks CUDA when available.
        """
        self.models_dir = Path(models_dir)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("Trainer initialised (models_dir=%s, device=%s)", self.models_dir, self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, pairs: list[dict], config: dict) -> dict:
        """Train a new model version from labelled pairs.

        For now uses synthetic training data — real video feature
        extraction comes in a later phase.

        Args:
            pairs: Training data (video path + annotations).
            config: Hyperparameters — ``epochs`` (default 20),
                     ``lr`` (default 0.001), ``batch_size`` (default 8).

        Returns:
            Dict with ``version``, ``model_path``, and ``metadata``.
        """
        epochs = config.get("epochs", 20)
        lr = config.get("lr", 0.001)
        batch_size = config.get("batch_size", 8)
        num_classes = len(STROKE_CLASSES)

        logger.info("Training new model: epochs=%d, lr=%s, batch_size=%d", epochs, lr, batch_size)

        model = Simple3DCNN(num_classes=num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Synthetic training data (placeholder until real feature extraction)
        num_samples = max(batch_size * 4, len(pairs) if pairs else batch_size * 4)
        synthetic_inputs = torch.randn(num_samples, 3, 16, 112, 112, device=self.device)
        synthetic_labels = torch.randint(0, num_classes, (num_samples,), device=self.device)

        model.train()
        final_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, num_samples, batch_size):
                batch_x = synthetic_inputs[i : i + batch_size]
                batch_y = synthetic_labels[i : i + batch_size]

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            final_loss = avg_loss
            if epoch % 5 == 0 or epoch == 1:
                logger.info("Epoch %d/%d — loss: %.4f", epoch, epochs, avg_loss)

        # Save as new version
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "task_model.pt"
        torch.save(model.state_dict(), str(model_path))

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "final_loss": round(final_loss, 6),
            "num_training_pairs": len(pairs) if pairs else 0,
            "num_classes": num_classes,
            "stroke_classes": STROKE_CLASSES,
            "type": "train",
        }
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info("Saved model %s (loss=%.4f)", version, final_loss)
        return {
            "version": version,
            "model_path": str(model_path),
            "metadata": metadata,
        }

    def fine_tune(self, corrections: list[dict], config: dict) -> dict:
        """Fine-tune the latest model with user corrections.

        Uses a low learning rate (0.00005 default) to prevent
        catastrophic forgetting.

        Args:
            corrections: List of correction dicts with ``corrected``
                         label and optionally ``frame_index``.
            config: Overrides — ``epochs`` (default 10),
                     ``lr`` (default 0.00005).

        Returns:
            Dict with ``version``, ``model_path``, and ``metadata``.
        """
        epochs = config.get("epochs", 10)
        lr = config.get("lr", 0.00005)
        batch_size = config.get("batch_size", 4)
        num_classes = len(STROKE_CLASSES)

        # Find latest version to use as base
        base_version = self._latest_version()
        if base_version is None:
            logger.warning("No base model found — training from scratch instead")
            return self.train([], {**config, "epochs": epochs, "lr": lr})

        base_dir = self.models_dir / base_version
        base_model_path = base_dir / "task_model.pt"

        logger.info("Fine-tuning from %s: epochs=%d, lr=%s, corrections=%d",
                     base_version, epochs, lr, len(corrections))

        # Load base model
        model = Simple3DCNN(num_classes=num_classes).to(self.device)
        state = torch.load(str(base_model_path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Build training data from corrections
        # Map correction labels to class indices; use synthetic input tensors
        # (real frame extraction comes in a later phase)
        label_to_idx = {name: i for i, name in enumerate(STROKE_CLASSES)}
        labels = []
        for c in corrections:
            corrected = c.get("corrected", "")
            idx = label_to_idx.get(corrected)
            if idx is not None:
                labels.append(idx)
            else:
                logger.warning("Unknown correction label %r — skipping", corrected)

        if not labels:
            logger.warning("No valid correction labels — returning base model unchanged")
            return {
                "version": base_version,
                "model_path": str(base_model_path),
                "metadata": {"note": "no valid corrections"},
            }

        num_samples = len(labels)
        synthetic_inputs = torch.randn(num_samples, 3, 16, 112, 112, device=self.device)
        target_labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        model.train()
        final_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, num_samples, batch_size):
                batch_x = synthetic_inputs[i : i + batch_size]
                batch_y = target_labels[i : i + batch_size]

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            final_loss = avg_loss
            if epoch % 5 == 0 or epoch == 1:
                logger.info("Fine-tune epoch %d/%d — loss: %.4f", epoch, epochs, avg_loss)

        # Save as new version
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "task_model.pt"
        torch.save(model.state_dict(), str(model_path))

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_version": base_version,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "final_loss": round(final_loss, 6),
            "num_corrections": len(labels),
            "corrections_used": [c.get("corrected") for c in corrections],
            "num_classes": num_classes,
            "type": "fine_tune",
        }
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info("Saved fine-tuned model %s (base=%s, loss=%.4f)",
                     version, base_version, final_loss)
        return {
            "version": version,
            "model_path": str(model_path),
            "metadata": metadata,
        }

    def get_versions(self) -> list[dict]:
        """List saved model version directories with metadata.

        Returns:
            List of dicts with ``name``, ``path``, and ``metadata``
            (if metadata.json exists) for each version subdirectory.
        """
        versions: list[dict] = []

        if not self.models_dir.exists():
            return versions

        for entry in sorted(self.models_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("v"):
                info: dict = {
                    "name": entry.name,
                    "path": str(entry),
                }
                meta_file = entry / "metadata.json"
                if meta_file.exists():
                    try:
                        info["metadata"] = json.loads(meta_file.read_text())
                    except (json.JSONDecodeError, OSError):
                        pass
                versions.append(info)

        return versions

    def rollback(self, version: str) -> dict:
        """Copy a version's .pt files to the root models directory.

        Args:
            version: Version name (e.g. ``"v2"``).

        Returns:
            Dict with rollback status and files copied.
        """
        version_dir = self.models_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found at {version_dir}")

        copied: list[str] = []
        for pt_file in version_dir.glob("*.pt"):
            dst = self.models_dir / pt_file.name
            shutil.copy2(str(pt_file), str(dst))
            copied.append(pt_file.name)
            logger.info("Rolled back %s -> %s", pt_file, dst)

        return {
            "version": version,
            "files_copied": copied,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_version(self) -> str:
        """Determine the next version number (v1, v2, ...)."""
        existing = self._version_numbers()
        next_num = max(existing, default=0) + 1
        return f"v{next_num}"

    def _latest_version(self) -> Optional[str]:
        """Return the latest version name, or None."""
        existing = self._version_numbers()
        if not existing:
            return None
        return f"v{max(existing)}"

    def _version_numbers(self) -> list[int]:
        """Return sorted list of version numbers found on disk."""
        nums: list[int] = []
        if not self.models_dir.exists():
            return nums
        for entry in self.models_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    nums.append(int(entry.name[1:]))
                except ValueError:
                    continue
        return sorted(nums)
