"""TrackNet v2 — Dense ball detection via heatmap regression.

U-Net style encoder-decoder that takes 3 background-subtracted BGR frames
(9 channels) and outputs a probability heatmap for the ball position.

Components:
    TrackNetV2          — PyTorch nn.Module (encoder-decoder)
    BackgroundSubtractor — Median-based static background removal
    TrackNetDetector    — Inference wrapper with frame buffering and peak detection
"""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

logger = logging.getLogger(__name__)

# Peak detection default threshold
_PEAK_THRESHOLD = 0.5

# Frame stride for triplet building: [N-4, N-2, N]
_TRIPLET_STRIDE = 2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _conv_block(in_ch: int, out_ch: int) -> "nn.Sequential":
    """VGG-style Conv3x3 + BN + ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNetV2(nn.Module):
    """U-Net style TrackNet v2 encoder-decoder.

    Input shape:  (B, 9, H, W)  — 3 BGR frames stacked channel-wise after
                                   background subtraction
    Output shape: (B, 1, H, W)  — sigmoid probability heatmap

    Approximate parameter count: ~2.5 M
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _conv_block(9, 64)       # -> (B, 64, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)       # -> (B, 64, H/2, W/2)

        self.enc2 = _conv_block(64, 128)      # -> (B, 128, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2, 2)       # -> (B, 128, H/4, W/4)

        self.enc3 = _conv_block(128, 256)     # -> (B, 256, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2, 2)       # -> (B, 256, H/8, W/8)

        self.enc4 = _conv_block(256, 512)     # -> (B, 512, H/8, W/8)

        # Decoder
        # Stage 1: 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _conv_block(512, 256)     # 256 (up) + 256 (skip enc3)

        # Stage 2: 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _conv_block(256, 128)     # 128 (up) + 128 (skip enc2)

        # Stage 3: 128 -> 64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _conv_block(128, 64)      # 64 (up) + 64 (skip enc1)

        # Output head
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _pad_to_match(x: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        """Pad x (upsampled) to match the spatial size of target (skip tensor).

        ConvTranspose2d with odd input dimensions can produce output that is
        1 pixel short on H or W.  We pad symmetrically on the right/bottom.
        """
        dh = target.size(2) - x.size(2)
        dw = target.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            # F.pad order: (left, right, top, bottom)
            x = F.pad(x, (0, dw, 0, dh))
        return x

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Encoder
        e1 = self.enc1(x)           # (B, 64, H, W)
        e2 = self.enc2(self.pool1(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool3(e3))   # (B, 512, H/8, W/8)

        # Decoder
        d3 = self.up3(e4)                           # (B, 256, H/4, W/4)
        d3 = self._pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                           # (B, 128, H/2, W/2)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                           # (B, 64, H, W)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 64, H, W)

        return torch.sigmoid(self.out_conv(d1))     # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------


class BackgroundSubtractor:
    """Static background model using median of reference frames.

    Usage::

        bg = BackgroundSubtractor()
        bg.build_reference(frames)          # list of np.ndarray (BGR)
        diff = bg.subtract(new_frame)       # absolute difference
    """

    def __init__(self) -> None:
        self._reference: Optional[np.ndarray] = None

    def build_reference(self, frames: list) -> None:
        """Compute per-pixel median across frames as the background.

        Args:
            frames: List of BGR ``np.ndarray`` frames, all same shape (H, W, 3).
                    At least 1 frame required; more frames yield a better median.
        """
        if not frames:
            raise ValueError("build_reference requires at least one frame")

        stack = np.stack(frames, axis=0).astype(np.float32)
        self._reference = np.median(stack, axis=0).astype(np.uint8)
        logger.debug(
            "BackgroundSubtractor: reference built from %d frames, shape=%s",
            len(frames),
            self._reference.shape,
        )

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """Return absolute difference between frame and reference background.

        Args:
            frame: BGR ``np.ndarray`` (H, W, 3), same resolution as reference.

        Returns:
            Absolute-difference image (H, W, 3) uint8.

        Raises:
            RuntimeError: If ``build_reference`` has not been called yet.
        """
        if self._reference is None:
            raise RuntimeError(
                "BackgroundSubtractor: call build_reference() before subtract()"
            )
        return cv2.absdiff(frame, self._reference)

    @property
    def is_ready(self) -> bool:
        """True once a reference has been built."""
        return self._reference is not None


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------


class TrackNetDetector:
    """Inference wrapper for TrackNetV2 with frame buffering and peak detection.

    Buffers incoming frames and assembles every-other-frame triplets
    [N-4, N-2, N] before running the model, matching the original TrackNet
    temporal receptive field design.

    Args:
        model_path: Optional path to saved weights (.pt).  If ``None`` the
                    model is initialised with random weights (useful for
                    smoke tests / parameter counting).
        device:     ``"auto"`` (default) selects CUDA when available,
                    ``"cpu"`` forces CPU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.model: Optional["TrackNetV2"] = None
        self.device_str = "cpu"

        if not _TORCH_OK:
            logger.warning("torch not installed — TrackNetDetector disabled")
            return

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device_str = "cuda"
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("TrackNet GPU: %s (%.1f GB)", name, mem)
            else:
                logger.info("TrackNet: no CUDA GPU — using CPU")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device_str = "cuda"
            else:
                logger.warning("TrackNet: CUDA requested but not available, falling back to CPU")
        elif device == "cpu":
            self.device_str = "cpu"
        else:
            logger.warning("TrackNet: unknown device %r, using CPU", device)

        self._device = torch.device(self.device_str)

        # Build and optionally load model
        try:
            self.model = TrackNetV2().to(self._device)
            self.model.eval()

            if model_path is not None:
                state = torch.load(model_path, map_location=self._device)
                # Accept both raw state_dicts and checkpoint dicts
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state)
                logger.info("TrackNet: loaded weights from %s", model_path)
            else:
                logger.info(
                    "TrackNet: no weights path given — using random weights "
                    "(suitable for testing only)"
                )

            # Warmup: single forward pass with a dummy input at expected resolution
            self._warmup(height=360, width=640)
        except Exception:
            logger.exception("TrackNet: failed to initialise model")
            self.model = None

        # Frame buffer — holds raw BGR frames; we need indices 0, 2, 4 (N-4, N-2, N)
        # deque maxlen=5 keeps the last 5 frames
        self._frame_buffer: deque = deque(maxlen=5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_frame(self, frame: np.ndarray) -> Optional[dict]:
        """Push one BGR frame into the buffer and run inference when ready.

        A detection is attempted only once at least 5 frames have been
        buffered (so that positions [0], [2], [4] — i.e. N-4, N-2, N —
        are all available).

        Args:
            frame: BGR ``np.ndarray`` (H, W, 3).

        Returns:
            ``{"x": float, "y": float, "confidence": float}`` when a ball is
            detected, or ``None`` when the buffer is not yet full or no ball
            is found above the threshold.
        """
        self._frame_buffer.append(frame)

        if len(self._frame_buffer) < 5:
            return None

        buf = list(self._frame_buffer)
        # Triplet: oldest (N-4), middle (N-2), newest (N)
        triplet = [buf[0], buf[2], buf[4]]
        diff_triplet = self._build_diff_triplet(triplet)

        results = self.detect_batch([diff_triplet])
        return results[0] if results else None

    def detect_batch(self, diff_triplets: list) -> list:
        """Run batch inference on pre-built difference triplets.

        Args:
            diff_triplets: List of triplets, each a list/tuple of 3 BGR
                           ``np.ndarray`` frames that have already been
                           background-subtracted (or raw frames when no
                           background subtractor is in use).

        Returns:
            List of the same length as ``diff_triplets``.  Each element is
            ``{"x": float, "y": float, "confidence": float}`` or ``None``.
        """
        if self.model is None or not diff_triplets:
            return [None] * len(diff_triplets)

        try:
            tensors = [self._triplet_to_tensor(t) for t in diff_triplets]
            batch = torch.stack(tensors, dim=0).to(self._device)  # (B, 9, H, W)

            with torch.no_grad():
                heatmaps = self.model(batch)  # (B, 1, H, W)

            results = []
            for i in range(heatmaps.shape[0]):
                hmap = heatmaps[i, 0].cpu().numpy()  # (H, W) float32 in [0, 1]
                results.append(self._peak_detect(hmap))
            return results
        except Exception:
            logger.warning("TrackNet detect_batch failed", exc_info=True)
            return [None] * len(diff_triplets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup(self, height: int = 360, width: int = 640) -> None:
        """Run one silent forward pass to initialise CUDA kernels."""
        if self.model is None:
            return
        dummy = torch.zeros(1, 9, height, width, device=self._device)
        with torch.no_grad():
            self.model(dummy)
        logger.info(
            "TrackNet warmup complete (device=%s, input=%dx%d)",
            self.device_str,
            width,
            height,
        )

    @staticmethod
    def _build_diff_triplet(frames: list) -> list:
        """No-op passthrough — frames are returned as-is.

        When a ``BackgroundSubtractor`` is used externally, pass the already-
        subtracted frames here.  This method exists as a hook for subclasses
        or pipelines that want inline subtraction.
        """
        return frames

    @staticmethod
    def _triplet_to_tensor(triplet: list) -> "torch.Tensor":
        """Convert a list of 3 BGR frames to a (9, H, W) float32 tensor.

        Channels are ordered: [B0, G0, R0, B1, G1, R1, B2, G2, R2].
        Values are normalised to [0, 1].
        """
        channels = []
        for frame in triplet:
            # frame shape: (H, W, 3) uint8 BGR
            arr = frame.astype(np.float32) / 255.0
            # Split into B, G, R planes and add as separate channels
            for c in range(3):
                channels.append(arr[:, :, c])

        stacked = np.stack(channels, axis=0)  # (9, H, W)
        return torch.from_numpy(stacked)

    @staticmethod
    def _peak_detect(
        heatmap: np.ndarray,
        threshold: float = _PEAK_THRESHOLD,
    ) -> Optional[dict]:
        """Locate the ball from a probability heatmap.

        Algorithm:
        1. Threshold at ``threshold`` to create a binary mask.
        2. Find connected components in the mask.
        3. Select the component whose maximum pixel value is highest
           (most confident region).
        4. Compute a weighted centroid within that component for
           sub-pixel accuracy.

        Args:
            heatmap:   (H, W) float32 array with values in [0, 1].
            threshold: Pixels below this value are ignored.

        Returns:
            ``{"x": float, "y": float, "confidence": float}`` or ``None``.
        """
        binary = (heatmap >= threshold).astype(np.uint8)

        if binary.max() == 0:
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            # Only background label (0) found
            return None

        # Find the component (excluding background label 0) with the highest
        # peak value in the original heatmap.
        best_label = -1
        best_peak = -1.0
        for lbl in range(1, num_labels):
            mask = labels == lbl
            peak = float(heatmap[mask].max())
            if peak > best_peak:
                best_peak = peak
                best_label = lbl

        if best_label == -1:
            return None

        # Weighted centroid within the best component
        component_mask = labels == best_label
        weights = heatmap * component_mask.astype(np.float32)
        total_weight = weights.sum()

        if total_weight < 1e-9:
            return None

        ys, xs = np.indices(heatmap.shape)
        cx = float((weights * xs).sum() / total_weight)
        cy = float((weights * ys).sum() / total_weight)

        return {"x": cx, "y": cy, "confidence": best_peak}
