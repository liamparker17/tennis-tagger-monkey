"""
Frame stability detection for intelligent court re-detection.

Detects scene changes to avoid wasteful re-detection of static elements.
"""

import cv2
import numpy as np
import logging
from typing import Optional


logger = logging.getLogger('Stability')


class SceneStabilityChecker:
    """
    Check if camera/scene is stable using frame similarity.
    """

    def __init__(self, threshold: float = 0.85, histogram_bins: int = 32):
        """
        Initialise stability checker.

        Args:
            threshold: Similarity threshold (0-1, higher = more similar required)
            histogram_bins: Number of bins for histogram comparison
        """
        self.threshold = threshold
        self.histogram_bins = histogram_bins
        self.last_histogram = None
        self.last_frame_hash = None

    def is_stable(self, frame: np.ndarray) -> bool:
        """
        Check if frame is similar to previous frame (scene is stable).

        Args:
            frame: Current frame (BGR)

        Returns:
            True if scene is stable, False if scene changed
        """
        # Compute histogram for current frame
        current_hist = self._compute_histogram(frame)

        if self.last_histogram is None:
            # First frame - assume unstable to trigger detection
            self.last_histogram = current_hist
            return False

        # Compare histograms using correlation
        similarity = cv2.compareHist(
            self.last_histogram,
            current_hist,
            cv2.HISTCMP_CORREL
        )

        # Update reference
        self.last_histogram = current_hist

        is_stable = similarity >= self.threshold

        if not is_stable:
            logger.debug(f"Scene change detected (similarity: {similarity:.3f})")

        return is_stable

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute colour histogram for a frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Normalised histogram
        """
        # Convert to HSV for better colour representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute 3D histogram (H, S, V)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.histogram_bins, self.histogram_bins, self.histogram_bins],
            [0, 180, 0, 256, 0, 256]
        )

        # Normalise
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        return hist

    def reset(self):
        """Reset the stability checker (for new video or scene)."""
        self.last_histogram = None
        self.last_frame_hash = None
        logger.debug("Stability checker reset")
