"""Frame-level visual metric calculations.

This module provides functions for calculating frame-level visual metrics
including optical flow motion and image sharpness.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from sceneflow.shared.constants import MOTION, SHARPNESS

logger = logging.getLogger(__name__)


class MotionTracker:
    """Tracks optical flow motion between consecutive frames."""

    def __init__(self):
        """Initialize motion tracker."""
        self.prev_frame_gray: Optional[np.ndarray] = None

    def calculate_motion_magnitude(self, frame: np.ndarray) -> float:
        """
        Calculate optical flow magnitude between current and previous frame.

        Uses Farneback dense optical flow for motion estimation.

        Args:
            frame: Current frame (BGR format)

        Returns:
            Median motion magnitude in pixels. Lower is better for cut points.
            - < 0.5: Very stable (BEST)
            - 0.5-2.0: Some movement
            - > 2.0: High motion (AVOID)

        Note:
            First call returns 0.0 as there's no previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            gray,
            None,
            pyr_scale=MOTION.PYR_SCALE,
            levels=MOTION.LEVELS,
            winsize=MOTION.WINSIZE,
            iterations=MOTION.ITERATIONS,
            poly_n=MOTION.POLY_N,
            poly_sigma=MOTION.POLY_SIGMA,
            flags=0
        )

        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Use median instead of mean to reduce outlier impact
        avg_magnitude = float(np.median(magnitude))

        self.prev_frame_gray = gray

        logger.debug("Motion magnitude: %.2f pixels", avg_magnitude)
        return avg_magnitude

    def reset(self) -> None:
        """
        Reset internal state.

        Call this when starting to process a new video or time range
        to ensure optical flow starts fresh.
        """
        self.prev_frame_gray = None
        logger.debug("Motion tracker state reset")


def calculate_visual_sharpness(frame: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.

    Higher values indicate sharper, clearer images.

    Args:
        frame: Input frame (BGR format)

    Returns:
        Sharpness score. Higher is better for cut points.
        - < 50: Blurry (AVOID)
        - 50-200: Acceptable
        - > 200: Sharp (BEST)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate variance as sharpness metric
    variance = laplacian.var()

    logger.debug("Visual sharpness: %.2f", variance)
    return float(variance)
