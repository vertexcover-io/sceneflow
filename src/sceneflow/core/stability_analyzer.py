"""Temporal stability analysis module.

This module provides the StabilityAnalyzer class which analyzes
temporal variance in motion and pose to boost scores of frames
in stable sequences.
"""

import logging
from typing import List

import numpy as np

from sceneflow.shared.constants import RANKING
from sceneflow.shared.models import FrameFeatures

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Analyzes temporal stability to boost scores of stable frames.

    Calculates variance in local time windows for motion and pose metrics.
    Frames in stable sequences (low variance) receive score boosts,
    as they're more suitable for clean cuts.

    Attributes:
        window_size: Size of sliding window for variance calculation (must be odd)

    Example:
        >>> analyzer = StabilityAnalyzer(window_size=5)
        >>> boosts = analyzer.calculate_stability_boosts(features)
        >>> # Stable frames get boost > 1.0 (up to 1.5x)
        >>> # Unstable frames get boost closer to 1.0
    """

    def __init__(self, window_size: int = RANKING.DEFAULT_STABILITY_WINDOW_SIZE):
        """
        Initialize stability analyzer with window size.

        Args:
            window_size: Size of sliding window for variance calculation.
                Must be odd number. Default: 5

        Raises:
            AssertionError: If window_size is not odd
        """
        assert window_size % 2 == 1, "Window size must be odd"
        self.window_size = window_size
        logger.debug("Initialized StabilityAnalyzer with window_size=%d", window_size)

    def calculate_stability_boosts(self, features: List[FrameFeatures]) -> List[float]:
        """
        Calculate stability boost multipliers for all frames.

        Analyzes variance in motion and pose within local time windows.
        Lower variance = more stable = higher boost.

        Args:
            features: List of frame features

        Returns:
            List of boost multipliers (1.0 to 1.5)
            - 1.5 = maximum boost (very stable sequence)
            - 1.0 = no boost (unstable sequence)
            - Boost = 1.0 + (0.5 × combined_stability)

        Note:
            Empty input returns empty list
        """
        if not features:
            logger.warning("No features provided to stability analyzer")
            return []

        half_window = self.window_size // 2
        motion_values = [f.motion_magnitude for f in features]
        pose_values = [f.pose_deviation for f in features]

        motion_variances = []
        pose_variances = []

        # Calculate variance in sliding windows
        for i in range(len(features)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(features), i + half_window + 1)

            motion_window = motion_values[start_idx:end_idx]
            pose_window = pose_values[start_idx:end_idx]

            motion_var = np.var(motion_window) if len(motion_window) > 1 else 0.0
            pose_var = np.var(pose_window) if len(pose_window) > 1 else 0.0

            motion_variances.append(motion_var)
            pose_variances.append(pose_var)

        motion_var_array = np.array(motion_variances)
        pose_var_array = np.array(pose_variances)

        # Normalize variances (low variance = high stability = high score)
        motion_normalized = self._normalize_variance(motion_var_array)
        pose_normalized = self._normalize_variance(pose_var_array)

        # Combine and calculate boosts
        boosts = []
        for motion_norm, pose_norm in zip(motion_normalized, pose_normalized):
            combined_stability = (motion_norm + pose_norm) / 2.0
            # Boost range: 1.0 to 1.5
            boost = 1.0 + (RANKING.MAX_STABILITY_BOOST_MULTIPLIER * combined_stability)
            boosts.append(boost)

        # Log statistics
        logger.debug(
            "Stability boosts: min=%.3f, max=%.3f, mean=%.3f",
            min(boosts),
            max(boosts),
            np.mean(boosts)
        )

        return boosts

    def _normalize_variance(self, variances: np.ndarray) -> np.ndarray:
        """
        Normalize variance array to [0, 1] range.

        Uses 95th percentile as maximum to reduce outlier impact.
        Inverts the scale so low variance = high score.

        Args:
            variances: Array of variance values

        Returns:
            Normalized array where:
            - 0.0 = high variance (unstable)
            - 1.0 = low variance (stable)
        """
        if len(variances) == 0:
            return variances

        max_var = np.percentile(variances, RANKING.VARIANCE_PERCENTILE_THRESHOLD)

        if max_var < 1e-9:
            # All variances are near zero (very stable)
            return np.ones_like(variances)

        # Normalize and clip to [0, 1]
        normalized = np.clip(variances / max_var, 0.0, 1.0)

        # Invert: low variance = high score
        normalized = 1.0 - normalized

        return normalized
