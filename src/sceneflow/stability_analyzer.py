"""Stability analysis for boosting temporally stable frames."""

import numpy as np
from typing import List
from .models import FrameFeatures


class StabilityAnalyzer:
    """Analyzes temporal stability and boosts stable frames."""

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Size of window for stability analysis (must be odd)
        """
        assert window_size % 2 == 1, "Window size must be odd"
        self.window_size = window_size

    def calculate_stability_boosts(self, features: List[FrameFeatures]) -> List[float]:
        """
        Calculate stability boosts based on local motion and pose consistency.
        Frames with low local variance get boosted.

        Args:
            features: List of frame features

        Returns:
            List of boost multipliers (1.0 to ~1.2)
        """
        if not features:
            return []

        half_window = self.window_size // 2
        motion_values = [f.motion_magnitude for f in features]
        pose_values = [f.pose_deviation for f in features]

        boosts = []
        for i in range(len(features)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(features), i + half_window + 1)

            # Calculate variance in local window
            motion_window = motion_values[start_idx:end_idx]
            pose_window = pose_values[start_idx:end_idx]

            motion_variance = np.var(motion_window) if len(motion_window) > 1 else 0.0
            pose_variance = np.var(pose_window) if len(pose_window) > 1 else 0.0

            # Combined stability score (lower variance = more stable)
            # Normalize and invert so stable frames get higher boost
            stability_score = 1.0 / (1.0 + motion_variance + pose_variance)

            # Map to boost range [1.0, 1.2]
            boost = 1.0 + (0.2 * stability_score)
            boosts.append(boost)

        return boosts
