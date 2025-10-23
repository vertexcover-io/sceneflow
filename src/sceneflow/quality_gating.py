"""Quality gating for filtering low-quality frames."""

import numpy as np
from typing import List
from .models import FrameFeatures


class QualityGate:
    """Applies quality-based penalties to frame scores."""

    def __init__(self, percentile_threshold: float = 75.0):
        """
        Args:
            percentile_threshold: Percentile threshold for quality gating (0-100)
        """
        self.percentile_threshold = percentile_threshold

    def calculate_penalties(self, features: List[FrameFeatures]) -> List[float]:
        """
        Calculate quality penalties based on visual sharpness.
        Frames below the percentile threshold get penalized.

        Args:
            features: List of frame features

        Returns:
            List of penalty multipliers (0.0 to 1.0)
        """
        if not features:
            return []

        sharpness_values = [f.sharpness for f in features]
        threshold = np.percentile(sharpness_values, self.percentile_threshold)

        penalties = []
        for sharpness in sharpness_values:
            if sharpness >= threshold:
                penalty = 1.0  # No penalty
            else:
                # Linear penalty based on how far below threshold
                penalty = max(0.5, sharpness / threshold)  # At least 50% penalty
            penalties.append(penalty)

        return penalties
