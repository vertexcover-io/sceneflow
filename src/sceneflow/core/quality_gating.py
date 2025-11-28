"""Quality gating module for frame filtering.

This module provides the QualityGate class which filters out frames
below a quality threshold based on visual sharpness.
"""

import logging
from typing import List

import numpy as np

from sceneflow.shared.constants import RANKING
from sceneflow.shared.models import FrameFeatures

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Filters frames based on visual quality (sharpness).

    Applies quality penalties to frames below a percentile threshold,
    preventing selection of objectively poor-quality frames even if
    they score well on other metrics.

    Attributes:
        percentile_threshold: Percentile threshold for quality filtering (0-100)

    Example:
        >>> gate = QualityGate(percentile_threshold=75.0)
        >>> penalties = gate.calculate_penalties(features)
        >>> # Frames above 75th percentile get penalty=1.0 (no penalty)
        >>> # Frames below get penalty=0.5-1.0 (linear scaling)
    """

    def __init__(self, percentile_threshold: float = RANKING.DEFAULT_QUALITY_GATE_PERCENTILE):
        """
        Initialize quality gate with percentile threshold.

        Args:
            percentile_threshold: Percentile threshold (0-100).
                Frames below this percentile receive quality penalties.
                Default: 75.0 (top 25% of frames by sharpness)
        """
        self.percentile_threshold = percentile_threshold
        logger.debug("Initialized QualityGate with %.1f%% threshold", percentile_threshold)

    def calculate_penalties(self, features: List[FrameFeatures]) -> List[float]:
        """
        Calculate quality penalties based on visual sharpness.

        Frames below the percentile threshold get penalized based on
        how far they are below the threshold. This prevents selection
        of objectively blurry or low-quality frames.

        Args:
            features: List of frame features

        Returns:
            List of penalty multipliers (0.5 to 1.0)
            - 1.0 = no penalty (sharpness >= threshold)
            - 0.5-1.0 = linear penalty (sharpness < threshold)
            - Minimum penalty is 50% (0.5)

        Note:
            Empty input returns empty list
        """
        if not features:
            logger.warning("No features provided to quality gate")
            return []

        sharpness_values = [f.sharpness for f in features]
        threshold = np.percentile(sharpness_values, self.percentile_threshold)

        logger.debug(
            "Quality gate threshold (%.1f%%): %.2f",
            self.percentile_threshold,
            threshold
        )

        penalties = []
        penalized_count = 0

        for sharpness in sharpness_values:
            if sharpness >= threshold:
                penalty = 1.0  # No penalty
            else:
                # Linear penalty based on how far below threshold
                # At least 50% penalty (MIN_QUALITY_PENALTY)
                penalty = max(RANKING.MIN_QUALITY_PENALTY, sharpness / threshold)
                penalized_count += 1

            penalties.append(penalty)

        if penalized_count > 0:
            logger.debug(
                "Applied quality penalties to %d/%d frames (%.1f%%)",
                penalized_count,
                len(features),
                100.0 * penalized_count / len(features)
            )

        return penalties
