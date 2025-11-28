"""Feature extraction package for video frame analysis.

This package provides modules for extracting facial and visual features from video frames.
"""

from sceneflow.extraction.extractor import (
    FeatureExtractor,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_OUTER_INDICES,
)

__all__ = [
    'FeatureExtractor',
    'LEFT_EYE_INDICES',
    'RIGHT_EYE_INDICES',
    'MOUTH_OUTER_INDICES',
]
