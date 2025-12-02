"""Face-specific metric calculations.

Provides EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) calculations
for detecting blinks and talking in podcast/talking head videos.
"""

import logging
import numpy as np

from sceneflow.shared.constants import EAR, MAR

logger = logging.getLogger(__name__)


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(point1 - point2))


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: Array of shape (7, 2) with (x, y) coordinates

    Returns:
        EAR value (0.25-0.35 normal, <0.2 = blink)
    """
    if len(eye_landmarks) < 7:
        return EAR.DEFAULT

    horizontal_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[4])
    vertical_dist1 = euclidean_distance(eye_landmarks[5], eye_landmarks[1])
    vertical_dist2 = euclidean_distance(eye_landmarks[3], eye_landmarks[2])

    if horizontal_dist == 0:
        return EAR.DEFAULT

    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    if not (EAR.MIN_VALID <= ear <= EAR.MAX_VALID):
        return EAR.DEFAULT

    return float(ear)


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth openness detection.

    MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)

    Args:
        mouth_landmarks: Array of shape (20, 2) with (x, y) coordinates

    Returns:
        MAR value (<0.15 closed, >0.3 = talking/open)
    """
    if len(mouth_landmarks) < 12:
        return MAR.DEFAULT

    vertical_dist1 = euclidean_distance(mouth_landmarks[2], mouth_landmarks[10])
    vertical_dist2 = euclidean_distance(mouth_landmarks[4], mouth_landmarks[8])
    horizontal_dist = euclidean_distance(mouth_landmarks[0], mouth_landmarks[6])

    if horizontal_dist == 0:
        return MAR.DEFAULT

    mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    if not (MAR.MIN_VALID <= mar <= MAR.MAX_VALID):
        return MAR.DEFAULT

    return float(mar)
