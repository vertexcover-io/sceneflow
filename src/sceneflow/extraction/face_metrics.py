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
        eye_landmarks: Array of shape (6, 2) with (x, y) coordinates
                      Points indexed as: [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR value:
            - 0.25-0.32: Eyes fully open (BEST)
            - 0.18-0.25: Partially closed/squinting
            - 0.10-0.18: Eyes closed/blink
            - <0.08 or >0.35: Invalid
        Returns DEFAULT if calculation fails or value is invalid
    """
    if len(eye_landmarks) < 6:
        return EAR.DEFAULT

    # Using 6-point eye model from InsightFace 106-landmark
    # Vertical distances (A and B)
    vertical_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])  # p2 to p6
    vertical_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])  # p3 to p5
    # Horizontal distance (C)
    horizontal_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[3])  # p1 to p4

    if horizontal_dist == 0:
        return EAR.DEFAULT

    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    # Validate EAR is in reasonable range
    if not (EAR.MIN_VALID <= ear <= EAR.MAX_VALID):
        logger.debug(f"Invalid EAR value: {ear:.4f} (valid range: {EAR.MIN_VALID}-{EAR.MAX_VALID})")
        return EAR.DEFAULT

    return float(ear)


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth openness detection.

    MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)

    Args:
        mouth_landmarks: Array of shape (20, 2) with (x, y) coordinates

    Returns:
        MAR value:
            - 0.20-0.35: Mouth closed (BEST)
            - 0.35-0.55: Slightly open (talking/small motion)
            - 0.55-0.80: Moderately open (speaking/laugh)
            - 0.80-1.40+: Yawn/fully open (AVOID)
            - <0.15 or >1.5: Invalid
        Returns DEFAULT if calculation fails or value is invalid
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
