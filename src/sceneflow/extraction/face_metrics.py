"""Face-specific metric calculations.

This module provides functions for calculating facial metrics used in
cut point detection, including Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and center-weighted aggregation for multi-face scenarios.
"""

import logging
from typing import List, Tuple

import numpy as np

from sceneflow.shared.constants import INSIGHTFACE, EAR, MAR
from sceneflow.shared.models import AggregatedFaceMetrics

logger = logging.getLogger(__name__)


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Euclidean distance between points
    """
    return float(np.linalg.norm(point1 - point2))


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.

    The EAR is computed using the formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    For InsightFace 106-landmark 7-point eye model:
    - Index 0: Leftmost corner
    - Index 4: Rightmost corner
    - Index 5: Top center
    - Index 1: Bottom left
    - Index 3: Center point
    - Index 2: Bottom right

    Args:
        eye_landmarks: Array of shape (7, 2) with (x, y) coordinates

    Returns:
        EAR value in range [0.1, 0.6], or default value if calculation fails.
        Normal range: 0.25-0.35

    Note:
        Returns default value (0.3) if:
        - Insufficient landmarks
        - Calculation produces invalid result

    Reference:
        Soukupová & Čech (2016). "Real-Time Eye Blink Detection using
        Facial Landmarks." 21st Computer Vision Winter Workshop.
    """
    if len(eye_landmarks) < 7:
        logger.debug(
            "Insufficient eye landmarks: expected 7, got %d. "
            "Using default EAR value.",
            len(eye_landmarks)
        )
        return EAR.DEFAULT

    # Horizontal distance (corner to corner)
    horizontal_dist = euclidean_distance(
        eye_landmarks[0],
        eye_landmarks[4]
    )

    # Vertical distances (top-bottom pairs)
    vertical_dist1 = euclidean_distance(
        eye_landmarks[5],
        eye_landmarks[1]
    )
    vertical_dist2 = euclidean_distance(
        eye_landmarks[3],
        eye_landmarks[2]
    )

    if horizontal_dist == 0:
        logger.debug("Zero horizontal distance in EAR calculation")
        return EAR.DEFAULT

    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    # Validate result
    if not (EAR.MIN_VALID <= ear <= EAR.MAX_VALID):
        logger.debug(
            "Calculated EAR %.3f outside valid range [%.2f, %.2f]. "
            "Using default value.",
            ear, EAR.MIN_VALID, EAR.MAX_VALID
        )
        return EAR.DEFAULT

    return float(ear)


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth openness detection.

    The MAR is computed using the formula:
        MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)

    For InsightFace 106-landmark 20-point mouth model, uses specific
    points for vertical (top-bottom) and horizontal (left-right) distances.

    Args:
        mouth_landmarks: Array of shape (20, 2) with (x, y) coordinates

    Returns:
        MAR value representing mouth openness. Lower is better for cut points.
        - < 0.15: Closed mouth (BEST)
        - 0.15-0.30: Slightly open
        - > 0.30: Open mouth (talking/yawning - AVOID)

    Note:
        Returns default value (0.2) if:
        - Insufficient landmarks
        - Calculation produces invalid result
    """
    if len(mouth_landmarks) < 12:
        logger.debug(
            "Insufficient mouth landmarks: expected 12+, got %d. "
            "Using default MAR value.",
            len(mouth_landmarks)
        )
        return MAR.DEFAULT

    # Vertical mouth distances
    vertical_dist1 = euclidean_distance(
        mouth_landmarks[2],
        mouth_landmarks[10]
    )
    vertical_dist2 = euclidean_distance(
        mouth_landmarks[4],
        mouth_landmarks[8]
    )

    # Horizontal mouth distance
    horizontal_dist = euclidean_distance(
        mouth_landmarks[0],
        mouth_landmarks[6]
    )

    if horizontal_dist == 0:
        logger.debug("Zero horizontal distance in MAR calculation")
        return MAR.DEFAULT

    mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    # Validate result
    if not (MAR.MIN_VALID <= mar <= MAR.MAX_VALID):
        logger.debug(
            "Calculated MAR %.3f outside valid range [%.2f, %.2f]. "
            "Using default value.",
            mar, MAR.MIN_VALID, MAR.MAX_VALID
        )
        return MAR.DEFAULT

    return float(mar)


def calculate_center_distance(
    bbox: np.ndarray,
    frame_shape: Tuple[int, int]
) -> float:
    """
    Calculate normalized distance from frame center to face center.

    Args:
        bbox: Face bounding box [x1, y1, x2, y2]
        frame_shape: Frame dimensions (height, width)

    Returns:
        Normalized distance from center (0 = at center, 1 = at edge)
    """
    frame_h, frame_w = frame_shape
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    # Calculate face center
    face_center_x = (bbox[0] + bbox[2]) / 2
    face_center_y = (bbox[1] + bbox[3]) / 2

    # Calculate Euclidean distance from frame center
    dx = (face_center_x - frame_center_x) / frame_w
    dy = (face_center_y - frame_center_y) / frame_h
    distance = np.sqrt(dx**2 + dy**2)

    # Normalize to [0, 1] - max possible distance is ~0.707 (corner to center)
    normalized_distance = min(
        distance / INSIGHTFACE.MAX_DISTANCE_NORMALIZED,
        1.0
    )

    return float(normalized_distance)


def calculate_center_weight(
    center_distance: float,
    center_weighting_strength: float
) -> float:
    """
    Calculate weight for center-weighted averaging.

    Args:
        center_distance: Normalized distance from center (0-1)
        center_weighting_strength: Controls weighting strength

    Returns:
        Weight value (higher for faces closer to center)
    """
    # Apply center weighting: weight = 1 / (1 + strength * distance)
    weight = 1.0 / (1.0 + center_weighting_strength * center_distance)
    return float(weight)


def aggregate_multi_face_metrics(
    face_features_list: List[Tuple[float, float, float]],
    weights: List[float]
) -> AggregatedFaceMetrics:
    """
    Aggregate metrics from multiple faces using weighted averaging.

    Args:
        face_features_list: List of tuples
            (eye_openness, expression_activity, pose_deviation)
        weights: List of weights corresponding to each face

    Returns:
        AggregatedFaceMetrics dataclass with aggregated metrics
    """
    if not face_features_list or not weights:
        logger.debug("No faces to aggregate, returning default penalty values")
        return AggregatedFaceMetrics(
            eye_openness=EAR.DEFAULT,
            expression_activity=MAR.DEFAULT,
            pose_deviation=0.5
        )

    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    if total_weight == 0:
        logger.warning("Total weight is zero, using equal weights")
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]

    # Weighted average for each metric
    eye_openness = sum(
        f[0] * w for f, w in zip(face_features_list, normalized_weights)
    )
    expression = sum(
        f[1] * w for f, w in zip(face_features_list, normalized_weights)
    )
    pose = sum(
        f[2] * w for f, w in zip(face_features_list, normalized_weights)
    )

    return AggregatedFaceMetrics(
        eye_openness=float(eye_openness),
        expression_activity=float(expression),
        pose_deviation=float(pose)
    )
