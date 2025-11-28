"""Main feature extraction module for video frame analysis.

This module provides the FeatureExtractor class which handles extraction
of facial features (eye openness, mouth activity, pose) and visual features
(motion, sharpness) from video frames using InsightFace and OpenCV.

The extractor supports both single-face and multi-face scenarios with
center-weighted aggregation for multi-face frames.

Example:
    >>> from sceneflow.extraction.extractor import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> num_faces, face_list, metrics = extractor.extract_all_faces(frame)
    >>> print(f"Detected {num_faces} faces")

Classes:
    FeatureExtractor: Main feature extraction class

Constants:
    LEFT_EYE_INDICES: Landmark indices for left eye (7 points)
    RIGHT_EYE_INDICES: Landmark indices for right eye (7 points)
    MOUTH_OUTER_INDICES: Landmark indices for outer mouth (20 points)
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    raise ImportError(
        "insightface not installed. Install it using: uv add insightface onnxruntime"
    ) from e

from sceneflow.shared.constants import INSIGHTFACE, EAR, MAR
from sceneflow.shared.exceptions import InsightFaceError
from sceneflow.shared.models import AggregatedFaceMetrics
from sceneflow.extraction.face_metrics import (
    calculate_ear,
    calculate_mar,
    calculate_center_distance,
    calculate_center_weight,
    aggregate_multi_face_metrics,
)
from sceneflow.extraction.frame_metrics import (
    MotionTracker,
    calculate_visual_sharpness,
)

logger = logging.getLogger(__name__)


# Landmark indices for 106-point model
LEFT_EYE_INDICES = list(range(
    INSIGHTFACE.LEFT_EYE_START,
    INSIGHTFACE.LEFT_EYE_END
))
RIGHT_EYE_INDICES = list(range(
    INSIGHTFACE.RIGHT_EYE_START,
    INSIGHTFACE.RIGHT_EYE_END
))
MOUTH_OUTER_INDICES = list(range(
    INSIGHTFACE.MOUTH_OUTER_START,
    INSIGHTFACE.MOUTH_OUTER_END
))


class FeatureExtractor:
    """
    Extracts facial and visual features for determining optimal video cut points.

    Uses InsightFace with 106-landmark detection for detailed facial analysis.

    Key principles:
    - Normal eye openness = better (not blinking, not wide open)
    - Lower motion = better (stable frame)
    - Lower expression activity = better (neutral face)
    - Stable pose = better (facing camera directly)
    - Higher sharpness = better (clear image)

    Multi-face support:
    - Detects all faces in frame
    - Uses center-weighted averaging for final metrics
    - Closer faces to center have higher influence

    Attributes:
        has_106_landmarks: Whether 106-landmark model is loaded
        app: InsightFace FaceAnalysis instance
        motion_tracker: MotionTracker for optical flow
        center_weighting_strength: Controls center distance weighting
        min_face_confidence: Minimum confidence for face detection
    """

    def __init__(
        self,
        center_weighting_strength: float = INSIGHTFACE.DEFAULT_CENTER_WEIGHTING_STRENGTH,
        min_face_confidence: float = INSIGHTFACE.MIN_FACE_CONFIDENCE
    ):
        """
        Initialize InsightFace with 106-landmark detection.

        Args:
            center_weighting_strength: Controls how strongly center distance
                affects weighting (higher = stronger center bias)
            min_face_confidence: Minimum confidence threshold for face detection (0-1)

        Raises:
            InsightFaceError: If InsightFace fails to initialize
        """
        self.center_weighting_strength = center_weighting_strength
        self.min_face_confidence = min_face_confidence

        logger.info("Initializing InsightFace with 106-landmark detection...")

        try:
            # Try to load with 106-landmark model
            self.app = FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=INSIGHTFACE.DEFAULT_DET_SIZE)
            self.has_106_landmarks = True
            logger.info("Successfully loaded InsightFace 106-landmark model")
        except Exception as e:
            logger.warning(
                "Could not load 106-landmark model (%s), "
                "falling back to 5-point detection",
                e
            )
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=-1, det_size=INSIGHTFACE.DEFAULT_DET_SIZE)
                self.has_106_landmarks = False
                logger.info("Loaded InsightFace with 5-point landmarks")
            except Exception as fallback_error:
                raise InsightFaceError(
                    f"Failed to initialize InsightFace: {fallback_error}"
                ) from fallback_error

        # Motion tracking
        self.motion_tracker = MotionTracker()

    def extract_all_faces(
        self,
        frame: np.ndarray
    ) -> Tuple[int, List, AggregatedFaceMetrics]:
        """
        Extract features from all detected faces in the frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Tuple containing:
            - num_faces: Number of faces detected
            - face_features_list: List of FaceFeatures objects (from models.py)
            - aggregated_metrics: AggregatedFaceMetrics dataclass with metrics
              aggregated across all faces

        Note:
            If no faces detected, returns (0, [], default_penalty_values)
        """
        from sceneflow.shared.models import FaceFeatures

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)
        frame_shape = frame.shape[:2]  # (height, width)

        # Filter faces by confidence
        if faces:
            faces_before = len(faces)
            faces = [
                f for f in faces
                if hasattr(f, 'det_score') and f.det_score >= self.min_face_confidence
            ]
            if len(faces) < faces_before:
                logger.debug(
                    "Filtered %d faces below confidence threshold %.2f",
                    faces_before - len(faces),
                    self.min_face_confidence
                )

        if not faces:
            logger.debug("No faces detected in frame")
            return 0, [], AggregatedFaceMetrics(
                eye_openness=EAR.DEFAULT,
                expression_activity=MAR.DEFAULT,
                pose_deviation=0.5
            )

        face_features_list: List[FaceFeatures] = []
        face_metrics: List[Tuple[float, float, float]] = []
        weights: List[float] = []

        for face_idx, face in enumerate(faces):
            # Calculate center distance and weight
            bbox = face.bbox
            center_distance = calculate_center_distance(bbox, frame_shape)
            center_weight = calculate_center_weight(
                center_distance,
                self.center_weighting_strength
            )

            # Extract per-face metrics
            eye_openness = self._extract_single_face_eye_openness(face)
            expression_activity = self._extract_single_face_expression(face)
            pose_deviation = self._extract_single_face_pose(face, bbox, frame_shape)

            # Store per-face data
            face_features_list.append(FaceFeatures(
                face_index=face_idx,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                center_distance=center_distance,
                center_weight=center_weight,
                eye_openness=eye_openness,
                expression_activity=expression_activity,
                pose_deviation=pose_deviation
            ))

            face_metrics.append((eye_openness, expression_activity, pose_deviation))
            weights.append(center_weight)

        # Aggregate metrics using center-weighted averaging
        aggregated = aggregate_multi_face_metrics(face_metrics, weights)

        logger.debug(
            "Extracted features from %d faces (aggregated: EAR=%.3f, MAR=%.3f, pose=%.3f)",
            len(faces), aggregated.eye_openness, aggregated.expression_activity, aggregated.pose_deviation
        )

        return len(faces), face_features_list, aggregated

    def _extract_single_face_eye_openness(self, face) -> float:
        """
        Extract eye openness for a single face object.

        Args:
            face: InsightFace face detection result

        Returns:
            Average EAR across both eyes, or default value if unavailable
        """
        # Check if 106 landmarks are available
        if (self.has_106_landmarks and
            hasattr(face, 'landmark_2d_106') and
            face.landmark_2d_106 is not None):

            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
                left_eye_points = landmarks[LEFT_EYE_INDICES]
                right_eye_points = landmarks[RIGHT_EYE_INDICES]

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)

                avg_ear = (left_ear + right_ear) / 2.0
                return avg_ear

        logger.debug("106 landmarks not available for eye openness")
        return EAR.DEFAULT

    def _extract_single_face_expression(self, face) -> float:
        """
        Extract expression activity for a single face object.

        Args:
            face: InsightFace face detection result

        Returns:
            MAR value representing expression activity
        """
        # Check if 106 landmarks are available
        if (self.has_106_landmarks and
            hasattr(face, 'landmark_2d_106') and
            face.landmark_2d_106 is not None):

            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(MOUTH_OUTER_INDICES):
                mouth_points = landmarks[MOUTH_OUTER_INDICES]
                mar = calculate_mar(mouth_points)
                return mar

        logger.debug("106 landmarks not available for expression activity")
        return MAR.DEFAULT

    def _extract_single_face_pose(
        self,
        face,
        bbox: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Extract pose deviation for a single face object.

        Args:
            face: InsightFace face detection result
            bbox: Face bounding box
            frame_shape: Frame dimensions (height, width)

        Returns:
            Pose deviation from center (0-1, lower is better)
        """
        frame_h, frame_w = frame_shape
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2

        # Normalized deviation from center
        center_x_dev = abs(face_center_x - frame_w/2) / (frame_w/2)
        center_y_dev = abs(face_center_y - frame_h/2) / (frame_h/2)

        deviation = (center_x_dev + center_y_dev) / 2.0
        return float(deviation)

    def extract_motion_magnitude(self, frame: np.ndarray) -> float:
        """
        Calculate optical flow magnitude between current and previous frame.

        Args:
            frame: Current frame (BGR format)

        Returns:
            Median motion magnitude in pixels
        """
        return self.motion_tracker.calculate_motion_magnitude(frame)

    def extract_visual_sharpness(self, frame: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Sharpness score
        """
        return calculate_visual_sharpness(frame)

    def reset(self) -> None:
        """
        Reset internal state (optical flow tracking).

        Call this when starting to process a new video or time range
        to ensure optical flow starts fresh.
        """
        self.motion_tracker.reset()
        logger.debug("Feature extractor state reset")
