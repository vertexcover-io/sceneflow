"""Main feature extraction module for video frame analysis.

Extracts for podcast/talking head videos:
- Eye openness (EAR) - detect blinks
- Mouth openness (MAR) - detect mid-sentence
- Sharpness - detect motion blur
- Face center - for consistency check

Example:
    >>> from sceneflow.extraction.extractor import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> metrics = extractor.extract_face_metrics(frame)
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

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
from sceneflow.extraction.face_metrics import calculate_ear, calculate_mar

logger = logging.getLogger(__name__)


@dataclass
class FaceMetrics:
    """Metrics extracted from a face."""
    ear: float           # Eye aspect ratio
    mar: float           # Mouth aspect ratio
    sharpness: float     # Face region sharpness
    center: Tuple[float, float]  # Face center position
    detected: bool       # Whether face was found


# Landmark indices for 106-point model
LEFT_EYE_INDICES = list(range(INSIGHTFACE.LEFT_EYE_START, INSIGHTFACE.LEFT_EYE_END))
RIGHT_EYE_INDICES = list(range(INSIGHTFACE.RIGHT_EYE_START, INSIGHTFACE.RIGHT_EYE_END))
MOUTH_OUTER_INDICES = list(range(INSIGHTFACE.MOUTH_OUTER_START, INSIGHTFACE.MOUTH_OUTER_END))


class FeatureExtractor:
    """
    Extracts facial features for podcast/talking head cut point detection.
    
    Extracts:
    - Eye Aspect Ratio (EAR) - detect blinks
    - Mouth Aspect Ratio (MAR) - detect talking
    - Sharpness - detect motion blur
    - Face center - for consistency check
    """

    def __init__(self, min_face_confidence: float = INSIGHTFACE.MIN_FACE_CONFIDENCE):
        """
        Initialize InsightFace with 106-landmark detection.

        Args:
            min_face_confidence: Minimum confidence threshold for face detection

        Raises:
            InsightFaceError: If 106-landmark model fails to load
        """
        self.min_face_confidence = min_face_confidence

        logger.info("Initializing InsightFace with 106-landmark detection...")

        try:
            self.app = FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(INSIGHTFACE.DEFAULT_DET_SIZE.width, INSIGHTFACE.DEFAULT_DET_SIZE.height))
            logger.info("Successfully loaded InsightFace 106-landmark model")
        except Exception as e:
            raise InsightFaceError(
                f"Failed to load 106-landmark model (required for EAR/MAR): {e}"
            ) from e

    def extract_face_metrics(self, frame: np.ndarray) -> FaceMetrics:
        """
        Extract all face metrics from the primary face in frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            FaceMetrics with EAR, MAR, sharpness, center, and detection status
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)

        # Filter by confidence
        if faces:
            faces = [
                f for f in faces
                if hasattr(f, 'det_score') and f.det_score >= self.min_face_confidence
            ]

        if not faces:
            logger.debug("No faces detected in frame")
            return FaceMetrics(
                ear=EAR.DEFAULT,
                mar=MAR.DEFAULT,
                sharpness=0.0,
                center=(0.0, 0.0),
                detected=False
            )

        # Use the largest face (primary subject in talking head videos)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Extract metrics
        ear = self._extract_ear(face)
        mar = self._extract_mar(face)
        sharpness = self._calculate_sharpness(frame, face.bbox)
        center = self._get_face_center(face.bbox)

        logger.debug("Extracted: EAR=%.3f, MAR=%.3f, sharpness=%.1f", ear, mar, sharpness)
        return FaceMetrics(
            ear=ear,
            mar=mar,
            sharpness=sharpness,
            center=center,
            detected=True
        )

    def _calculate_sharpness(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """
        Calculate sharpness of the face region using Laplacian variance.
        Higher value = sharper image, lower = blurry/motion blur.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add some padding
        h, w = frame.shape[:2]
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return 0.0
        
        # Convert to grayscale and compute Laplacian variance
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)

    def _get_face_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Get the center point of the face bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _extract_ear(self, face) -> float:
        """
        Extract Eye Aspect Ratio from face landmarks with intelligent handling.

        Calculates EAR for each eye separately. If one eye has an abnormal value
        (outside valid range), uses only the correct eye's value instead of averaging.
        This handles cases like partial occlusion, winking, or landmark detection errors.

        Returns:
            Average EAR if both eyes valid, single eye EAR if one invalid, DEFAULT if both invalid
        """
        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            return EAR.DEFAULT

        landmarks = face.landmark_2d_106.astype(int)

        if len(landmarks) <= max(LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
            return EAR.DEFAULT

        left_eye = landmarks[LEFT_EYE_INDICES]
        right_eye = landmarks[RIGHT_EYE_INDICES]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Intelligent handling: use only valid values
        left_valid = left_ear != EAR.DEFAULT
        right_valid = right_ear != EAR.DEFAULT

        if left_valid and right_valid:
            # Both eyes valid - use average
            return (left_ear + right_ear) / 2.0
        elif left_valid:
            # Only left eye valid - use it
            logger.debug(f"Using only left eye EAR: {left_ear:.4f} (right eye invalid)")
            return left_ear
        elif right_valid:
            # Only right eye valid - use it
            logger.debug(f"Using only right eye EAR: {right_ear:.4f} (left eye invalid)")
            return right_ear
        else:
            # Both invalid - return default
            logger.debug("Both eyes have invalid EAR values")
            return EAR.DEFAULT

    def _extract_mar(self, face) -> float:
        """Extract Mouth Aspect Ratio from face landmarks."""
        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            return MAR.DEFAULT

        landmarks = face.landmark_2d_106.astype(int)

        if len(landmarks) <= max(MOUTH_OUTER_INDICES):
            return MAR.DEFAULT

        mouth = landmarks[MOUTH_OUTER_INDICES]
        return calculate_mar(mouth)
