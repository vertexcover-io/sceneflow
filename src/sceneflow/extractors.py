import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from typing import Optional, Tuple, List

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "insightface not installed. Install it using: uv add insightface onnxruntime"
    )


# Landmark indices for 106-point model (from test_insightface.py)
LEFT_EYE_INDICES = list(range(35, 42))      # 7 points for left eye
RIGHT_EYE_INDICES = list(range(42, 49))     # 7 points for right eye
MOUTH_OUTER_INDICES = list(range(52, 72))   # 20 points for outer mouth


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
    """

    def __init__(self, center_weighting_strength: float = 1.0, min_face_confidence: float = 0.5):
        """
        Initialize InsightFace with 106-landmark detection.

        Args:
            center_weighting_strength: Controls how strongly center distance affects weighting
            min_face_confidence: Minimum confidence threshold for face detection
        """
        self.center_weighting_strength = center_weighting_strength
        self.min_face_confidence = min_face_confidence
        print("Initializing InsightFace with 106-landmark detection...")

        try:
            # Try to load with 106-landmark model
            self.app = FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.has_106_landmarks = True
            print("Successfully loaded 106-landmark model")
        except Exception as e:
            print(f"Warning: Could not load 106-landmark model: {e}")
            print("Falling back to standard 5-point detection...")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.has_106_landmarks = False

        # Optical flow tracking
        self.prev_frame_gray = None

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)

    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        For InsightFace 106-landmark 7-point eye model:
        - Index 0: Leftmost corner
        - Index 4: Rightmost corner
        - Index 5: Top center
        - Index 1: Bottom left
        - Index 3: Center point
        - Index 2: Bottom right

        Lower EAR indicates closed/closing eyes (blinking).
        Typical threshold: EAR < 0.25 indicates a blink.
        Normal range: 0.25-0.35
        """
        if len(eye_landmarks) < 7:
            return 0.3  # Default neutral value

        # Horizontal distance (corner to corner)
        C = self._euclidean_distance(eye_landmarks[0], eye_landmarks[4])

        # Vertical distances (top-bottom pairs)
        A = self._euclidean_distance(eye_landmarks[5], eye_landmarks[1])  # Top to bottom-left
        B = self._euclidean_distance(eye_landmarks[3], eye_landmarks[2])  # Center to bottom-right

        if C == 0:
            return 0.3

        ear = (A + B) / (2.0 * C)

        # Sanity check: EAR should be in valid range
        if ear < 0.10 or ear > 0.60:
            return 0.3  # Return neutral value if calculation seems wrong

        return ear

    def _calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for mouth openness detection.

        MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)

        For InsightFace 106-landmark 20-point mouth model:
        - Uses specific points for vertical (top-bottom) and horizontal (left-right) distances
        - Higher MAR indicates mouth is open (talking, yawning)

        Typical values:
        - < 0.15: Closed mouth (BEST for cut points)
        - 0.15-0.30: Slightly open
        - > 0.30: Open mouth (talking/yawning - AVOID for cut points)
        """
        if len(mouth_landmarks) < 12:
            return 0.2  # Default low activity (closed mouth)

        # Vertical mouth distances
        A = self._euclidean_distance(mouth_landmarks[2], mouth_landmarks[10])
        B = self._euclidean_distance(mouth_landmarks[4], mouth_landmarks[8])

        # Horizontal mouth distance
        C = self._euclidean_distance(mouth_landmarks[0], mouth_landmarks[6])

        if C == 0:
            return 0.2

        mar = (A + B) / (2.0 * C)

        # Sanity check: MAR should be in valid range
        # If calculation is way off, return default closed-mouth value
        if mar < 0.0 or mar > 1.5:
            return 0.2

        return mar

    def _calculate_center_distance(self, bbox: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Calculate normalized distance from frame center to face center.

        Args:
            bbox: Face bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width)

        Returns:
            float: Normalized distance from center (0 = at center, 1 = at edge)
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
        normalized_distance = min(distance / 0.707, 1.0)

        return float(normalized_distance)

    def _calculate_center_weight(self, center_distance: float) -> float:
        """
        Calculate weight for center-weighted averaging based on distance from center.

        Args:
            center_distance: Normalized distance from center (0-1)

        Returns:
            float: Weight value (higher for faces closer to center)
        """
        # Apply center weighting: weight = 1 / (1 + strength * distance)
        weight = 1.0 / (1.0 + self.center_weighting_strength * center_distance)
        return float(weight)

    def _aggregate_multi_face_metrics(
        self,
        face_features_list: list,
        weights: list
    ) -> Tuple[float, float, float]:
        """
        Aggregate metrics from multiple faces using weighted averaging.

        Args:
            face_features_list: List of tuples (eye_openness, expression_activity, pose_deviation)
            weights: List of weights corresponding to each face

        Returns:
            Tuple of aggregated (eye_openness, expression_activity, pose_deviation)
        """
        if not face_features_list or not weights:
            return 0.3, 0.5, 0.5  # Default penalty values

        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total_weight for w in weights]

        # Weighted average for each metric
        eye_openness = sum(f[0] * w for f, w in zip(face_features_list, normalized_weights))
        expression = sum(f[1] * w for f, w in zip(face_features_list, normalized_weights))
        pose = sum(f[2] * w for f, w in zip(face_features_list, normalized_weights))

        return float(eye_openness), float(expression), float(pose)

    def extract_all_faces(self, frame: np.ndarray) -> Tuple[int, List, Tuple[float, float, float]]:
        """
        Extract features from all detected faces in the frame.

        Returns:
            Tuple containing:
            - num_faces: Number of faces detected
            - face_features_list: List of FaceFeatures objects (from models.py)
            - aggregated_metrics: Tuple of (eye_openness, expression_activity, pose_deviation)
        """
        from .models import FaceFeatures

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)
        frame_shape = frame.shape[:2]  # (height, width)

        # Filter faces by confidence
        if faces:
            faces = [f for f in faces if hasattr(f, 'det_score') and f.det_score >= self.min_face_confidence]

        if not faces or len(faces) == 0:
            # No faces detected - return default values
            return 0, [], (0.3, 0.5, 0.5)

        face_features_list = []
        face_metrics = []  # List of (eye_openness, expression, pose) tuples
        weights = []

        for face_idx, face in enumerate(faces):
            # Calculate center distance and weight
            bbox = face.bbox
            center_distance = self._calculate_center_distance(bbox, frame_shape)
            center_weight = self._calculate_center_weight(center_distance)

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
        aggregated = self._aggregate_multi_face_metrics(face_metrics, weights)

        return len(faces), face_features_list, aggregated

    def _extract_single_face_eye_openness(self, face) -> float:
        """Extract eye openness for a single face object."""
        # Check if 106 landmarks are available
        if self.has_106_landmarks and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
                left_eye_points = landmarks[LEFT_EYE_INDICES]
                right_eye_points = landmarks[RIGHT_EYE_INDICES]

                left_ear = self._calculate_ear(left_eye_points)
                right_ear = self._calculate_ear(right_eye_points)

                avg_ear = (left_ear + right_ear) / 2.0
                return avg_ear

        return 0.3  # Default neutral value

    def _extract_single_face_expression(self, face) -> float:
        """Extract expression activity for a single face object."""
        # Check if 106 landmarks are available
        if self.has_106_landmarks and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(MOUTH_OUTER_INDICES):
                mouth_points = landmarks[MOUTH_OUTER_INDICES]
                mar = self._calculate_mar(mouth_points)

                # MAR directly represents expression activity
                # No need to normalize here - will be done in scorer.py
                # Lower MAR = less activity (better for cut points)
                return mar

        return 0.2  # Default low activity (closed mouth)

    def _extract_single_face_pose(self, face, bbox: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """Extract pose deviation for a single face object."""
        frame_h, frame_w = frame_shape
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2

        # Normalized deviation from center
        center_x_dev = abs(face_center_x - frame_w/2) / (frame_w/2)
        center_y_dev = abs(face_center_y - frame_h/2) / (frame_h/2)

        deviation = (center_x_dev + center_y_dev) / 2.0
        return float(deviation)

    def extract_eye_openness(self, frame: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) averaged across both eyes.

        Returns:
            float: Eye aspect ratio. Normal range is ~0.25-0.35.
                   - < 0.2: Eyes closing/blinking
                   - 0.25-0.35: Normal open eyes (BEST for cut points)
                   - > 0.4: Eyes wide open (surprise/speaking)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)

        if not faces or len(faces) == 0:
            return 0.3  # Default neutral value

        face = faces[0]  # Use first detected face

        # Check if 106 landmarks are available
        if self.has_106_landmarks and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
                left_eye_points = landmarks[LEFT_EYE_INDICES]
                right_eye_points = landmarks[RIGHT_EYE_INDICES]

                left_ear = self._calculate_ear(left_eye_points)
                right_ear = self._calculate_ear(right_eye_points)

                avg_ear = (left_ear + right_ear) / 2.0
                return avg_ear

        # Fallback: use 5-point landmarks if available
        if hasattr(face, 'kps') and face.kps is not None:
            # Simple estimation from 5-point landmarks
            # This is less accurate but better than nothing
            return 0.3  # Default neutral value

        return 0.3

    def extract_motion_magnitude(self, frame: np.ndarray) -> float:
        """
        Calculate optical flow magnitude between current and previous frame.

        Returns:
            float: Average motion magnitude. Lower is better for cut points.
                   - < 0.5: Very stable (BEST)
                   - 0.5-2.0: Some movement
                   - > 2.0: High motion (AVOID)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Use median instead of mean to reduce outlier impact
        avg_magnitude = float(np.median(magnitude))

        self.prev_frame_gray = gray
        return avg_magnitude

    def extract_expression_activity(self, frame: np.ndarray) -> float:
        """
        Calculate facial expression activity level using InsightFace landmarks.

        Uses Mouth Aspect Ratio (MAR) as the primary indicator.

        Returns:
            float: Expression activity score (MAR value). Lower is better for cut points.
                   - < 0.15: Neutral expression (BEST)
                   - 0.15-0.30: Slight expression
                   - > 0.30: Active expression - speaking/emoting (AVOID)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)

        if not faces or len(faces) == 0:
            return 0.5  # High penalty if no face detected

        face = faces[0]

        # Check if 106 landmarks are available
        if self.has_106_landmarks and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(int)

            if len(landmarks) > max(MOUTH_OUTER_INDICES):
                mouth_points = landmarks[MOUTH_OUTER_INDICES]
                mar = self._calculate_mar(mouth_points)

                # Return raw MAR value - will be normalized in scorer.py
                # Lower MAR = better for cut points (closed mouth)
                return mar

        return 0.2  # Default low activity if can't analyze

    def extract_pose_deviation(self, frame: np.ndarray,
                              median_pose: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate head pose deviation using InsightFace landmarks.

        Returns:
            float: Pose deviation. Lower is better for cut points.
                   - < 0.1: Stable, centered pose (BEST)
                   - 0.1-0.3: Slight head tilt/turn
                   - > 0.3: Significant pose change (AVOID)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_frame)

        if not faces or len(faces) == 0:
            return 0.5  # High penalty if no face

        face = faces[0]

        # Try to use pose estimation from face detection
        # InsightFace provides bbox which we can use to estimate rough pose
        if hasattr(face, 'bbox'):
            bbox = face.bbox
            # Simple heuristic: centered faces are better
            # Assume frame center is ideal
            frame_h, frame_w = frame.shape[:2]
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2

            # Normalized deviation from center
            center_x_dev = abs(face_center_x - frame_w/2) / (frame_w/2)
            center_y_dev = abs(face_center_y - frame_h/2) / (frame_h/2)

            deviation = (center_x_dev + center_y_dev) / 2.0
            return float(deviation)

        return 0.3  # Default moderate deviation

    def extract_visual_sharpness(self, frame: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Higher values indicate sharper, clearer images.

        Returns:
            float: Sharpness score. Higher is better for cut points.
                   - < 50: Blurry (AVOID)
                   - 50-200: Acceptable
                   - > 200: Sharp (BEST)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculate variance as sharpness metric
        variance = laplacian.var()

        return float(variance)

    def reset(self):
        """Reset internal state (optical flow tracking)."""
        self.prev_frame_gray = None
