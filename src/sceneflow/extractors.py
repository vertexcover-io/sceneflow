import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class FeatureExtractor:
    """
    Extracts facial and visual features for determining optimal video cut points.

    Key principles:
    - Lower expression activity = better cut point (neutral face)
    - Normal eye openness = better (not blinking, not wide open)
    - Lower motion = better (stable frame)
    - Stable pose = better (facing camera directly)
    - Higher sharpness = better (clear image)
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # MediaPipe Face Mesh landmark indices
        # Eyes
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

        # Mouth - key points for aspect ratio
        self.MOUTH_TOP = 13      # Upper lip center
        self.MOUTH_BOTTOM = 14   # Lower lip center
        self.MOUTH_LEFT = 78     # Left corner
        self.MOUTH_RIGHT = 308   # Right corner

        # Eyebrows
        self.LEFT_EYEBROW_TOP = 70
        self.RIGHT_EYEBROW_TOP = 300
        self.LEFT_EYE_CENTER = 159
        self.RIGHT_EYE_CENTER = 386

        # Pose reference points
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.FOREHEAD = 10

        # Optical flow tracking
        self.prev_frame_gray = None

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
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.3  # Default neutral value

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES, w, h)

        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear

    def _calculate_ear(self, landmarks, indices, w, h) -> float:
        """Calculate Eye Aspect Ratio for a single eye."""
        points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in indices
        ])

        # Vertical distances
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        # Horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])

        if horizontal == 0:
            return 0.3

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

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
        Calculate facial expression activity level.

        Combines multiple facial features to detect active expressions:
        - Mouth openness (speaking, laughing)
        - Eyebrow raise (surprise, emphasis)
        - Jaw drop (speaking, yawning)
        - Mouth stretch (smiling, wide expressions)

        Returns:
            float: Expression activity score. Lower is better for cut points.
                   - < 0.1: Neutral expression (BEST)
                   - 0.1-0.3: Slight expression
                   - > 0.3: Active expression - speaking/emoting (AVOID)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.5  # High penalty if no face detected

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Calculate individual expression components
        mouth_aspect_ratio = self._calculate_mouth_aspect_ratio(landmarks, w, h)
        eyebrow_raise = self._calculate_eyebrow_raise(landmarks, w, h)
        jaw_openness = self._calculate_jaw_openness(landmarks, w, h)
        mouth_stretch = self._calculate_mouth_stretch(landmarks, w, h)

        # Weighted combination - mouth is most important for detecting speech
        activity = (
            0.45 * mouth_aspect_ratio +   # Mouth vertical opening
            0.25 * jaw_openness +          # Jaw drop
            0.20 * eyebrow_raise +         # Eyebrow movement
            0.10 * mouth_stretch           # Mouth horizontal stretch
        )

        return activity

    def _calculate_mouth_aspect_ratio(self, landmarks, w, h) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) - vertical opening vs horizontal width.

        Higher values indicate open mouth (speaking, surprise, etc).
        """
        # Vertical distance (top to bottom lip)
        top_lip = np.array([landmarks[self.MOUTH_TOP].x * w,
                           landmarks[self.MOUTH_TOP].y * h])
        bottom_lip = np.array([landmarks[self.MOUTH_BOTTOM].x * w,
                              landmarks[self.MOUTH_BOTTOM].y * h])
        vertical_dist = np.linalg.norm(top_lip - bottom_lip)

        # Horizontal distance (mouth width)
        left_corner = np.array([landmarks[self.MOUTH_LEFT].x * w,
                               landmarks[self.MOUTH_LEFT].y * h])
        right_corner = np.array([landmarks[self.MOUTH_RIGHT].x * w,
                                landmarks[self.MOUTH_RIGHT].y * h])
        horizontal_dist = np.linalg.norm(left_corner - right_corner)

        if horizontal_dist == 0:
            return 0.0

        # Normalize by mouth width
        mar = vertical_dist / horizontal_dist

        # Typical values: closed ~0.05-0.15, open ~0.3-0.6+
        return mar

    def _calculate_eyebrow_raise(self, landmarks, w, h) -> float:
        """
        Calculate eyebrow raise amount.

        Higher values indicate raised eyebrows (surprise, emphasis while speaking).
        """
        # Distance from eyebrow to eye
        left_brow_y = landmarks[self.LEFT_EYEBROW_TOP].y * h
        left_eye_y = landmarks[self.LEFT_EYE_CENTER].y * h
        right_brow_y = landmarks[self.RIGHT_EYEBROW_TOP].y * h
        right_eye_y = landmarks[self.RIGHT_EYE_CENTER].y * h

        # Face height for normalization
        face_height = abs(landmarks[self.FOREHEAD].y - landmarks[self.CHIN].y) * h

        if face_height == 0:
            return 0.0

        # Calculate normalized distances (y increases downward, so brow - eye)
        left_raise = abs(left_brow_y - left_eye_y) / face_height
        right_raise = abs(right_brow_y - right_eye_y) / face_height

        avg_raise = (left_raise + right_raise) / 2.0

        # Typical values: neutral ~0.05-0.08, raised ~0.1-0.15+
        return float(avg_raise)

    def _calculate_jaw_openness(self, landmarks, w, h) -> float:
        """
        Calculate jaw drop/openness.

        Measures vertical distance from nose to chin, normalized by face height.
        Increases when mouth opens wide.
        """
        nose = np.array([landmarks[self.NOSE_TIP].x * w,
                        landmarks[self.NOSE_TIP].y * h])
        chin = np.array([landmarks[self.CHIN].x * w,
                        landmarks[self.CHIN].y * h])
        forehead = np.array([landmarks[self.FOREHEAD].x * w,
                            landmarks[self.FOREHEAD].y * h])

        face_height = np.linalg.norm(forehead - chin)

        if face_height == 0:
            return 0.0

        nose_to_chin = np.linalg.norm(nose - chin)
        jaw_ratio = nose_to_chin / face_height

        return jaw_ratio

    def _calculate_mouth_stretch(self, landmarks, w, h) -> float:
        """
        Calculate horizontal mouth stretch (smiling, wide expressions).

        Compares current mouth width to typical proportions.
        """
        left_corner = np.array([landmarks[self.MOUTH_LEFT].x * w,
                               landmarks[self.MOUTH_LEFT].y * h])
        right_corner = np.array([landmarks[self.MOUTH_RIGHT].x * w,
                                landmarks[self.MOUTH_RIGHT].y * h])
        mouth_width = np.linalg.norm(left_corner - right_corner)

        # Face width for normalization
        left_face = np.array([landmarks[234].x * w, landmarks[234].y * h])
        right_face = np.array([landmarks[454].x * w, landmarks[454].y * h])
        face_width = np.linalg.norm(left_face - right_face)

        if face_width == 0:
            return 0.0

        # Ratio of mouth width to face width
        stretch_ratio = mouth_width / face_width

        # Typical neutral: ~0.35-0.45, wide smile: ~0.5+
        # Return deviation from neutral (0.4)
        deviation = abs(stretch_ratio - 0.4)

        return float(deviation)

    def extract_pose_deviation(self, frame: np.ndarray,
                              median_pose: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate head pose deviation.

        Measures pitch, yaw, and roll angles. Lower deviation = more stable/centered pose.

        Args:
            frame: Input frame
            median_pose: Optional reference pose (pitch, yaw, roll) to compare against

        Returns:
            float: Pose deviation. Lower is better for cut points.
                   - < 0.1: Stable, centered pose (BEST)
                   - 0.1-0.3: Slight head tilt/turn
                   - > 0.3: Significant pose change (AVOID)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.5  # High penalty if no face

        landmarks = results.multi_face_landmarks[0].landmark

        # Extract 3D coordinates
        nose_tip = np.array([landmarks[self.NOSE_TIP].x,
                            landmarks[self.NOSE_TIP].y,
                            landmarks[self.NOSE_TIP].z])
        chin = np.array([landmarks[self.CHIN].x,
                        landmarks[self.CHIN].y,
                        landmarks[self.CHIN].z])
        left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

        # Calculate pose angles
        pitch = np.arctan2(nose_tip[1] - chin[1], nose_tip[2] - chin[2])
        yaw = np.arctan2(left_eye[0] - right_eye[0], left_eye[2] - right_eye[2])
        roll = np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])

        if median_pose is None:
            # Return absolute pose deviation
            deviation = np.sqrt(pitch**2 + yaw**2 + roll**2)
        else:
            # Return deviation from median pose
            median_pitch, median_yaw, median_roll = median_pose
            deviation = np.sqrt(
                (pitch - median_pitch)**2 +
                (yaw - median_yaw)**2 +
                (roll - median_roll)**2
            )

        return float(deviation)

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
