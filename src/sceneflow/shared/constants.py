"""Application-wide constants for SceneFlow.

This module contains all magic numbers, thresholds, and configuration
constants used throughout the application. Centralizing these values
improves maintainability and makes it easier to tune the system.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class VideoConstants:
    """Video processing and download constants."""

    # Download settings
    DOWNLOAD_TIMEOUT_SECONDS: int = 30
    DOWNLOAD_CHUNK_SIZE_BYTES: int = 8192

    # Video encoding quality
    JPEG_QUALITY_DEFAULT: int = 85
    JPEG_QUALITY_HIGH: int = 95

    # Supported video formats
    SUPPORTED_EXTENSIONS: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')


@dataclass(frozen=True)
class VADConstants:
    """Voice Activity Detection (Silero VAD) constants."""

    # Audio processing
    TARGET_SAMPLE_RATE: int = 16000  # Hz - optimal for Silero VAD

    # VAD thresholds
    THRESHOLD: float = 0.7  # Speech detection threshold
    NEG_THRESHOLD: float = 0.5  # Silence detection threshold

    # Timing parameters
    MIN_SILENCE_DURATION_MS: int = 0
    SPEECH_PAD_MS: int = 0
    TIME_RESOLUTION: int = 4  # Decimal places for timestamp precision

    # Confidence calculation
    MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE: float = 0.5  # seconds


@dataclass(frozen=True)
class EARConstants:
    """Eye Aspect Ratio (EAR) constants and thresholds.

    EAR is used for detecting eye openness and blink states.
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Reference:
        Soukupová & Čech (2016). "Real-Time Eye Blink Detection using
        Facial Landmarks." 21st Computer Vision Winter Workshop.
    """

    # Valid range boundaries
    MIN_VALID: float = 0.10  # Below this indicates closed/blinking eyes
    MAX_VALID: float = 0.60  # Above this indicates unnaturally wide eyes

    # Normal open eyes range
    NORMAL_MIN: float = 0.25  # Lower bound of normal range
    NORMAL_MAX: float = 0.35  # Upper bound of normal range

    # Default value when calculation fails or landmarks missing
    DEFAULT: float = 0.30  # Neutral value representing normal open eyes


@dataclass(frozen=True)
class MARConstants:
    """Mouth Aspect Ratio (MAR) constants for expression detection.

    MAR is used for detecting mouth openness (talking, yawning, etc.).
    Formula: MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)
    """

    # MAR interpretation thresholds
    CLOSED_MOUTH_MAX: float = 0.15  # Below this: closed mouth (BEST)
    SLIGHTLY_OPEN_MAX: float = 0.30  # Between 0.15-0.30: slightly open
    # Above 0.30: open mouth (talking/yawning - AVOID)

    # Valid range
    MIN_VALID: float = 0.0
    MAX_VALID: float = 1.5

    # Default when calculation fails
    DEFAULT: float = 0.2  # Represents closed/neutral mouth


@dataclass(frozen=True)
class MotionConstants:
    """Optical flow motion detection constants."""

    # Motion magnitude interpretation (pixels)
    VERY_STABLE_MAX: float = 0.5  # Below this: very stable (BEST)
    SOME_MOVEMENT_MAX: float = 2.0  # 0.5-2.0: some movement
    # Above 2.0: high motion (AVOID)

    # Optical flow parameters
    PYR_SCALE: float = 0.5
    LEVELS: int = 3
    WINSIZE: int = 15
    ITERATIONS: int = 3
    POLY_N: int = 5
    POLY_SIGMA: float = 1.2


@dataclass(frozen=True)
class SharpnessConstants:
    """Visual sharpness (Laplacian variance) constants."""

    # Sharpness interpretation thresholds
    BLURRY_MAX: float = 50.0  # Below this: blurry (AVOID)
    ACCEPTABLE_MIN: float = 50.0  # 50-200: acceptable
    SHARP_MIN: float = 200.0  # Above 200: sharp (BEST)


@dataclass(frozen=True)
class EnergyRefinementConstants:
    """Audio energy refinement constants."""

    # Energy drop detection
    DEFAULT_THRESHOLD_DB: float = 8.0  # Minimum dB drop to detect speech end
    DEFAULT_LOOKBACK_FRAMES: int = 20  # Frames to search backward from VAD
    DEFAULT_MIN_SILENCE_FRAMES: int = 1  # Consecutive low-energy frames required

    # Tolerance for silence verification
    SILENCE_TOLERANCE_DB: float = 3.0  # Allow 3 dB variation in silence


@dataclass(frozen=True)
class InsightFaceConstants:
    """InsightFace model configuration."""

    # Detection parameters
    DEFAULT_DET_SIZE: tuple = (640, 640)  # Detection size for face detection
    MIN_FACE_CONFIDENCE: float = 0.5  # Minimum confidence for face detection

    # Landmark indices for 106-point model
    LEFT_EYE_START: int = 35
    LEFT_EYE_END: int = 42  # 7 points
    RIGHT_EYE_START: int = 42
    RIGHT_EYE_END: int = 49  # 7 points
    MOUTH_OUTER_START: int = 52
    MOUTH_OUTER_END: int = 72  # 20 points

    # Multi-face handling
    DEFAULT_CENTER_WEIGHTING_STRENGTH: float = 1.0
    MAX_DISTANCE_NORMALIZED: float = 0.707  # sqrt(0.5^2 + 0.5^2) = diagonal


@dataclass(frozen=True)
class RankingConstants:
    """Frame ranking algorithm constants."""

    # Default scoring weights (must sum to 1.0)
    DEFAULT_EYE_OPENNESS_WEIGHT: float = 0.20
    DEFAULT_MOTION_STABILITY_WEIGHT: float = 0.25
    DEFAULT_EXPRESSION_NEUTRALITY_WEIGHT: float = 0.30
    DEFAULT_POSE_STABILITY_WEIGHT: float = 0.15
    DEFAULT_VISUAL_SHARPNESS_WEIGHT: float = 0.10

    # Context window (must be odd)
    DEFAULT_CONTEXT_WINDOW_SIZE: int = 5

    # Quality gating
    DEFAULT_QUALITY_GATE_PERCENTILE: float = 75.0
    MIN_QUALITY_PENALTY: float = 0.5  # At least 50% score for poor quality

    # Stability analysis (must be odd)
    DEFAULT_STABILITY_WINDOW_SIZE: int = 5
    MAX_STABILITY_BOOST_MULTIPLIER: float = 0.5  # 50% boost for stable sequences

    # Variance normalization
    VARIANCE_PERCENTILE_THRESHOLD: float = 95.0


@dataclass(frozen=True)
class NormalizationConstants:
    """Normalization algorithm constants."""

    # Minimum variance threshold
    MIN_VARIANCE_THRESHOLD: float = 1e-9

    # Default neutral value when normalization fails
    DEFAULT_NORMALIZED_VALUE: float = 0.5

    # Robust normalization percentiles
    ROBUST_PERCENTILE_LOW: float = 5.0
    ROBUST_PERCENTILE_HIGH: float = 95.0

    # Z-score clipping
    ZSCORE_CLIP_STD: float = 3.0

    # Sigmoid steepness
    DEFAULT_SIGMOID_STEEPNESS: float = 10.0


@dataclass(frozen=True)
class FFmpegConstants:
    """FFmpeg command configuration."""

    # Video codec settings
    VIDEO_CODEC: str = 'libx264'
    AUDIO_CODEC: str = 'aac'

    # Timeout for ffmpeg operations
    TIMEOUT_SECONDS: int = 300  # 5 minutes


@dataclass(frozen=True)
class LLMConstants:
    """LLM (GPT-4o) frame selection constants."""

    # Model configuration
    MODEL_NAME: str = 'gpt-4o'
    MAX_TOKENS: int = 10
    TEMPERATURE: float = 0.0  # Deterministic selection

    # Image quality
    IMAGE_DETAIL: str = 'high'

    # Selection parameters
    TOP_N_CANDIDATES: int = 5  # Number of frames to send to LLM


# Create singleton instances for easy import
VIDEO = VideoConstants()
VAD = VADConstants()
EAR = EARConstants()
MAR = MARConstants()
MOTION = MotionConstants()
SHARPNESS = SharpnessConstants()
ENERGY = EnergyRefinementConstants()
INSIGHTFACE = InsightFaceConstants()
RANKING = RankingConstants()
NORMALIZATION = NormalizationConstants()
FFMPEG = FFmpegConstants()
LLM = LLMConstants()


# Type aliases for clarity
VideoPath = str
Timestamp = float
FrameIndex = int
Score = float
