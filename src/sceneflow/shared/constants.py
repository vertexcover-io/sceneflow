"""Application-wide constants for SceneFlow.

This module contains all magic numbers, thresholds, and configuration
constants used throughout the application. Centralizing these values
improves maintainability and makes it easier to tune the system.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class DetectionSize:
    """Face detection size configuration."""

    width: int
    height: int


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

    MIN_VALID: float = 0.08
    MAX_VALID: float = 0.35
    NORMAL_MIN: float = 0.25
    NORMAL_MAX: float = 0.32
    DEFAULT: float = 0.28


@dataclass(frozen=True)
class MARConstants:
    """Mouth Aspect Ratio (MAR) constants for expression detection.

    MAR is used for detecting mouth openness (talking, yawning, etc.).
    Formula: MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)
    """

    MIN_VALID: float = 0.15
    MAX_VALID: float = 1.5
    CLOSED_MIN: float = 0.20
    CLOSED_MAX: float = 0.35
    DEFAULT: float = 0.27


@dataclass(frozen=True)
class InsightFaceConstants:
    """InsightFace model configuration."""

    DEFAULT_DET_SIZE: DetectionSize = DetectionSize(640, 640)
    MIN_FACE_CONFIDENCE: float = 0.5
    LEFT_EYE_START: int = 37
    LEFT_EYE_END: int = 43
    RIGHT_EYE_START: int = 43
    RIGHT_EYE_END: int = 49
    MOUTH_OUTER_START: int = 52
    MOUTH_OUTER_END: int = 72


@dataclass(frozen=True)
class FFmpegConstants:
    """FFmpeg command configuration."""

    VIDEO_CODEC: str = 'libx264'
    AUDIO_CODEC: str = 'aac'
    TIMEOUT_SECONDS: int = 300


# Create singleton instances for easy import
VIDEO = VideoConstants()
VAD = VADConstants()
EAR = EARConstants()
MAR = MARConstants()
INSIGHTFACE = InsightFaceConstants()
FFMPEG = FFmpegConstants()


# Type aliases for clarity
# VideoPath = str
# Timestamp = float
# FrameIndex = int
# Score = float
