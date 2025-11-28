"""Shared foundational modules.

This package contains core data structures, configuration, constants,
and exceptions used throughout SceneFlow.
"""

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.constants import (
    INSIGHTFACE,
    EAR,
    MAR,
    MOTION,
    SHARPNESS,
    RANKING,
    VAD,
    FFMPEG,
)
from sceneflow.shared.exceptions import (
    SceneFlowError,
    VideoError,
    VideoNotFoundError,
    VideoOpenError,
    VideoDownloadError,
    NoValidFramesError,
    InsightFaceError,
    LandmarkDetectionError,
    VADModelError,
    AudioLoadError,
    NoSpeechDetectedError,
    FFmpegNotFoundError,
    FFmpegExecutionError,
    InvalidConfigError,
    AirtableError,
)
from sceneflow.shared.models import (
    FaceFeatures,
    FrameFeatures,
    FrameScore,
    RankedFrame,
)

__all__ = [
    # Configuration
    'RankingConfig',

    # Constants
    'INSIGHTFACE',
    'EAR',
    'MAR',
    'MOTION',
    'SHARPNESS',
    'RANKING',
    'VAD',
    'FFMPEG',

    # Exceptions
    'SceneFlowError',
    'VideoError',
    'VideoNotFoundError',
    'VideoOpenError',
    'VideoDownloadError',
    'NoValidFramesError',
    'InsightFaceError',
    'LandmarkDetectionError',
    'VADModelError',
    'AudioLoadError',
    'NoSpeechDetectedError',
    'FFmpegNotFoundError',
    'FFmpegExecutionError',
    'InvalidConfigError',
    'AirtableError',

    # Models
    'FaceFeatures',
    'FrameFeatures',
    'FrameScore',
    'RankedFrame',
]
