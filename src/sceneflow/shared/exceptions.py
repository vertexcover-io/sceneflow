"""Custom exceptions for SceneFlow.

This module defines domain-specific exceptions that provide better
error handling and more informative error messages than generic exceptions.
"""


class SceneFlowError(Exception):
    """Base exception for all SceneFlow errors."""

    pass


class VideoError(SceneFlowError):
    """Base exception for video-related errors."""

    pass


class VideoNotFoundError(VideoError):
    """Raised when video file cannot be found."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Video file not found: {path}")


class VideoOpenError(VideoError):
    """Raised when video file cannot be opened."""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.reason = reason
        msg = f"Failed to open video: {path}"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg)


class VideoPropertiesError(VideoError):
    """Raised when video has invalid properties (fps, frame count, etc.)."""

    def __init__(self, path: str, details: str):
        self.path = path
        self.details = details
        super().__init__(f"Invalid video properties for {path}: {details}")


class VideoDownloadError(VideoError):
    """Raised when video download fails."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to download video from {url}: {reason}")


class UnsupportedFormatError(VideoError):
    """Raised when video format is not supported."""

    def __init__(self, path: str, extension: str):
        self.path = path
        self.extension = extension
        super().__init__(
            f"Unsupported video format '{extension}' for file: {path}"
        )


class FeatureExtractionError(SceneFlowError):
    """Base exception for feature extraction errors."""

    pass


class NoFaceDetectedError(FeatureExtractionError):
    """Raised when no face is detected in a frame."""

    def __init__(self, frame_index: int = None):
        self.frame_index = frame_index
        msg = "No face detected in frame"
        if frame_index is not None:
            msg += f" {frame_index}"
        super().__init__(msg)


class LandmarkDetectionError(FeatureExtractionError):
    """Raised when facial landmark detection fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Landmark detection failed: {reason}")


class InsightFaceError(FeatureExtractionError):
    """Raised when InsightFace model fails to load or execute."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"InsightFace error: {reason}")


class SpeechDetectionError(SceneFlowError):
    """Base exception for speech detection errors."""

    pass


class VADModelError(SpeechDetectionError):
    """Raised when Silero VAD model fails to load."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Failed to load Silero VAD model: {reason}")


class AudioLoadError(SpeechDetectionError):
    """Raised when audio cannot be loaded from video."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load audio from {path}: {reason}")


class NoSpeechDetectedError(SpeechDetectionError):
    """Raised when no speech is detected in audio."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"No speech detected in video: {path}")


class ConfigurationError(SceneFlowError):
    """Base exception for configuration errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid configuration: {reason}")


class WeightSumError(ConfigurationError):
    """Raised when scoring weights don't sum to 1.0."""

    def __init__(self, total: float):
        self.total = total
        super().__init__(
            f"Scoring weights must sum to 1.0, got {total:.4f}"
        )


class WindowSizeError(ConfigurationError):
    """Raised when window size is not odd."""

    def __init__(self, window_name: str, size: int):
        self.window_name = window_name
        self.size = size
        super().__init__(
            f"{window_name} must be odd, got {size}"
        )


class RankingError(SceneFlowError):
    """Base exception for ranking errors."""

    pass


class NoValidFramesError(RankingError):
    """Raised when no valid frames found for ranking."""

    def __init__(self, reason: str = ""):
        msg = "No valid frames found for ranking"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class InsufficientFramesError(RankingError):
    """Raised when insufficient frames available for analysis."""

    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient frames: required {required}, available {available}"
        )


class SecurityError(SceneFlowError):
    """Base exception for security-related errors."""

    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attempt is detected."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Path traversal detected in: {path}")


class InvalidURLError(SecurityError):
    """Raised when URL is invalid or suspicious."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid URL '{url}': {reason}")


class APIError(SceneFlowError):
    """Base exception for external API errors."""

    pass


class LLMError(APIError):
    """Raised when LLM API call fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"LLM API error: {reason}")


class AirtableError(APIError):
    """Raised when Airtable API call fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Airtable API error: {reason}")


class MissingAPIKeyError(APIError):
    """Raised when required API key is missing."""

    def __init__(self, service: str, env_var: str):
        self.service = service
        self.env_var = env_var
        super().__init__(
            f"{service} API key not found. Set {env_var} environment variable."
        )


class FFmpegError(SceneFlowError):
    """Base exception for FFmpeg errors."""

    pass


class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg executable is not found."""

    def __init__(self):
        super().__init__(
            "FFmpeg not found. Please install FFmpeg to use video cutting features."
        )


class FFmpegExecutionError(FFmpegError):
    """Raised when FFmpeg command execution fails."""

    def __init__(self, command: str, stderr: str):
        self.command = command
        self.stderr = stderr
        super().__init__(f"FFmpeg command failed: {stderr}")
