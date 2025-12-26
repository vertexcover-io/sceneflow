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
        super().__init__(f"Unsupported video format '{extension}' for file: {path}")


class FeatureExtractionError(SceneFlowError):
    """Base exception for feature extraction errors."""

    pass


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


class NoValidFramesError(SceneFlowError):
    """Raised when no valid frames found for ranking."""

    def __init__(self, reason: str = ""):
        msg = "No valid frames found for ranking"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


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


class FFmpegError(SceneFlowError):
    """Base exception for FFmpeg errors."""

    pass


class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg executable is not found."""

    def __init__(self):
        super().__init__("FFmpeg not found. Please install FFmpeg to use video cutting features.")


class FFmpegExecutionError(FFmpegError):
    """Raised when FFmpeg command execution fails."""

    def __init__(self, command: str, stderr: str):
        self.command = command
        self.stderr = stderr
        super().__init__(f"FFmpeg command failed: {stderr}")
