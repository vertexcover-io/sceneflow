"""Video utilities for downloading, processing, and metadata extraction.

This module provides shared utilities for video operations to eliminate
code duplication and provide consistent error handling.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from contextlib import contextmanager

import cv2
import requests

from sceneflow.shared.constants import VIDEO
from sceneflow.shared.exceptions import (
    VideoNotFoundError,
    VideoOpenError,
    VideoPropertiesError,
    VideoDownloadError,
    InvalidURLError,
    UnsupportedFormatError,
)
from sceneflow.shared.models import VideoProperties

logger = logging.getLogger(__name__)


def is_url(source: str) -> bool:
    """
    Check if the source string is a URL.

    Args:
        source: Path or URL string to check

    Returns:
        True if source is a URL, False otherwise

    Example:
        >>> is_url("https://example.com/video.mp4")
        True
        >>> is_url("/path/to/video.mp4")
        False
    """
    return source.startswith(('http://', 'https://'))


def validate_video_path(path: str) -> Path:
    """
    Validate and sanitize video file path.

    Args:
        path: Path to video file

    Returns:
        Validated and resolved Path object

    Raises:
        VideoNotFoundError: If file doesn't exist
        PathTraversalError: If path contains traversal attempts
        UnsupportedFormatError: If file extension not supported

    Example:
        >>> path = validate_video_path("/videos/sample.mp4")
        >>> print(path.exists())
        True
    """
    from sceneflow.shared.exceptions import PathTraversalError

    video_path = Path(path).resolve()

    # Check for path traversal
    if '..' in video_path.parts:
        raise PathTraversalError(str(path))

    # Check file exists
    if not video_path.exists():
        raise VideoNotFoundError(str(path))

    # Check extension
    if video_path.suffix.lower() not in VIDEO.SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(str(path), video_path.suffix)

    return video_path


def validate_video_url(url: str) -> str:
    """
    Validate video URL before download.

    Args:
        url: URL to validate

    Returns:
        Validated URL string

    Raises:
        InvalidURLError: If URL is invalid or suspicious

    Example:
        >>> url = validate_video_url("https://example.com/video.mp4")
        >>> print(url)
        https://example.com/video.mp4
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise InvalidURLError(url, f"Failed to parse URL: {e}")

    # Check scheme
    if parsed.scheme not in ('http', 'https'):
        raise InvalidURLError(url, f"Invalid scheme: {parsed.scheme}")

    # Check domain
    if not parsed.netloc:
        raise InvalidURLError(url, "Missing domain")

    # Check for suspicious characters
    suspicious_chars = ['<', '>', '"', "'", '\n', '\r', '\x00']
    if any(char in url for char in suspicious_chars):
        raise InvalidURLError(url, "Contains suspicious characters")

    # Warn if doesn't look like a video file
    path = parsed.path.lower()
    if not any(path.endswith(ext) for ext in VIDEO.SUPPORTED_EXTENSIONS):
        logger.warning(
            "URL does not appear to be a video file: %s",
            url
        )

    return url


def download_video(url: str) -> str:
    """
    Download video from URL to temporary directory using HTTP GET.

    Args:
        url: Direct video URL (e.g., .mp4, .avi file URLs)

    Returns:
        Path to downloaded video file

    Raises:
        VideoDownloadError: If download fails
        InvalidURLError: If URL is invalid

    Example:
        >>> path = download_video("https://example.com/video.mp4")
        >>> print(Path(path).exists())
        True
    """
    # Validate URL first
    url = validate_video_url(url)

    logger.info("Downloading video from URL: %s", url)

    # Create temp directory for downloaded video
    temp_dir = Path(tempfile.mkdtemp(prefix="sceneflow_"))

    # Extract filename from URL or use default
    url_path = Path(url.split('?')[0])  # Remove query params
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        # Download with streaming
        response = requests.get(
            url,
            stream=True,
            timeout=VIDEO.DOWNLOAD_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Write to file
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=VIDEO.DOWNLOAD_CHUNK_SIZE_BYTES):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug("Download progress: %.1f%%", progress)

        logger.info("Video downloaded successfully to: %s", output_path)
        return str(output_path)

    except requests.exceptions.Timeout:
        raise VideoDownloadError(
            url,
            f"Download timed out after {VIDEO.DOWNLOAD_TIMEOUT_SECONDS} seconds"
        )
    except requests.exceptions.ConnectionError as e:
        raise VideoDownloadError(url, f"Connection failed: {e}")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "unknown"
        raise VideoDownloadError(url, f"HTTP error {status_code}")
    except OSError as e:
        raise VideoDownloadError(url, f"File system error: {e}")
    except Exception as e:
        logger.exception("Unexpected error during download")
        raise VideoDownloadError(url, f"Unexpected error: {e}") from e


def cleanup_downloaded_video(video_path: str) -> None:
    """
    Delete downloaded video and its temporary directory.

    Args:
        video_path: Path to the downloaded video file

    Example:
        >>> cleanup_downloaded_video("/tmp/sceneflow_abc123/video.mp4")
    """
    try:
        video_file = Path(video_path)
        temp_dir = video_file.parent

        # Delete the video file
        if video_file.exists():
            video_file.unlink()
            logger.debug("Deleted downloaded video: %s", video_path)

        # Delete the temporary directory if it's empty or contains only the video
        if temp_dir.exists() and temp_dir.name.startswith("sceneflow_"):
            # Check if directory is empty
            remaining_files = list(temp_dir.iterdir())
            if not remaining_files:
                temp_dir.rmdir()
                logger.debug("Deleted temporary directory: %s", temp_dir)
            else:
                logger.warning(
                    "Temporary directory not empty, skipping deletion: %s",
                    temp_dir
                )

    except Exception as e:
        logger.warning("Failed to clean up downloaded video: %s", e)


@contextmanager
def temporary_video_download(url: str):
    """
    Context manager for downloading and auto-cleaning up video.

    Args:
        url: Video URL to download

    Yields:
        Path to downloaded video file

    Raises:
        VideoDownloadError: If download fails

    Example:
        >>> with temporary_video_download("https://example.com/video.mp4") as path:
        ...     process_video(path)
        # Video automatically cleaned up here
    """
    video_path = None
    try:
        video_path = download_video(url)
        yield video_path
    finally:
        if video_path:
            cleanup_downloaded_video(video_path)


class VideoCapture:
    """
    Context manager for OpenCV VideoCapture with automatic resource cleanup.

    This ensures video capture objects are always properly released,
    even when exceptions occur.

    Example:
        >>> with VideoCapture("/path/to/video.mp4") as cap:
        ...     fps = cap.get(cv2.CAP_PROP_FPS)
        ...     # Use cap...
        # Automatically released here
    """

    def __init__(self, video_path: str):
        """
        Initialize VideoCapture context manager.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> cv2.VideoCapture:
        """Open video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise VideoOpenError(
                self.video_path,
                "VideoCapture.isOpened() returned False"
            )
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
        return False  # Don't suppress exceptions


def get_video_properties(video_path: str) -> VideoProperties:
    """
    Extract video properties (fps, frame count, duration, dimensions).

    Args:
        video_path: Path to video file

    Returns:
        VideoProperties dataclass with fps, frame_count, duration, width, height

    Raises:
        VideoOpenError: If video cannot be opened
        VideoPropertiesError: If video has invalid properties

    Example:
        >>> props = get_video_properties("video.mp4")
        >>> print(f"Duration: {props.duration:.2f}s")
        Duration: 10.50s
    """
    with VideoCapture(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate properties
        if fps <= 0 or frame_count <= 0:
            raise VideoPropertiesError(
                video_path,
                f"Invalid fps={fps} or frame_count={frame_count}"
            )

        if width <= 0 or height <= 0:
            raise VideoPropertiesError(
                video_path,
                f"Invalid dimensions: {width}x{height}"
            )

        duration = frame_count / fps

        return VideoProperties(
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            width=width,
            height=height
        )


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        VideoOpenError: If video cannot be opened
        VideoPropertiesError: If video has invalid properties

    Example:
        >>> duration = get_video_duration("video.mp4")
        >>> print(f"{duration:.2f}s")
        10.50s
    """
    props = get_video_properties(video_path)
    return props.duration


def extract_frame(
    video_path: str,
    frame_index: int,
    jpeg_quality: int = VIDEO.JPEG_QUALITY_DEFAULT
) -> bytes:
    """
    Extract a specific frame from video as JPEG bytes.

    Args:
        video_path: Path to video file
        frame_index: Frame number to extract (0-indexed)
        jpeg_quality: JPEG compression quality (0-100, default: 85)

    Returns:
        JPEG image as bytes

    Raises:
        VideoOpenError: If video cannot be opened
        ValueError: If frame cannot be extracted

    Example:
        >>> frame_bytes = extract_frame("video.mp4", 100)
        >>> with open("frame.jpg", "wb") as f:
        ...     f.write(frame_bytes)
    """
    with VideoCapture(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            raise ValueError(
                f"Failed to extract frame {frame_index} from {video_path}"
            )

        # Encode frame as JPEG
        success, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )

        if not success:
            raise ValueError("Failed to encode frame as JPEG")

        return buffer.tobytes()


def extract_frame_at_timestamp(
    video_path: str,
    timestamp: float,
    jpeg_quality: int = VIDEO.JPEG_QUALITY_DEFAULT
) -> bytes:
    """
    Extract frame at specific timestamp from video as JPEG bytes.

    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        jpeg_quality: JPEG compression quality (0-100, default: 85)

    Returns:
        JPEG image as bytes

    Raises:
        VideoOpenError: If video cannot be opened
        ValueError: If frame cannot be extracted

    Example:
        >>> frame_bytes = extract_frame_at_timestamp("video.mp4", 5.5)
        >>> len(frame_bytes)
        123456
    """
    props = get_video_properties(video_path)
    frame_index = int(timestamp * props.fps)
    return extract_frame(video_path, frame_index, jpeg_quality)


def cut_video(
    video_path: str,
    cut_timestamp: float,
    output_path: Optional[str] = None
) -> str:
    """
    Cut video from start to the specified timestamp using FFmpeg.

    Args:
        video_path: Path to input video file
        cut_timestamp: Timestamp where to cut the video (in seconds)
        output_path: Optional custom output path for the cut video.
                    If None, saves to output/<video_name>_cut.mp4

    Returns:
        Path to the saved cut video

    Raises:
        FFmpegNotFoundError: If ffmpeg is not installed
        FFmpegExecutionError: If ffmpeg command fails

    Example:
        >>> output = cut_video("video.mp4", 5.5)
        >>> print(output)
        output/video_cut.mp4
    """
    import subprocess
    from sceneflow.shared.constants import FFMPEG
    from sceneflow.shared.exceptions import FFmpegNotFoundError, FFmpegExecutionError

    if output_path:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_base_name = Path(video_path).stem
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{video_base_name}_cut.mp4"
        final_output_path = output_dir / output_filename

    logger.info("Cutting video from 0.0000s to %.4fs using FFmpeg", cut_timestamp)

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-t', f'{cut_timestamp:.6f}',
        '-c:v', FFMPEG.VIDEO_CODEC,
        '-c:a', FFMPEG.AUDIO_CODEC,
        '-y',
        str(final_output_path)
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG.TIMEOUT_SECONDS
        )
        logger.info("Saved cut video (0.0000s - %.4fs) to: %s", cut_timestamp, final_output_path)
        return str(final_output_path)

    except FileNotFoundError:
        raise FFmpegNotFoundError()
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout after %d seconds", FFMPEG.TIMEOUT_SECONDS)
        raise FFmpegExecutionError(
            ' '.join(cmd),
            f"Timeout after {FFMPEG.TIMEOUT_SECONDS} seconds"
        )
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg failed: %s", e.stderr)
        raise FFmpegExecutionError(' '.join(cmd), e.stderr)


def cut_video_to_bytes(video_path: str, cut_timestamp: float) -> bytes:
    """
    Cut video from start to timestamp and return as bytes.

    Useful for uploading to external services like Airtable.

    Args:
        video_path: Path to input video file
        cut_timestamp: Timestamp where to cut the video (in seconds)

    Returns:
        Cut video as bytes

    Raises:
        FFmpegNotFoundError: If ffmpeg is not installed
        FFmpegExecutionError: If ffmpeg command fails

    Example:
        >>> video_bytes = cut_video_to_bytes("video.mp4", 5.5)
        >>> len(video_bytes)
        123456
    """
    import subprocess
    from sceneflow.shared.constants import FFMPEG
    from sceneflow.shared.exceptions import FFmpegNotFoundError, FFmpegExecutionError

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', f'{cut_timestamp:.6f}',
            '-c:v', FFMPEG.VIDEO_CODEC,
            '-c:a', FFMPEG.AUDIO_CODEC,
            '-y',
            temp_path
        ]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG.TIMEOUT_SECONDS
        )

        with open(temp_path, 'rb') as f:
            return f.read()

    except FileNotFoundError:
        raise FFmpegNotFoundError()
    except subprocess.TimeoutExpired:
        raise FFmpegExecutionError(
            ' '.join(cmd),
            f"Timeout after {FFMPEG.TIMEOUT_SECONDS} seconds"
        )
    except subprocess.CalledProcessError as e:
        raise FFmpegExecutionError(' '.join(cmd), e.stderr)
    finally:
        try:
            Path(temp_path).unlink()
        except Exception:
            pass
