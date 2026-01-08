"""Video utilities for downloading, processing, and metadata extraction.

This module provides shared utilities for video operations to eliminate
code duplication and provide consistent error handling.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse
from contextlib import contextmanager, asynccontextmanager
import subprocess
from sceneflow.shared.constants import FFMPEG
from sceneflow.shared.exceptions import FFmpegNotFoundError, FFmpegExecutionError
import librosa

import aiofiles
import cv2
import httpx
import requests

from sceneflow.shared.constants import VIDEO
from sceneflow.shared.exceptions import (
    VideoOpenError,
    VideoPropertiesError,
    VideoDownloadError,
    InvalidURLError,
)
from sceneflow.shared.models import VideoProperties
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Tuple
import functools
import numpy as np

logger = logging.getLogger(__name__)

_temp_files_to_cleanup: List[str] = []


def _register_temp_file(path: str) -> None:
    """Register a temp file for cleanup on process exit."""
    _temp_files_to_cleanup.append(path)


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
    return source.startswith(("http://", "https://"))


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
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(url, f"Invalid scheme: {parsed.scheme}")

    # Check domain
    if not parsed.netloc:
        raise InvalidURLError(url, "Missing domain")

    # Check for suspicious characters
    suspicious_chars = ["<", ">", '"', "'", "\n", "\r", "\x00"]
    if any(char in url for char in suspicious_chars):
        raise InvalidURLError(url, "Contains suspicious characters")

    # Warn if doesn't look like a video file
    path = parsed.path.lower()
    if not any(path.endswith(ext) for ext in VIDEO.SUPPORTED_EXTENSIONS):
        logger.warning("URL does not appear to be a video file: %s", url)

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

    # Create temp directory for downloaded video
    temp_dir = Path(tempfile.mkdtemp(prefix="sceneflow_"))

    # Extract filename from URL or use default
    url_path = Path(url.split("?")[0])  # Remove query params
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=VIDEO.DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Write to file
        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=VIDEO.DOWNLOAD_CHUNK_SIZE_BYTES):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug("Download progress: %.1f%%", progress)

        logger.info("Downloaded video to: %s", output_path)
        return str(output_path)

    except requests.exceptions.Timeout:
        raise VideoDownloadError(
            url, f"Download timed out after {VIDEO.DOWNLOAD_TIMEOUT_SECONDS} seconds"
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
                logger.warning("Temporary directory not empty, skipping deletion: %s", temp_dir)

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
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> cv2.VideoCapture:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.cap.release()
            raise VideoOpenError(self.video_path, "VideoCapture.isOpened() returned False")
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return False


@functools.lru_cache(maxsize=32)
def _extract_frame(video_path: str, frame_index: int) -> np.ndarray:
    with VideoCapture(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise IndexError(f"Frame index out of range: {frame_index}")
        return frame


@dataclass
class VideoSession:
    """
    Safe video session with on-demand frame extraction.

    Frames are extracted lazily via LRU cache instead of loading all into memory.
    """

    video_path: str

    _properties: Optional[VideoProperties] = field(default=None, init=False, repr=False)
    _audio_cache: Optional[Tuple[np.ndarray, int]] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "VideoSession":
        self._load_properties()
        return self

    def __exit__(self, *_):
        _extract_frame.cache_clear()
        self._audio_cache = None
        self._properties = None
        return False

    @property
    def properties(self) -> VideoProperties:
        if self._properties is None:
            raise RuntimeError("VideoSession not initialized")
        return self._properties

    def _load_audio(self, sr: Optional[int]) -> Tuple[np.ndarray, int]:
        return librosa.load(self.video_path, sr=sr)

    def get_audio(self, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if self._audio_cache is None:
            self._audio_cache = self._load_audio(sr)
        return self._audio_cache

    def get_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.properties.frame_count:
            raise IndexError(f"Frame index out of range: {frame_index}")
        return _extract_frame(self.video_path, frame_index).copy()

    def get_frame_as_jpeg(
        self, frame_index: int, jpeg_quality: int = VIDEO.JPEG_QUALITY_DEFAULT
    ) -> bytes:
        frame = self.get_frame(frame_index)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        return buffer.tobytes()

    def iterate_frames(
        self,
        start_frame: int,
        end_frame: int,
        sample_rate: int = 1,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        end_frame = min(end_frame, self.properties.frame_count - 1)

        with VideoCapture(self.video_path) as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                if (idx - start_frame) % sample_rate == 0:
                    yield idx, frame.copy()

    # -------------------------
    # Async Lifecycle
    # -------------------------

    async def __aenter__(self) -> "VideoSession":
        await asyncio.to_thread(self._load_properties)
        return self

    async def __aexit__(self, *_):
        _extract_frame.cache_clear()
        self._audio_cache = None
        self._properties = None
        return False

    def _load_properties(self) -> None:
        with VideoCapture(self.video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0 or frame_count <= 0:
                raise VideoPropertiesError(
                    self.video_path,
                    f"Invalid fps={fps} or frame_count={frame_count}",
                )

            if width <= 0 or height <= 0:
                raise VideoPropertiesError(self.video_path, f"Invalid dimensions: {width}x{height}")

            self._properties = VideoProperties(
                fps=fps,
                frame_count=frame_count,
                duration=frame_count / fps,
                width=width,
                height=height,
            )

    async def iterate_frames_async(
        self,
        start_frame: int,
        end_frame: int,
        sample_rate: int = 1,
    ) -> AsyncIterator[Tuple[int, np.ndarray]]:
        end_frame = min(end_frame, self.properties.frame_count - 1)

        def read_frames():
            frames = []
            with VideoCapture(self.video_path) as cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for idx in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if (idx - start_frame) % sample_rate == 0:
                        frames.append((idx, frame.copy()))
            return frames

        frames = await asyncio.to_thread(read_frames)
        for idx, frame in frames:
            yield idx, frame


def _resolve_output_path(video_path: str, output_path: Optional[str]) -> Path:
    if output_path:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_base_name = Path(video_path).stem
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{video_base_name}_cut.mp4"
        final_output_path = output_dir / output_filename
    return final_output_path


def cut_video(video_path: str, cut_timestamp: float, output_path: Optional[str] = None) -> str:
    final_output_path = _resolve_output_path(video_path, output_path)
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-t",
        f"{cut_timestamp:.6f}",
        "-c:v",
        FFMPEG.VIDEO_CODEC,
        "-c:a",
        FFMPEG.AUDIO_CODEC,
        "-y",
        str(final_output_path),
    ]

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=FFMPEG.TIMEOUT_SECONDS
        )
        logger.info("Cut video saved (0.0000s - %.4fs) to: %s", cut_timestamp, final_output_path)
        return str(final_output_path)

    except FileNotFoundError:
        raise FFmpegNotFoundError()
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout after %d seconds", FFMPEG.TIMEOUT_SECONDS)
        raise FFmpegExecutionError(" ".join(cmd), f"Timeout after {FFMPEG.TIMEOUT_SECONDS} seconds")
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg failed: %s", e.stderr)
        raise FFmpegExecutionError(" ".join(cmd), e.stderr)


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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    _register_temp_file(temp_path)

    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-t",
            f"{cut_timestamp:.6f}",
            "-c:v",
            FFMPEG.VIDEO_CODEC,
            "-c:a",
            FFMPEG.AUDIO_CODEC,
            "-y",
            temp_path,
        ]

        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=FFMPEG.TIMEOUT_SECONDS
        )

        with open(temp_path, "rb") as f:
            return f.read()

    except FileNotFoundError:
        raise FFmpegNotFoundError()
    except subprocess.TimeoutExpired:
        raise FFmpegExecutionError(" ".join(cmd), f"Timeout after {FFMPEG.TIMEOUT_SECONDS} seconds")
    except subprocess.CalledProcessError as e:
        raise FFmpegExecutionError(" ".join(cmd), e.stderr)
    finally:
        try:
            Path(temp_path).unlink()
            _temp_files_to_cleanup.remove(temp_path)
        except Exception:
            pass


async def download_video_async(url: str) -> str:
    """
    Async version of download_video using httpx.

    Downloads video from URL to temporary directory using async HTTP GET.

    Args:
        url: Direct video URL (e.g., .mp4, .avi file URLs)

    Returns:
        Path to downloaded video file

    Raises:
        VideoDownloadError: If download fails
        InvalidURLError: If URL is invalid
    """
    url = validate_video_url(url)

    temp_dir = Path(tempfile.mkdtemp(prefix="sceneflow_"))
    url_path = Path(url.split("?")[0])
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        async with httpx.AsyncClient(timeout=VIDEO.DOWNLOAD_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                async with aiofiles.open(output_path, "wb") as f:
                    downloaded = 0
                    async for chunk in response.aiter_bytes(
                        chunk_size=VIDEO.DOWNLOAD_CHUNK_SIZE_BYTES
                    ):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug("Download progress: %.1f%%", progress)

        logger.info("Downloaded video to: %s", output_path)
        return str(output_path)

    except httpx.TimeoutException:
        raise VideoDownloadError(
            url, f"Download timed out after {VIDEO.DOWNLOAD_TIMEOUT_SECONDS} seconds"
        )
    except httpx.ConnectError as e:
        raise VideoDownloadError(url, f"Connection failed: {e}")
    except httpx.HTTPStatusError as e:
        raise VideoDownloadError(url, f"HTTP error {e.response.status_code}")
    except OSError as e:
        raise VideoDownloadError(url, f"File system error: {e}")
    except Exception as e:
        logger.exception("Unexpected error during async download")
        raise VideoDownloadError(url, f"Unexpected error: {e}") from e


async def cleanup_downloaded_video_async(video_path: str) -> None:
    """
    Async version of cleanup_downloaded_video.

    Delete downloaded video and its temporary directory.

    Args:
        video_path: Path to the downloaded video file
    """
    try:
        video_file = Path(video_path)
        temp_dir = video_file.parent

        if video_file.exists():
            video_file.unlink()
            logger.debug("Deleted downloaded video: %s", video_path)

        if temp_dir.exists() and temp_dir.name.startswith("sceneflow_"):
            remaining_files = list(temp_dir.iterdir())
            if not remaining_files:
                temp_dir.rmdir()
                logger.debug("Deleted temporary directory: %s", temp_dir)
            else:
                logger.warning("Temporary directory not empty, skipping deletion: %s", temp_dir)

    except Exception as e:
        logger.warning("Failed to clean up downloaded video: %s", e)


@asynccontextmanager
async def temporary_video_download_async(url: str):
    """
    Async context manager for downloading and auto-cleaning up video.

    Args:
        url: Video URL to download

    Yields:
        Path to downloaded video file

    Raises:
        VideoDownloadError: If download fails
    """
    video_path = None
    try:
        video_path = await download_video_async(url)
        yield video_path
    finally:
        if video_path:
            await cleanup_downloaded_video_async(video_path)


async def cut_video_async(
    video_path: str, cut_timestamp: float, output_path: Optional[str] = None
) -> str:
    """
    Async version of cut_video using asyncio subprocess.

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
    """

    if output_path:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_base_name = Path(video_path).stem
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{video_base_name}_cut.mp4"
        final_output_path = output_dir / output_filename

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-t",
        f"{cut_timestamp:.6f}",
        "-c:v",
        FFMPEG.VIDEO_CODEC,
        "-c:a",
        FFMPEG.AUDIO_CODEC,
        "-y",
        str(final_output_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=FFMPEG.TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise FFmpegExecutionError(
                " ".join(cmd), f"Timeout after {FFMPEG.TIMEOUT_SECONDS} seconds"
            )

        if process.returncode != 0:
            stderr_text = stderr.decode() if stderr else ""
            logger.error("FFmpeg failed: %s", stderr_text)
            raise FFmpegExecutionError(" ".join(cmd), stderr_text)

        logger.info("Cut video saved (0.0000s - %.4fs) to: %s", cut_timestamp, final_output_path)
        return str(final_output_path)

    except FileNotFoundError:
        raise FFmpegNotFoundError()
