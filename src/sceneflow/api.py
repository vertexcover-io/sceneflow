"""High-level API functions for SceneFlow."""

import tempfile
import warnings
import os
import logging
from pathlib import Path
from typing import Optional, List
import cv2
import requests

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logger = logging.getLogger(__name__)

from .speech_detector import SpeechDetector
from .ranker import CutPointRanker
from .config import RankingConfig


def _is_url(source: str) -> bool:
    """Check if the source is a URL."""
    return source.startswith(('http://', 'https://', 'www.'))


def _download_video(url: str) -> str:
    """
    Download video from URL to temporary directory using HTTP GET.

    Args:
        url: Direct video URL (e.g., .mp4, .avi file URLs)

    Returns:
        Path to downloaded video file

    Raises:
        RuntimeError: If download fails
    """
    logger.info(f"Downloading video from URL: {url}")

    # Create temp directory for downloaded video
    temp_dir = Path(tempfile.mkdtemp(prefix="sceneflow_"))

    # Extract filename from URL or use default
    url_path = Path(url.split('?')[0])  # Remove query params
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Write to file
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")

        logger.info(f"Video downloaded successfully to: {output_path}")
        return str(output_path)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise RuntimeError(f"Error downloading video: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        raise RuntimeError(f"Error downloading video: {str(e)}")


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def _cleanup_downloaded_video(video_path: str) -> None:
    """
    Delete downloaded video and its temporary directory.

    Args:
        video_path: Path to the downloaded video file
    """
    try:
        video_file = Path(video_path)
        temp_dir = video_file.parent

        # Delete the video file
        if video_file.exists():
            video_file.unlink()
            logger.debug(f"Deleted downloaded video: {video_path}")

        # Delete the temporary directory if it's empty or contains only the video
        if temp_dir.exists() and temp_dir.name.startswith("sceneflow_"):
            # Check if directory is empty or only has the video we just deleted
            remaining_files = list(temp_dir.iterdir())
            if not remaining_files:
                temp_dir.rmdir()
                logger.debug(f"Deleted temporary directory: {temp_dir}")
            else:
                logger.warning(f"Temporary directory not empty, skipping deletion: {temp_dir}")

    except Exception as e:
        logger.warning(f"Failed to clean up downloaded video: {str(e)}")


def get_cut_frame(
    source: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 2,
    save_video: bool = False,
    save_frames: bool = False
) -> float:
    """
    Get the single best cut point timestamp for a video.

    This is a high-level convenience function that handles the complete pipeline:
    1. Downloads video if source is a URL
    2. Detects when speech ends using Silero VAD
    3. Ranks all frames after speech ends
    4. Returns the timestamp of the best cut point

    Args:
        source: Video file path or direct video URL (e.g., .mp4 URL)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2 for speed)
        save_video: If True, saves cut video to output directory
        save_frames: If True, saves annotated frames with landmarks

    Returns:
        Timestamp in seconds of the best cut point

    Raises:
        RuntimeError: If video processing fails
    """
    logger.info("=" * 60)
    logger.info("SceneFlow: Finding optimal cut point")
    logger.info("=" * 60)

    # Handle URL downloads
    video_path = source
    is_downloaded = _is_url(source)
    if is_downloaded:
        logger.info("Source is URL, downloading video...")
        video_path = _download_video(source)
    else:
        logger.info(f"Analyzing local video: {source}")

    try:
        # Stage 1: Detect when speech ends
        logger.info("Stage 1/2: Detecting speech end time...")
        detector = SpeechDetector()
        speech_end_time = detector.get_speech_end_time(video_path)
        logger.info(f"Speech ends at: {speech_end_time:.2f}s")

        # Get video duration
        duration = _get_video_duration(video_path)
        logger.info(f"Analyzing frames from {speech_end_time:.2f}s to {duration:.2f}s")

        # Stage 2: Rank frames after speech ends
        logger.info("Stage 2/2: Ranking frames based on visual quality...")
        ranker = CutPointRanker(config)
        ranked_frames = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_video=save_video,
            save_frames=save_frames
        )

        if not ranked_frames:
            logger.error("No valid frames found for ranking")
            raise RuntimeError("No valid frames found for ranking")

        # Return the timestamp of the best frame
        best_timestamp = ranked_frames[0].timestamp
        logger.info("=" * 60)
        logger.info(f"Best cut point found: {best_timestamp:.2f}s (score: {ranked_frames[0].score:.4f})")
        logger.info("=" * 60)

        return best_timestamp

    finally:
        # Clean up downloaded video
        if is_downloaded:
            _cleanup_downloaded_video(video_path)


def get_ranked_cut_frames(
    source: str,
    n: int = 5,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 2,
    save_frames: bool = False
) -> List[float]:
    """
    Get the top N best cut point timestamps for a video.

    This is a high-level convenience function that handles the complete pipeline:
    1. Downloads video if source is a URL
    2. Detects when speech ends using Silero VAD
    3. Ranks all frames after speech ends
    4. Returns the timestamps of the top N cut points

    Args:
        source: Video file path or direct video URL (e.g., .mp4 URL)
        n: Number of top cut points to return (default: 5)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2 for speed)
        save_frames: If True, saves annotated frames with landmarks

    Returns:
        List of timestamps in seconds for the top N cut points, ordered by rank

    Raises:
        RuntimeError: If video processing fails
        ValueError: If n is less than 1
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    logger.info("=" * 60)
    logger.info(f"SceneFlow: Finding top {n} cut points")
    logger.info("=" * 60)

    # Handle URL downloads
    video_path = source
    is_downloaded = _is_url(source)
    if is_downloaded:
        logger.info("Source is URL, downloading video...")
        video_path = _download_video(source)
    else:
        logger.info(f"Analyzing local video: {source}")

    try:
        # Stage 1: Detect when speech ends
        logger.info("Stage 1/2: Detecting speech end time...")
        detector = SpeechDetector()
        speech_end_time = detector.get_speech_end_time(video_path)
        logger.info(f"Speech ends at: {speech_end_time:.2f}s")

        # Get video duration
        duration = _get_video_duration(video_path)
        logger.info(f"Analyzing frames from {speech_end_time:.2f}s to {duration:.2f}s")

        # Stage 2: Rank frames after speech ends
        logger.info("Stage 2/2: Ranking frames based on visual quality...")
        ranker = CutPointRanker(config)
        ranked_frames = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_video=False,
            save_frames=save_frames
        )

        if not ranked_frames:
            logger.error("No valid frames found for ranking")
            raise RuntimeError("No valid frames found for ranking")

        top_timestamps = [frame.timestamp for frame in ranked_frames[:n]]

        logger.info("=" * 60)
        logger.info(f"Top {len(top_timestamps)} cut points found:")
        for i, (frame, timestamp) in enumerate(zip(ranked_frames[:n], top_timestamps), 1):
            logger.info(f"  {i}. {timestamp:.2f}s (score: {frame.score:.4f})")
        logger.info("=" * 60)

        return top_timestamps

    finally:
        # Clean up downloaded video
        if is_downloaded:
            _cleanup_downloaded_video(video_path)
