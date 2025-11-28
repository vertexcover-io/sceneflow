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
from .llm_selector import LLMFrameSelector
from .energy_refiner import EnergyRefiner


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
    save_frames: bool = False,
    save_logs: bool = False,
    upload_to_airtable: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20
) -> float:
    """
    Get the single best cut point timestamp for a video.

    This is a high-level convenience function that handles the complete pipeline:
    1. Downloads video if source is a URL
    2. Detects when speech ends using Silero VAD
    3. Optionally refines VAD timestamp using audio energy analysis
    4. Ranks all frames after speech ends
    5. Optionally uses LLM vision model to select best from top 5 candidates
    6. Returns the timestamp of the best cut point
    7. Optionally uploads results to Airtable

    Args:
        source: Video file path or direct video URL (e.g., .mp4 URL)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2 for speed)
        save_video: If True, saves cut video to output directory
        save_frames: If True, saves annotated frames with landmarks
        save_logs: If True, saves detailed analysis logs
        upload_to_airtable: If True, uploads results to Airtable
        airtable_access_token: Airtable access token (defaults to AIRTABLE_ACCESS_TOKEN env var)
        airtable_base_id: Airtable base ID (defaults to AIRTABLE_BASE_ID env var)
        airtable_table_name: Table name (defaults to AIRTABLE_TABLE_NAME or "SceneFlow Analysis")
        use_llm_selection: If True, uses GPT-4o to select best from top 5 frames (default: True)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD timestamp using energy drops (default: True)
        energy_threshold_db: Minimum dB drop to consider speech end (default: 8.0)
        energy_lookback_frames: Maximum frames to search backward from VAD (default: 20)

    Returns:
        Timestamp in seconds of the best cut point

    Raises:
        RuntimeError: If video processing or Airtable upload fails
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
        logger.info("Stage 1: Detecting speech end time...")
        detector = SpeechDetector()
        vad_speech_end_time = detector.get_speech_end_time(video_path)
        logger.info(f"VAD detected speech end at: {vad_speech_end_time:.2f}s")

        # Stage 1.5: Refine with energy analysis
        speech_end_time = vad_speech_end_time
        if use_energy_refinement:
            logger.info("Stage 1.5: Refining speech end time with energy analysis...")
            refiner = EnergyRefiner(
                threshold_db=energy_threshold_db,
                lookback_frames=energy_lookback_frames
            )
            speech_end_time, metadata = refiner.refine_speech_end(vad_speech_end_time, video_path)

            if metadata['frames_adjusted'] > 0:
                logger.info(f"Energy refinement adjusted timestamp by {metadata['frames_adjusted']} frames")
            else:
                logger.info("Energy refinement: No adjustment needed")

        # Get video duration
        duration = _get_video_duration(video_path)
        logger.info(f"Analyzing frames from {speech_end_time:.4f}s to {duration:.4f}s")

        # Stage 2: Rank frames after speech ends
        logger.info("Stage 2: Ranking frames based on visual quality...")
        ranker = CutPointRanker(config)

        if upload_to_airtable or use_llm_selection:
            # Disable save_video in ranker if using LLM selection
            # (we'll save after LLM picks the best frame)
            ranked_frames, features, scores = ranker.rank_frames(
                video_path=video_path,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_video=False,  # Disable here, save later with correct timestamp
                save_frames=save_frames,
                save_logs=save_logs,
                return_internals=True
            )
        else:
            ranked_frames = ranker.rank_frames(
                video_path=video_path,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_video=save_video,
                save_frames=save_frames,
                save_logs=save_logs
            )

        if not ranked_frames:
            logger.error("No valid frames found for ranking")
            raise RuntimeError("No valid frames found for ranking")

        best_frame = ranked_frames[0]

        if use_llm_selection and len(ranked_frames) > 1:
            try:
                logger.info("Stage 3: Using LLM to select best frame from top 5 candidates...")
                selector = LLMFrameSelector(api_key=openai_api_key)
                best_frame = selector.select_best_frame(
                    video_path=video_path,
                    ranked_frames=ranked_frames[:5],
                    speech_end_time=speech_end_time,
                    video_duration=duration,
                    all_scores=scores,
                    all_features=features
                )
                logger.info(f"LLM selected frame at {best_frame.timestamp:.2f}s : frame {best_frame.frame_index}")
            except Exception as e:
                logger.warning(f"LLM selection failed: {str(e)}, using top algorithmic result")
                best_frame = ranked_frames[0]

        best_timestamp = best_frame.timestamp
        logger.info("=" * 60)
        logger.info(f"Best cut point found: {best_timestamp:.4f}s (score: {best_frame.score:.4f}) : frame {best_frame.frame_index}")
        logger.info("=" * 60)

        # Save video with the correct timestamp (after LLM selection if enabled)
        if save_video:
            ranker._save_cut_video(video_path, best_timestamp)

        # Upload to Airtable if requested
        if upload_to_airtable:
            try:
                from .airtable_uploader import upload_to_airtable as airtable_upload

                logger.info("Uploading results to Airtable...")

                best_score = next((s for s in scores if s.frame_index == best_frame.frame_index), None)
                best_features = next((f for f in features if f.frame_index == best_frame.frame_index), None)

                if best_score and best_features:
                    config_dict = {
                        "sample_rate": sample_rate,
                        "weights": {
                            "eye_openness": config.eye_openness_weight if config else 0.30,
                            "motion_stability": config.motion_stability_weight if config else 0.25,
                            "expression_neutrality": config.expression_neutrality_weight if config else 0.20,
                            "pose_stability": config.pose_stability_weight if config else 0.15,
                            "visual_sharpness": config.visual_sharpness_weight if config else 0.10
                        }
                    }

                    record_id = airtable_upload(
                        video_path=video_path,
                        best_frame=best_frame,
                        frame_score=best_score,
                        frame_features=best_features,
                        speech_end_time=speech_end_time,
                        duration=duration,
                        config_dict=config_dict,
                        access_token=airtable_access_token,
                        base_id=airtable_base_id,
                        table_name=airtable_table_name
                    )

                    logger.info(f"Successfully uploaded to Airtable! Record ID: {record_id}")
                else:
                    logger.warning("Could not upload to Airtable - missing score or features data")

            except Exception as e:
                logger.error(f"Failed to upload to Airtable: {str(e)}")
                # Don't fail the entire function, just log the error
                raise RuntimeError(f"Airtable upload failed: {str(e)}")

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
    save_frames: bool = False,
    upload_to_airtable: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20
) -> List[float]:
    """
    Get the top N best cut point timestamps for a video.

    This is a high-level convenience function that handles the complete pipeline:
    1. Downloads video if source is a URL
    2. Detects when speech ends using Silero VAD
    3. Optionally refines VAD timestamp using audio energy analysis
    4. Ranks all frames after speech ends
    5. Returns the timestamps of the top N cut points
    6. Optionally uploads best result to Airtable

    Args:
        source: Video file path or direct video URL (e.g., .mp4 URL)
        n: Number of top cut points to return (default: 5)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2 for speed)
        save_frames: If True, saves annotated frames with landmarks
        upload_to_airtable: If True, uploads best result to Airtable
        airtable_access_token: Airtable access token (defaults to AIRTABLE_ACCESS_TOKEN env var)
        airtable_base_id: Airtable base ID (defaults to AIRTABLE_BASE_ID env var)
        airtable_table_name: Table name (defaults to AIRTABLE_TABLE_NAME or "SceneFlow Analysis")
        use_energy_refinement: If True, refines VAD timestamp using energy drops (default: True)
        energy_threshold_db: Minimum dB drop to consider speech end (default: 8.0)
        energy_lookback_frames: Maximum frames to search backward from VAD (default: 20)

    Returns:
        List of timestamps in seconds for the top N cut points, ordered by rank

    Raises:
        RuntimeError: If video processing or Airtable upload fails
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
        logger.info("Stage 1: Detecting speech end time...")
        detector = SpeechDetector()
        vad_speech_end_time = detector.get_speech_end_time(video_path)
        logger.info(f"VAD detected speech end at: {vad_speech_end_time:.2f}s")

        # Stage 1.5: Refine with energy analysis
        speech_end_time = vad_speech_end_time
        if use_energy_refinement:
            logger.info("Stage 1.5: Refining speech end time with energy analysis...")
            refiner = EnergyRefiner(
                threshold_db=energy_threshold_db,
                lookback_frames=energy_lookback_frames
            )
            speech_end_time, metadata = refiner.refine_speech_end(vad_speech_end_time, video_path)

            if metadata['frames_adjusted'] > 0:
                logger.info(f"Energy refinement adjusted timestamp by {metadata['frames_adjusted']} frames")
            else:
                logger.info("Energy refinement: No adjustment needed")

        # Get video duration
        duration = _get_video_duration(video_path)
        logger.info(f"Analyzing frames from {speech_end_time:.4f}s to {duration:.4f}s")

        # Stage 2: Rank frames after speech ends
        logger.info("Stage 2: Ranking frames based on visual quality...")
        ranker = CutPointRanker(config)

        # Get internals if uploading to Airtable to avoid re-processing
        if upload_to_airtable:
            ranked_frames, features, scores = ranker.rank_frames(
                video_path=video_path,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_video=False,
                save_frames=save_frames,
                return_internals=True
            )
        else:
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

        # Upload to Airtable if requested (only uploads the best frame)
        if upload_to_airtable:
            try:
                from .airtable_uploader import upload_to_airtable as airtable_upload

                logger.info("Uploading best result to Airtable...")

                # Find score and features for best frame (already computed, no re-processing!)
                best_score = next((s for s in scores if s.frame_index == ranked_frames[0].frame_index), None)
                best_features = next((f for f in features if f.frame_index == ranked_frames[0].frame_index), None)

                if best_score and best_features:
                    # Prepare config dict
                    config_dict = {
                        "sample_rate": sample_rate,
                        "weights": {
                            "eye_openness": config.eye_openness_weight if config else 0.30,
                            "motion_stability": config.motion_stability_weight if config else 0.25,
                            "expression_neutrality": config.expression_neutrality_weight if config else 0.20,
                            "pose_stability": config.pose_stability_weight if config else 0.15,
                            "visual_sharpness": config.visual_sharpness_weight if config else 0.10
                        }
                    }

                    record_id = airtable_upload(
                        video_path=video_path,
                        best_frame=ranked_frames[0],
                        frame_score=best_score,
                        frame_features=best_features,
                        speech_end_time=speech_end_time,
                        duration=duration,
                        config_dict=config_dict,
                        access_token=airtable_access_token,
                        base_id=airtable_base_id,
                        table_name=airtable_table_name
                    )

                    logger.info(f"Successfully uploaded to Airtable! Record ID: {record_id}")
                else:
                    logger.warning("Could not upload to Airtable - missing score or features data")

            except Exception as e:
                logger.error(f"Failed to upload to Airtable: {str(e)}")
                raise RuntimeError(f"Airtable upload failed: {str(e)}")

        return top_timestamps

    finally:
        # Clean up downloaded video
        if is_downloaded:
            _cleanup_downloaded_video(video_path)
