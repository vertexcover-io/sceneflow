"""High-level Airtable integration API for SceneFlow.

This module provides convenience functions for analyzing videos and
automatically uploading results to Airtable. These are separate from
the core API to keep the main analysis functions clean and focused.
"""

import logging
from typing import Optional, Tuple, List

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.integration.airtable import upload_to_airtable
from sceneflow.api._internal import detect_speech_end, select_best_with_llm
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_duration,
    cut_video,
)
from sceneflow.utils.output import save_annotated_frames, save_analysis_logs

logger = logging.getLogger(__name__)


def analyze_and_upload_to_airtable(
    source: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[float, str]:
    """Analyze video and upload results to Airtable.

    This is a convenience function that combines video analysis with
    Airtable upload. It performs the complete SceneFlow pipeline and
    uploads all results to your Airtable base.

    Args:
        source: Video file path or direct video URL
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        airtable_access_token: Airtable access token (or use AIRTABLE_ACCESS_TOKEN env var)
        airtable_base_id: Airtable base ID (or use AIRTABLE_BASE_ID env var)
        airtable_table_name: Table name (or use AIRTABLE_TABLE_NAME env var, defaults to "SceneFlow Analysis")
        use_llm_selection: If True, uses LLM to select best frame from top 5
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time

    Returns:
        Tuple of (cut_timestamp, airtable_record_id)

    Example:
        >>> from sceneflow.integration import analyze_and_upload_to_airtable
        >>> timestamp, record_id = analyze_and_upload_to_airtable(
        ...     "video.mp4",
        ...     airtable_access_token="your-token",
        ...     airtable_base_id="appXXXXXXXXXXXXXX"
        ... )
        >>> print(f"Cut at {timestamp:.2f}s, uploaded to {record_id}")
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        # Stage 1: Detect speech end
        speech_end_time = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.warning(
                "Visual analysis disabled - cannot upload to Airtable without frame analysis"
            )
            raise ValueError("Cannot upload to Airtable when visual analysis is disabled")

        # Stage 2: Rank frames with internals for upload
        duration = get_video_duration(video_path)

        ranker = CutPointRanker(config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if not result.features or not result.scores:
            raise RuntimeError("Failed to extract frame features and scores")

        best_frame = result.ranked_frames[0]

        # Stage 3: Optional LLM selection
        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = select_best_with_llm(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                result.scores,
                result.features,
                openai_api_key,
            )

        # Stage 4: Upload to Airtable
        best_score = next(
            (s for s in result.scores if s.frame_index == best_frame.frame_index), None
        )
        best_features = next(
            (f for f in result.features if f.frame_index == best_frame.frame_index), None
        )

        if not best_score or not best_features:
            raise RuntimeError("Could not find score and features for best frame")

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        logger.info(
            "Uploaded to Airtable: timestamp=%.4fs, record_id=%s",
            best_frame.timestamp,
            record_id,
        )
        return best_frame.timestamp, record_id

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


def analyze_ranked_and_upload_to_airtable(
    source: str,
    n: int = 5,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[List[float], str]:
    """Analyze video for top N cut points and upload best result to Airtable.

    This function ranks multiple cut points but only uploads the best one
    to Airtable along with full analysis details.

    Args:
        source: Video file path or direct video URL
        n: Number of top cut points to return (default: 5)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        airtable_access_token: Airtable access token (or use AIRTABLE_ACCESS_TOKEN env var)
        airtable_base_id: Airtable base ID (or use AIRTABLE_BASE_ID env var)
        airtable_table_name: Table name (or use AIRTABLE_TABLE_NAME env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking

    Returns:
        Tuple of (list of top N timestamps, airtable_record_id for best result)

    Example:
        >>> from sceneflow.integration import analyze_ranked_and_upload_to_airtable
        >>> timestamps, record_id = analyze_ranked_and_upload_to_airtable(
        ...     "video.mp4",
        ...     n=5,
        ...     airtable_base_id="appXXXXXXXXXXXXXX"
        ... )
        >>> print(f"Top 5: {timestamps}, uploaded best to {record_id}")
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        # Stage 1: Detect speech end
        speech_end_time = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.warning(
                "Visual analysis disabled - cannot upload to Airtable without frame analysis"
            )
            raise ValueError("Cannot upload to Airtable when visual analysis is disabled")

        # Stage 2: Rank frames with internals for upload
        duration = get_video_duration(video_path)

        ranker = CutPointRanker(config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if not result.features or not result.scores:
            raise RuntimeError("Failed to extract frame features and scores")

        # Stage 3: Upload best result to Airtable
        best_frame = result.ranked_frames[0]
        best_score = next(
            (s for s in result.scores if s.frame_index == best_frame.frame_index), None
        )
        best_features = next(
            (f for f in result.features if f.frame_index == best_frame.frame_index), None
        )

        if not best_score or not best_features:
            raise RuntimeError("Could not find score and features for best frame")

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        # Return top N timestamps
        top_timestamps = [frame.timestamp for frame in result.ranked_frames[:n]]
        logger.info(
            "Uploaded to Airtable: top %d cut points, best at %.4fs, record_id=%s",
            len(top_timestamps),
            top_timestamps[0],
            record_id,
        )
        return top_timestamps, record_id

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


def cut_and_upload_to_airtable(
    source: str,
    output_path: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    save_frames: bool = False,
    save_logs: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[float, str]:
    """Cut video at optimal point and upload results to Airtable.

    This function combines video cutting with Airtable upload, providing
    both a saved cut video file and a complete Airtable record.

    Args:
        source: Video file path or direct video URL
        output_path: Output path for the cut video (required)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        save_frames: If True, saves annotated frames with landmarks
        save_logs: If True, saves detailed analysis logs
        airtable_access_token: Airtable access token (or use AIRTABLE_ACCESS_TOKEN env var)
        airtable_base_id: Airtable base ID (or use AIRTABLE_BASE_ID env var)
        airtable_table_name: Table name (or use AIRTABLE_TABLE_NAME env var)
        use_llm_selection: If True, uses LLM to select best frame
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking

    Returns:
        Tuple of (cut_timestamp, airtable_record_id)

    Example:
        >>> from sceneflow.integration import cut_and_upload_to_airtable
        >>> timestamp, record_id = cut_and_upload_to_airtable(
        ...     "video.mp4",
        ...     output_path="output.mp4",
        ...     airtable_base_id="appXXXXXXXXXXXXXX"
        ... )
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        # Stage 1: Detect speech end
        speech_end_time = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.warning(
                "Visual analysis disabled - cannot upload to Airtable without frame analysis"
            )
            raise ValueError("Cannot upload to Airtable when visual analysis is disabled")

        # Stage 2: Rank frames with internals for upload
        duration = get_video_duration(video_path)

        ranker = CutPointRanker(config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if not result.features or not result.scores:
            raise RuntimeError("Failed to extract frame features and scores")

        best_frame = result.ranked_frames[0]

        # Stage 3: Optional LLM selection
        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = select_best_with_llm(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                result.scores,
                result.features,
                openai_api_key,
            )

        # Handle side effects after best frame is determined
        if save_frames:
            save_annotated_frames(video_path, result.ranked_frames, ranker.extractor)

        if save_logs:
            save_analysis_logs(video_path, result.ranked_frames, result.features, result.scores)

        # Cut the video
        cut_video(video_path, best_frame.timestamp, output_path)

        # Stage 4: Upload to Airtable
        best_score = next(
            (s for s in result.scores if s.frame_index == best_frame.frame_index), None
        )
        best_features = next(
            (f for f in result.features if f.frame_index == best_frame.frame_index), None
        )

        if not best_score or not best_features:
            raise RuntimeError("Could not find score and features for best frame")

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        logger.info(
            "Cut video and uploaded to Airtable: timestamp=%.4fs, record_id=%s",
            best_frame.timestamp,
            record_id,
        )
        return best_frame.timestamp, record_id

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


def _build_config_dict(config: Optional[RankingConfig], sample_rate: int) -> dict:
    """Build configuration dictionary for Airtable metadata.

    Args:
        config: Optional RankingConfig instance
        sample_rate: Frame sampling rate used

    Returns:
        Dictionary containing configuration details
    """
    if config is None:
        config = RankingConfig()

    return {
        "sample_rate": sample_rate,
        "weights": {
            "eye_openness": config.eye_openness_weight,
            "motion_stability": config.motion_stability_weight,
            "expression_neutrality": config.expression_neutrality_weight,
            "pose_stability": config.pose_stability_weight,
            "visual_sharpness": config.visual_sharpness_weight,
        },
        "context_window_size": config.context_window_size,
        "quality_gate_percentile": config.quality_gate_percentile,
        "local_stability_window": config.local_stability_window,
    }
