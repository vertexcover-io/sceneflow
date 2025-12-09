"""High-level API functions for SceneFlow.

This module provides the main public API functions for finding optimal
cut points in videos using multi-stage analysis.
"""

import logging
from typing import List, Optional

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_duration,
)
from sceneflow.api._internal import (
    detect_speech_end,
    rank_frames,
    select_best_with_llm,
    upload_to_airtable as upload_results_to_airtable,
)

logger = logging.getLogger(__name__)


def get_cut_frame(
    source: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    upload_to_airtable: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> float:
    """Get the single best cut point timestamp for a video.

    Args:
        source: Video file path or direct video URL
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2)
        upload_to_airtable: If True, uploads results to Airtable
        airtable_access_token: Airtable access token
        airtable_base_id: Airtable base ID
        airtable_table_name: Table name
        use_llm_selection: If True, uses LLM to select best frame
        openai_api_key: OpenAI API key
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time

    Returns:
        Timestamp in seconds of the best cut point
    """
    logger.info("SceneFlow: Finding optimal cut point")

    video_path = source
    is_downloaded = is_url(source)

    if is_downloaded:
        logger.info("Source is URL, downloading video...")
        video_path = download_video(source)
    else:
        logger.info("Analyzing local video: %s", source)

    try:
        speech_end_time, visual_search_end_time, _ = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info("Visual analysis disabled - returning speech end time: %.4fs", speech_end_time)
            return speech_end_time

        duration = get_video_duration(video_path)
        need_internals = use_llm_selection or upload_to_airtable

        ranked_frames, features, scores = rank_frames(
            video_path=video_path,
            speech_end_time=speech_end_time,
            duration=duration,
            config=config,
            sample_rate=sample_rate,
            visual_search_end_time=visual_search_end_time,
            return_internals=need_internals
        )

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = ranked_frames[0]

        if use_llm_selection and features and scores:
            best_frame = select_best_with_llm(
                video_path, ranked_frames, speech_end_time,
                duration, scores, features, openai_api_key
            )

        if upload_to_airtable and features and scores:
            upload_results_to_airtable(
                video_path, best_frame, scores, features,
                speech_end_time, duration, config, sample_rate,
                airtable_access_token, airtable_base_id, airtable_table_name
            )

        logger.info("Best cut point: %.4fs (frame: %d, score: %.3f)", best_frame.timestamp, best_frame.frame_index, best_frame.score)
        return best_frame.timestamp

    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)


def get_ranked_cut_frames(
    source: str,
    n: int = 5,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    upload_to_airtable: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> List[float]:
    """Get the top N best cut point timestamps for a video.

    Args:
        source: Video file path or direct video URL
        n: Number of top cut points to return (default: 5)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2)
        upload_to_airtable: If True, uploads best result to Airtable
        airtable_access_token: Airtable access token
        airtable_base_id: Airtable base ID
        airtable_table_name: Table name
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time only

    Returns:
        List of timestamps in seconds for the top N cut points
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    logger.info("SceneFlow: Finding top %d cut points", n)

    video_path = source
    is_downloaded = is_url(source)

    if is_downloaded:
        logger.info("Source is URL, downloading video...")
        video_path = download_video(source)
    else:
        logger.info("Analyzing local video: %s", source)

    try:
        speech_end_time, visual_search_end_time, _ = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info("Visual analysis disabled - returning speech end time: %.4fs", speech_end_time)
            return [speech_end_time]

        duration = get_video_duration(video_path)

        ranked_frames, features, scores = rank_frames(
            video_path=video_path,
            speech_end_time=speech_end_time,
            duration=duration,
            config=config,
            sample_rate=sample_rate,
            visual_search_end_time=visual_search_end_time,
            return_internals=upload_to_airtable
        )

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if upload_to_airtable and features and scores:
            upload_results_to_airtable(
                video_path, ranked_frames[0], scores, features,
                speech_end_time, duration, config, sample_rate,
                airtable_access_token, airtable_base_id, airtable_table_name
            )

        top_timestamps = [frame.timestamp for frame in ranked_frames[:n]]
        logger.info("Top %d cut points found", len(top_timestamps))
        return top_timestamps

    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)


def cut_video(
    source: str,
    output_path: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
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
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> float:
    """Cut a video at the optimal point and save the cut video.

    This function finds the best cut point and always saves the cut video.
    Optionally saves annotated frames and detailed logs.
    Use get_cut_frame() if you only need the timestamp without saving.

    Args:
        source: Video file path or direct video URL
        output_path: Output path for the cut video (required)
        config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 2)
        save_frames: If True, saves annotated frames with landmarks
        save_logs: If True, saves detailed analysis logs
        upload_to_airtable: If True, uploads results to Airtable
        airtable_access_token: Airtable access token
        airtable_base_id: Airtable base ID
        airtable_table_name: Table name
        use_llm_selection: If True, uses LLM to select best frame
        openai_api_key: OpenAI API key
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and cuts at speech end time

    Returns:
        Timestamp in seconds of the best cut point
    """
    logger.info("SceneFlow: Cutting video at optimal point")

    video_path = source
    is_downloaded = is_url(source)

    if is_downloaded:
        logger.info("Source is URL, downloading video...")
        video_path = download_video(source)
    else:
        logger.info("Analyzing local video: %s", source)

    try:
        speech_end_time, visual_search_end_time, vad_timestamps = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info("Visual analysis disabled - cutting at speech end time: %.4fs", speech_end_time)
            ranker = CutPointRanker(config)
            ranker._save_cut_video(video_path, speech_end_time, output_path=output_path)
            return speech_end_time

        duration = get_video_duration(video_path)
        need_internals = use_llm_selection or upload_to_airtable

        ranker = CutPointRanker(config)

        end_time = visual_search_end_time if visual_search_end_time > 0 else duration

        rank_kwargs = {
            "video_path": video_path,
            "start_time": speech_end_time,
            "end_time": end_time,
            "sample_rate": sample_rate,
            "save_frames": save_frames,
            "save_video": not use_llm_selection,
            "output_path": output_path if not use_llm_selection else None,
        }

        if save_logs:
            rank_kwargs["save_logs"] = True
            rank_kwargs["vad_timestamps"] = vad_timestamps

        if need_internals:
            ranked_frames, features, scores = ranker.rank_frames(
                **rank_kwargs,
                return_internals=True
            )
        else:
            ranked_frames = ranker.rank_frames(**rank_kwargs)
            features = None
            scores = None

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = ranked_frames[0]

        if use_llm_selection and features and scores:
            best_frame = select_best_with_llm(
                video_path, ranked_frames, speech_end_time,
                duration, scores, features, openai_api_key
            )

        if use_llm_selection:
            ranker._save_cut_video(video_path, best_frame.timestamp, output_path=output_path)

        if upload_to_airtable and features and scores:
            upload_results_to_airtable(
                video_path, best_frame, scores, features,
                speech_end_time, duration, config, sample_rate,
                airtable_access_token, airtable_base_id, airtable_table_name
            )

        logger.info("Best cut point: %.4fs (frame: %d, score: %.3f)", best_frame.timestamp, best_frame.frame_index, best_frame.score)
        return best_frame.timestamp

    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)
