"""High-level API functions for SceneFlow.

This module provides the main public API functions for finding optimal
cut points in videos using multi-stage analysis.
"""

import asyncio
import logging
from typing import List, Optional

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_properties,
    cut_video as _cut_video_util,
    download_video_async,
    cleanup_downloaded_video_async,
    get_video_properties_async,
    cut_video_async as _cut_video_util_async,
)
from sceneflow.utils.output import save_annotated_frames, save_analysis_logs
from sceneflow.shared.pipeline import (
    detect_speech_end,
    detect_speech_end_async,
    select_best_with_llm,
    select_best_with_llm_async,
)

logger = logging.getLogger(__name__)


def get_cut_frame(
    source: str,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
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
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        use_llm_selection: If True, uses LLM to select best frame from top 5
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time

    Returns:
        Timestamp in seconds of the best cut point

    Note:
        For Airtable integration, use analyze_and_upload_to_airtable() from
        sceneflow.integration instead.
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        speech_end_time, _ = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - returning speech end time: %.4fs", speech_end_time
            )
            return speech_end_time

        duration = get_video_properties(video_path).duration

        ranker = CutPointRanker(ranking_config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = result.ranked_frames[0]

        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = select_best_with_llm(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                openai_api_key,
            )

        logger.info(
            "Best cut point: %.4fs (frame: %d, score: %.4f)",
            best_frame.timestamp,
            best_frame.frame_index,
            best_frame.score,
        )
        return best_frame.timestamp

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


async def get_cut_frame_async(
    source: str,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> float:
    """
    Async version of get_cut_frame.

    Get the single best cut point timestamp for a video.

    Args:
        source: Video file path or direct video URL
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        use_llm_selection: If True, uses LLM to select best frame from top 5
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time

    Returns:
        Timestamp in seconds of the best cut point

    Note:
        For Airtable integration, use analyze_and_upload_to_airtable() from
        sceneflow.integration instead.
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = await download_video_async(source)

    try:
        speech_end_time, _ = await detect_speech_end_async(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - returning speech end time: %.4fs", speech_end_time
            )
            return speech_end_time

        duration = (await get_video_properties_async(video_path)).duration

        ranker = CutPointRanker(ranking_config)
        result = await ranker.rank_frames_async(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = result.ranked_frames[0]

        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = await select_best_with_llm_async(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                openai_api_key,
            )

        logger.info(
            "Best cut point: %.4fs (frame: %d, score: %.4f)",
            best_frame.timestamp,
            best_frame.frame_index,
            best_frame.score,
        )
        return best_frame.timestamp

    finally:
        if is_url_source:
            await cleanup_downloaded_video_async(video_path)


def get_ranked_cut_frames(
    source: str,
    n: int = 5,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> List[float]:
    """Get the top N best cut point timestamps for a video.

    Args:
        source: Video file path or direct video URL
        n: Number of top cut points to return (default: 5)
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time only

    Returns:
        List of timestamps in seconds for the top N cut points

    Note:
        For Airtable integration, use analyze_ranked_and_upload_to_airtable() from
        sceneflow.integration instead.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        speech_end_time, _ = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - returning speech end time: %.4fs", speech_end_time
            )
            return [speech_end_time]

        duration = get_video_properties(video_path).duration

        ranker = CutPointRanker(ranking_config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        top_timestamps = [frame.timestamp for frame in result.ranked_frames[:n]]
        logger.info(
            "Found top %d cut points: best at %.4fs", len(top_timestamps), top_timestamps[0]
        )
        return top_timestamps

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


async def get_ranked_cut_frames_async(
    source: str,
    n: int = 5,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> List[float]:
    """
    Async version of get_ranked_cut_frames.

    Get the top N best cut point timestamps for a video.

    Args:
        source: Video file path or direct video URL
        n: Number of top cut points to return (default: 5)
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and returns speech end time only

    Returns:
        List of timestamps in seconds for the top N cut points

    Note:
        For Airtable integration, use analyze_ranked_and_upload_to_airtable() from
        sceneflow.integration instead.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = await download_video_async(source)

    try:
        speech_end_time, _ = await detect_speech_end_async(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - returning speech end time: %.4fs", speech_end_time
            )
            return [speech_end_time]

        duration = (await get_video_properties_async(video_path)).duration

        ranker = CutPointRanker(ranking_config)
        result = await ranker.rank_frames_async(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        top_timestamps = [frame.timestamp for frame in result.ranked_frames[:n]]
        logger.info(
            "Found top %d cut points: best at %.4fs", len(top_timestamps), top_timestamps[0]
        )
        return top_timestamps

    finally:
        if is_url_source:
            await cleanup_downloaded_video_async(video_path)


def cut_video(
    source: str,
    output_path: str,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    save_frames: bool = False,
    save_logs: bool = False,
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
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        save_frames: If True, saves annotated frames with landmarks
        save_logs: If True, saves detailed analysis logs
        use_llm_selection: If True, uses LLM to select best frame from top 5
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and cuts at speech end time

    Returns:
        Timestamp in seconds of the best cut point
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        speech_end_time, _ = detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - cutting at speech end time: %.4fs", speech_end_time
            )
            _cut_video_util(video_path, speech_end_time, output_path)
            return speech_end_time

        duration = get_video_properties(video_path).duration

        ranker = CutPointRanker(ranking_config)
        result = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = result.ranked_frames[0]

        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = select_best_with_llm(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                openai_api_key,
            )

        # Handle side effects after best frame is determined
        if save_frames:
            save_annotated_frames(video_path, result.ranked_frames, ranker.extractor)

        if save_logs and result.features and result.scores:
            save_analysis_logs(video_path, result.ranked_frames, result.features, result.scores)

        _cut_video_util(video_path, best_frame.timestamp, output_path)

        logger.info(
            "Cut video saved to %s (best cut point: %.4fs, frame: %d, score: %.4f)",
            output_path,
            best_frame.timestamp,
            best_frame.frame_index,
            best_frame.score,
        )
        return best_frame.timestamp

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


async def cut_video_async(
    source: str,
    output_path: str,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    save_frames: bool = False,
    save_logs: bool = False,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> float:
    """
    Async version of cut_video.

    Cut a video at the optimal point and save the cut video.

    This function finds the best cut point and always saves the cut video.
    Optionally saves annotated frames and detailed logs.
    Use get_cut_frame_async() if you only need the timestamp without saving.

    Args:
        source: Video file path or direct video URL
        output_path: Output path for the cut video (required)
        ranking_config: Optional custom RankingConfig for scoring weights
        sample_rate: Process every Nth frame (default: 1)
        save_frames: If True, saves annotated frames with landmarks
        save_logs: If True, saves detailed analysis logs
        use_llm_selection: If True, uses LLM to select best frame from top 5
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD
        disable_visual_analysis: If True, skips visual ranking and cuts at speech end time

    Returns:
        Timestamp in seconds of the best cut point
    """
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = await download_video_async(source)

    try:
        speech_end_time, _ = await detect_speech_end_async(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - cutting at speech end time: %.4fs", speech_end_time
            )
            await _cut_video_util_async(video_path, speech_end_time, output_path)
            return speech_end_time

        duration = (await get_video_properties_async(video_path)).duration

        ranker = CutPointRanker(ranking_config)
        result = await ranker.rank_frames_async(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = result.ranked_frames[0]

        if use_llm_selection and len(result.ranked_frames) > 1:
            best_frame = await select_best_with_llm_async(
                video_path,
                result.ranked_frames,
                speech_end_time,
                duration,
                openai_api_key,
            )

        side_effect_tasks = []

        if save_frames:
            side_effect_tasks.append(
                asyncio.to_thread(
                    save_annotated_frames,
                    video_path,
                    result.ranked_frames,
                    ranker.extractor,
                )
            )

        if save_logs and result.features and result.scores:
            side_effect_tasks.append(
                asyncio.to_thread(
                    save_analysis_logs,
                    video_path,
                    result.ranked_frames,
                    result.features,
                    result.scores,
                )
            )

        side_effect_tasks.append(
            _cut_video_util_async(video_path, best_frame.timestamp, output_path)
        )

        await asyncio.gather(*side_effect_tasks)

        logger.info(
            "Cut video saved to %s (best cut point: %.4fs, frame: %d, score: %.4f)",
            output_path,
            best_frame.timestamp,
            best_frame.frame_index,
            best_frame.score,
        )
        return best_frame.timestamp

    finally:
        if is_url_source:
            await cleanup_downloaded_video_async(video_path)
