"""High-level API functions for SceneFlow.

This module provides the main public API functions for finding optimal
cut points in videos using multi-stage analysis.
"""

import logging
from typing import List, Optional, Tuple

from sceneflow.shared.config import RankingConfig
from sceneflow.detection import EnergyRefiner
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.selection import LLMFrameSelector
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_duration,
)

logger = logging.getLogger(__name__)


def _detect_speech_end(
    video_path: str,
    use_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int
) -> float:
    """
    Detect when speech ends in video using VAD and optional energy refinement.

    Args:
        video_path: Path to video file
        use_energy_refinement: Whether to refine with energy analysis
        energy_threshold_db: Energy drop threshold in dB
        energy_lookback_frames: Frames to search backward

    Returns:
        Speech end timestamp in seconds
    """
    logger.info("Stage 1: Detecting speech end time...")
    detector = SpeechDetector()
    vad_speech_end_time = detector.get_speech_end_time(video_path)
    logger.info("VAD detected speech end at: %.2fs", vad_speech_end_time)

    speech_end_time = vad_speech_end_time

    if use_energy_refinement:
        logger.info("Stage 1.5: Refining speech end time with energy analysis...")
        refiner = EnergyRefiner(
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames
        )
        result = refiner.refine_speech_end(
            vad_speech_end_time,
            video_path
        )

        speech_end_time = result.refined_timestamp

        if result.frames_adjusted > 0:
            logger.info(
                "Energy refinement adjusted timestamp by %d frames",
                result.frames_adjusted
            )
        else:
            logger.info("Energy refinement: No adjustment needed")

    return speech_end_time


def _rank_frames(
    video_path: str,
    speech_end_time: float,
    duration: float,
    config: Optional[RankingConfig],
    sample_rate: int,
    return_internals: bool = False
) -> Tuple[List[RankedFrame], Optional[List[FrameFeatures]], Optional[List[FrameScore]]]:
    """Rank frames after speech ends."""
    logger.info("Stage 2: Ranking frames based on visual quality...")
    logger.info("Analyzing frames from %.4fs to %.4fs", speech_end_time, duration)

    ranker = CutPointRanker(config)

    if return_internals:
        ranked_frames, features, scores = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            return_internals=True
        )
        return ranked_frames, features, scores
    else:
        ranked_frames = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )
        return ranked_frames, None, None


def _select_best_with_llm(
    video_path: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    openai_api_key: Optional[str]
) -> RankedFrame:
    """Use LLM to select best frame from top candidates."""
    if len(ranked_frames) < 2:
        return ranked_frames[0]
    try:
        selector = LLMFrameSelector(api_key=openai_api_key)
        return selector.select_best_frame(
            video_path=video_path,
            ranked_frames=ranked_frames,
            speech_end_time=speech_end_time,
            video_duration=duration,
        )
    except Exception as e:
        logger.warning("LLM selection failed: %s, using top result", e)
        return ranked_frames[0]


def _upload_to_airtable(
    video_path: str,
    best_frame: RankedFrame,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    speech_end_time: float,
    duration: float,
    config: Optional[RankingConfig],
    sample_rate: int,
    airtable_access_token: Optional[str],
    airtable_base_id: Optional[str],
    airtable_table_name: Optional[str]
) -> str:
    """Upload analysis results to Airtable."""
    from sceneflow.integration import upload_to_airtable as airtable_upload

    best_score = next((s for s in scores if s.frame_index == best_frame.frame_index), None)
    best_features = next((f for f in features if f.frame_index == best_frame.frame_index), None)

    if not best_score or not best_features:
        raise RuntimeError("Could not upload to Airtable - missing data")

    config_dict = {
        "sample_rate": sample_rate,
        "weights": {
            "eye": config.eye_weight if config else 0.4,
            "mouth": config.mouth_weight if config else 0.6,
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
    return record_id





def get_cut_frame(
    source: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 2,
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
        speech_end_time = _detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames
        )

        duration = get_video_duration(video_path)
        need_internals = use_llm_selection or upload_to_airtable

        ranked_frames, features, scores = _rank_frames(
            video_path=video_path,
            speech_end_time=speech_end_time,
            duration=duration,
            config=config,
            sample_rate=sample_rate,
            return_internals=need_internals
        )

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = ranked_frames[0]

        if use_llm_selection and features and scores:
            best_frame = _select_best_with_llm(
                video_path, ranked_frames, speech_end_time,
                duration, scores, features, openai_api_key
            )

        if upload_to_airtable and features and scores:
            _upload_to_airtable(
                video_path, best_frame, scores, features,
                speech_end_time, duration, config, sample_rate,
                airtable_access_token, airtable_base_id, airtable_table_name
            )

        logger.info("Best cut point: %.2fs (score: %.3f)", best_frame.timestamp, best_frame.score)
        return best_frame.timestamp

    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)


def get_ranked_cut_frames(
    source: str,
    n: int = 5,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 2,
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
        speech_end_time = _detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames
        )

        duration = get_video_duration(video_path)

        ranked_frames, features, scores = _rank_frames(
            video_path=video_path,
            speech_end_time=speech_end_time,
            duration=duration,
            config=config,
            sample_rate=sample_rate,
            return_internals=upload_to_airtable
        )

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if upload_to_airtable and features and scores:
            _upload_to_airtable(
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
    sample_rate: int = 2,
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
    Cut a video at the optimal point and save the cut video.

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
        # Stage 1: Detect speech end
        speech_end_time = _detect_speech_end(
            video_path,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames
        )

        duration = get_video_duration(video_path)
        need_internals = use_llm_selection or upload_to_airtable

        # Stage 2: Rank frames with save options
        ranker = CutPointRanker(config)

        if need_internals:
            ranked_frames, features, scores = ranker.rank_frames(
                video_path=video_path,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_frames=save_frames,
                save_video=not use_llm_selection,  # Save now if not using LLM, save later if using LLM
                output_path=output_path if not use_llm_selection else None,
                save_logs=save_logs,
                return_internals=True
            )
        else:
            ranked_frames = ranker.rank_frames(
                video_path=video_path,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_frames=save_frames,
                save_video=not use_llm_selection,  # Save now if not using LLM, save later if using LLM
                output_path=output_path if not use_llm_selection else None,
                save_logs=save_logs,
            )
            features = None
            scores = None

        if not ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        best_frame = ranked_frames[0]

        # Stage 3: Optional LLM selection
        if use_llm_selection and features and scores:
            best_frame = _select_best_with_llm(
                video_path, ranked_frames, speech_end_time,
                duration, scores, features, openai_api_key
            )

        # Save video with LLM-selected timestamp (only when using LLM selection)
        if use_llm_selection:
            ranker._save_cut_video(video_path, best_frame.timestamp, output_path=output_path)

        # Upload to Airtable if requested
        if upload_to_airtable and features and scores:
            _upload_to_airtable(
                video_path, best_frame, scores, features,
                speech_end_time, duration, config, sample_rate,
                airtable_access_token, airtable_base_id, airtable_table_name
            )

        logger.info("Best cut point: %.2fs (score: %.3f)", best_frame.timestamp, best_frame.score)
        return best_frame.timestamp

    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)
