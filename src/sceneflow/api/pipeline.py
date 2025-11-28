"""Pipeline helper functions for the SceneFlow API.

This module provides internal helper functions for the multi-stage processing pipeline.
"""

import logging
from typing import List, Optional, Tuple

from sceneflow.shared.config import RankingConfig
from sceneflow.detection import SpeechDetector, EnergyRefiner
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.core import CutPointRanker
from sceneflow.selection import LLMFrameSelector

logger = logging.getLogger(__name__)


def detect_speech_end(
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


def rank_frames(
    video_path: str,
    speech_end_time: float,
    duration: float,
    config: Optional[RankingConfig],
    sample_rate: int,
    save_frames: bool,
    save_logs: bool,
    save_video: bool = False,
    return_internals: bool = False
) -> Tuple[List[RankedFrame], Optional[List[FrameFeatures]], Optional[List[FrameScore]]]:
    """
    Rank frames after speech ends.

    Args:
        video_path: Path to video file
        speech_end_time: When speech ends (seconds)
        duration: Video duration (seconds)
        config: Ranking configuration
        sample_rate: Frame sampling rate
        save_frames: Whether to save annotated frames
        save_logs: Whether to save analysis logs
        save_video: Whether to save cut video
        return_internals: Whether to return features and scores

    Returns:
        Tuple of (ranked_frames, features, scores)
        If return_internals=False, features and scores are None

    Raises:
        NoValidFramesError: If no valid frames found
    """
    logger.info("Stage 2: Ranking frames based on visual quality...")
    logger.info("Analyzing frames from %.4fs to %.4fs", speech_end_time, duration)

    ranker = CutPointRanker(config)

    if return_internals:
        ranked_frames, features, scores = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_video=save_video,
            save_frames=save_frames,
            save_logs=save_logs,
            return_internals=True
        )
        return ranked_frames, features, scores
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
        return ranked_frames, None, None


def select_best_with_llm(
    video_path: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    openai_api_key: Optional[str]
) -> RankedFrame:
    """
    Use LLM to select best frame from top candidates.

    Args:
        video_path: Path to video file
        ranked_frames: All ranked frames
        speech_end_time: When speech ends
        duration: Video duration
        scores: All frame scores
        features: All frame features
        openai_api_key: OpenAI API key

    Returns:
        Best selected frame (falls back to algorithmic top result on error)
    """
    if len(ranked_frames) < 2:
        logger.info("Only one frame available, skipping LLM selection")
        return ranked_frames[0]

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
        logger.info(
            "LLM selected frame at %.2fs (frame %d)",
            best_frame.timestamp,
            best_frame.frame_index
        )
        return best_frame

    except Exception as e:
        logger.warning(
            "LLM selection failed: %s, using top algorithmic result",
            e
        )
        return ranked_frames[0]


def upload_to_airtable(
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
    """
    Upload analysis results to Airtable.

    Args:
        video_path: Path to video file
        best_frame: Best ranked frame
        scores: All frame scores
        features: All frame features
        speech_end_time: When speech ends
        duration: Video duration
        config: Ranking configuration
        sample_rate: Frame sampling rate
        airtable_access_token: Airtable access token
        airtable_base_id: Airtable base ID
        airtable_table_name: Table name

    Returns:
        Airtable record ID

    Raises:
        RuntimeError: If upload fails or data is missing
    """
    from sceneflow.integration import upload_to_airtable as airtable_upload

    logger.info("Uploading results to Airtable...")

    # Find score and features for best frame
    best_score = next(
        (s for s in scores if s.frame_index == best_frame.frame_index),
        None
    )
    best_features = next(
        (f for f in features if f.frame_index == best_frame.frame_index),
        None
    )

    if not best_score or not best_features:
        raise RuntimeError("Could not upload to Airtable - missing score or features data")

    # Prepare config dictionary
    config_dict = {
        "sample_rate": sample_rate,
        "weights": {
            "eye_openness": config.eye_openness_weight if config else 0.20,
            "motion_stability": config.motion_stability_weight if config else 0.25,
            "expression_neutrality": config.expression_neutrality_weight if config else 0.30,
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

    logger.info("Successfully uploaded to Airtable! Record ID: %s", record_id)
    return record_id
