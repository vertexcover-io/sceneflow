"""Internal pipeline functions for SceneFlow API.

This module contains implementation details and should not be imported directly.
Use the public API functions from sceneflow.api instead.
"""

import logging
from typing import Dict, List, Optional, Tuple

from sceneflow.shared.config import RankingConfig
from sceneflow.detection import EnergyRefiner
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector
from sceneflow.selection import LLMFrameSelector
from sceneflow.utils.video import get_video_duration

logger = logging.getLogger(__name__)


def detect_speech_end(
    video_path: str,
    use_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Detect when speech ends in video using VAD and optional refinements."""
    logger.info("Stage 1: Detecting speech end time...")
    detector = SpeechDetector()

    vad_speech_end_time, vad_timestamps = detector.get_speech_timestamps(video_path)
    logger.info("VAD detected speech end at: %.4fs", vad_speech_end_time)

    speech_end_time = vad_speech_end_time
    pre_refinement_time = vad_speech_end_time

    if use_energy_refinement:
        logger.info("Stage 1.5: Refining VAD-detected speech end time with energy analysis...")

        before_energy = speech_end_time

        refiner = EnergyRefiner(
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames
        )
        result = refiner.refine_speech_end(
            speech_end_time,
            video_path
        )

        speech_end_time = result.refined_timestamp

        if result.frames_adjusted > 0:
            logger.info(
                "Energy refinement adjusted timestamp by %d frames",
                result.frames_adjusted
            )
            pre_refinement_time = before_energy
        else:
            logger.info("Energy refinement: No adjustment needed")

    visual_search_end_time = pre_refinement_time if speech_end_time < pre_refinement_time else -1.0

    return speech_end_time, visual_search_end_time, vad_timestamps


def rank_frames(
    video_path: str,
    speech_end_time: float,
    duration: float,
    config: Optional[RankingConfig],
    sample_rate: int,
    visual_search_end_time: float = -1.0,
    return_internals: bool = False
) -> Tuple[List[RankedFrame], Optional[List[FrameFeatures]], Optional[List[FrameScore]]]:
    """Rank frames after speech ends."""
    end_time = visual_search_end_time if visual_search_end_time > 0 else duration

    logger.info("Stage 2: Ranking frames based on visual quality...")
    logger.info("Analyzing frames from %.4fs to %.4fs", speech_end_time, end_time)

    ranker = CutPointRanker(config)

    if return_internals:
        ranked_frames, features, scores = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=end_time,
            sample_rate=sample_rate,
            return_internals=True
        )
        return ranked_frames, features, scores
    else:
        ranked_frames = ranker.rank_frames(
            video_path=video_path,
            start_time=speech_end_time,
            end_time=end_time,
            sample_rate=sample_rate,
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
