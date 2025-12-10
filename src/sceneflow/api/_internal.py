"""Internal pipeline functions for SceneFlow API.

This module contains implementation details and should not be imported directly.
Use the public API functions from sceneflow.api instead.
"""

import logging
from typing import  List, Optional, Tuple

from sceneflow.shared.config import RankingConfig
from sceneflow.detection import EnergyRefiner
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.detection import SpeechDetector
from sceneflow.selection import LLMFrameSelector

logger = logging.getLogger(__name__)

def detect_speech_end(
    video_path: str,
    use_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
) -> Tuple[float, float]:
    """
    Detect when speech ends in a video using VAD and optionally refine using energy analysis.
    
    Returns:
        speech_end_time: Final refined speech end time in seconds.
        visual_search_end_time: Time to stop visual search, or -1 if no refinement.
    """
    logger.info("Stage 1: Detecting speech end time...")

    detector = SpeechDetector()
    vad_end_time, _ = detector.get_speech_end_time(video_path)

    logger.info("VAD detected speech end at: %.4f s", vad_end_time)

    speech_end_time = vad_end_time
    pre_refinement_time = vad_end_time

    if use_energy_refinement:
        logger.info("Stage 1.5: Refining VAD-detected speech end time with energy analysis...")

        refiner = EnergyRefiner(
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames,
        )

        result = refiner.refine_speech_end(vad_end_time, video_path)

        if result.frames_adjusted > 0:
            logger.info(
                "Energy refinement adjusted timestamp by %d frames",
                result.frames_adjusted,
            )
            pre_refinement_time = vad_end_time

        speech_end_time = result.refined_timestamp
    else:
        logger.debug("Energy refinement disabled.")


    return speech_end_time, pre_refinement_time




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
