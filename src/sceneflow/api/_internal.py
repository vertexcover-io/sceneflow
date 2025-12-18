"""Internal pipeline functions for SceneFlow API.

This module contains implementation details and should not be imported directly.
Use the public API functions from sceneflow.api instead.
"""

import logging
from typing import List, Optional

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
) -> float:
    """
    Detect when speech ends in a video using VAD and optionally refine using energy analysis.

    Returns:
        speech_end_time: Final refined speech end time in seconds.
    """
    logger.info("Stage 1: Detecting speech end time...")

    detector = SpeechDetector()
    vad_end_time, _ = detector.get_speech_end_time(video_path)

    logger.info("VAD detected speech end at: %.4f s", vad_end_time)

    speech_end_time = vad_end_time

    if use_energy_refinement:
        logger.info("Stage 1.5: Refining VAD-detected speech end time with energy analysis...")

        refiner = EnergyRefiner(
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames,
        )

        result = refiner.refine_speech_end(vad_end_time, video_path)
        frames_adjusted = result.vad_frame - result.refined_frame
        if result.vad_frame - result.refined_frame > 0:
            logger.info(
                "Energy refinement adjusted timestamp by %d frames",
                frames_adjusted,
            )

        speech_end_time = result.refined_timestamp
    else:
        logger.debug("Energy refinement disabled.")

    return speech_end_time


def select_best_with_llm(
    video_path: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    openai_api_key: Optional[str],
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
