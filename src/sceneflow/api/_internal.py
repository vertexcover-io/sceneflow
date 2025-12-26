"""Internal pipeline functions for SceneFlow API.

This module contains implementation details and should not be imported directly.
Use the public API functions from sceneflow.api instead.
"""

import logging
from typing import List, Optional

from sceneflow.detection import refine_speech_end
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
    detector = SpeechDetector()
    vad_end_time, _ = detector.get_speech_end_time(video_path)

    logger.info("Speech end detected at: %.4fs (VAD)", vad_end_time)

    speech_end_time = vad_end_time

    if use_energy_refinement:
        result = refine_speech_end(
            vad_timestamp=vad_end_time,
            video_path=video_path,
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames,
        )
        frames_adjusted = result.vad_frame - result.refined_frame
        if frames_adjusted > 0:
            logger.info(
                "Speech end refined to: %.4fs (adjusted %d frames backward using energy analysis)",
                result.refined_timestamp,
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
