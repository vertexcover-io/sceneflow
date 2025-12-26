"""Shared pipeline functions for SceneFlow.

This module contains core business logic used by both API and CLI.
Do not import this module directly - use through api or cli modules.
"""

import logging
from typing import List, Optional, Tuple

from sceneflow.detection import SpeechDetector, refine_speech_end, refine_speech_end_async
from sceneflow.selection import LLMFrameSelector
from sceneflow.shared.models import RankedFrame

logger = logging.getLogger(__name__)


def detect_speech_end(
    video_path: str,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
) -> Tuple[float, float]:
    """Detect when speech ends in a video using VAD and optionally refine using energy analysis.

    Args:
        video_path: Path to video file
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD

    Returns:
        Tuple of (speech_end_time, confidence) where:
        - speech_end_time: Final refined timestamp in seconds
        - confidence: Detection confidence score (0.0-1.0)
    """
    detector = SpeechDetector()
    vad_end_time, confidence = detector.get_speech_end_time(video_path)

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

    return speech_end_time, confidence


async def detect_speech_end_async(
    video_path: str,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
) -> Tuple[float, float]:
    """Async version of detect_speech_end.

    Detect when speech ends in a video using VAD and optionally refine using energy analysis.

    Args:
        video_path: Path to video file
        use_energy_refinement: If True, refines VAD with energy drops
        energy_threshold_db: Minimum dB drop to consider speech end
        energy_lookback_frames: Max frames to search backward from VAD

    Returns:
        Tuple of (speech_end_time, confidence) where:
        - speech_end_time: Final refined timestamp in seconds
        - confidence: Detection confidence score (0.0-1.0)
    """
    detector = SpeechDetector()
    vad_end_time, confidence = await detector.get_speech_end_time_async(video_path)

    logger.info("Speech end detected at: %.4fs (VAD)", vad_end_time)

    speech_end_time = vad_end_time

    if use_energy_refinement:
        result = await refine_speech_end_async(
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

    return speech_end_time, confidence


def select_best_with_llm(
    video_path: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    openai_api_key: Optional[str] = None,
) -> RankedFrame:
    """Use LLM to select best frame from top candidates.

    Args:
        video_path: Path to video file
        ranked_frames: List of ranked frames (uses top 5)
        speech_end_time: When speech ends in seconds
        duration: Video duration in seconds
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)

    Returns:
        Best frame selected by LLM, or top algorithmic result on failure
    """
    if len(ranked_frames) < 2:
        return ranked_frames[0]

    try:
        selector = LLMFrameSelector(api_key=openai_api_key)
        best_frame = selector.select_best_frame(
            video_path=video_path,
            ranked_frames=ranked_frames[:5],
            speech_end_time=speech_end_time,
            video_duration=duration,
        )
        logger.info("LLM selected frame at %.4fs", best_frame.timestamp)
        return best_frame
    except Exception as e:
        logger.warning("LLM selection failed: %s, using top algorithmic result", e)
        return ranked_frames[0]


async def select_best_with_llm_async(
    video_path: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    openai_api_key: Optional[str] = None,
) -> RankedFrame:
    """Async version of select_best_with_llm.

    Use LLM to select best frame from top candidates.

    Args:
        video_path: Path to video file
        ranked_frames: List of ranked frames (uses top 5)
        speech_end_time: When speech ends in seconds
        duration: Video duration in seconds
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)

    Returns:
        Best frame selected by LLM, or top algorithmic result on failure
    """
    if len(ranked_frames) < 2:
        return ranked_frames[0]

    try:
        selector = LLMFrameSelector(api_key=openai_api_key)
        best_frame = await selector.select_best_frame_async(
            video_path=video_path,
            ranked_frames=ranked_frames[:5],
            speech_end_time=speech_end_time,
            video_duration=duration,
        )
        logger.info("LLM selected frame at %.4fs", best_frame.timestamp)
        return best_frame
    except Exception as e:
        logger.warning("LLM selection failed: %s, using top algorithmic result", e)
        return ranked_frames[0]
