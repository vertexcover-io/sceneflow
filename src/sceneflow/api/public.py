"""Public API for SceneFlow."""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.shared.models import RankedFrame, RankingResult
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    VideoSession,
    is_url,
    download_video_async,
    cleanup_downloaded_video_async,
    cut_video_async as _cut_video_util_async,
)
from sceneflow.utils.output import save_annotated_frames, save_analysis_logs
from sceneflow.detection import SpeechDetector
from sceneflow.selection import LLMFrameSelector

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Full analysis result with all internal data."""

    ranked_frames: List[RankedFrame]
    speech_end_time: float
    duration: float
    ranker: Optional[CutPointRanker] = None
    ranking_result: Optional[RankingResult] = None


async def run_analysis_async(
    video_path: str,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
    save_frames: bool = False,
) -> AnalysisResult:
    """Run full analysis pipeline with single video open.

    Args:
        video_path: Local path to video file (URLs must be downloaded before calling)
    """
    with VideoSession(video_path) as session:
        detector = SpeechDetector()
        speech_end_time, _ = await detector.get_speech_end_time_async(
            session,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            logger.info(
                "Visual analysis disabled - returning speech end time: %.4fs", speech_end_time
            )
            return AnalysisResult(
                ranked_frames=[
                    RankedFrame(rank=1, frame_index=0, timestamp=speech_end_time, score=1.0)
                ],
                speech_end_time=speech_end_time,
                duration=speech_end_time,
            )

        duration = session.properties.duration
        ranker = CutPointRanker(ranking_config)

        result = await ranker.rank_frames_async(
            session=session,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if save_frames:
            await asyncio.to_thread(
                save_annotated_frames,
                session,
                result.ranked_frames,
                ranker.extractor,
            )

        return AnalysisResult(
            ranked_frames=result.ranked_frames,
            speech_end_time=speech_end_time,
            duration=duration,
            ranker=ranker,
            ranking_result=result,
        )


async def get_cut_frames_async(
    source: str,
    n: int = 1,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> List[float]:
    """Get the top N best cut point timestamps for a video.

    Handles URL downloads automatically. For local videos, use run_analysis_async directly.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    video_path = source
    is_downloaded = False

    if is_url(source):
        video_path = await download_video_async(source)
        is_downloaded = True

    try:
        analysis = await run_analysis_async(
            video_path=video_path,
            ranking_config=ranking_config,
            sample_rate=sample_rate,
            use_energy_refinement=use_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
        )

        ranked_frames = list(analysis.ranked_frames)

        # Apply LLM selection if requested and we have multiple frames
        if use_llm_selection and n == 1 and len(ranked_frames) > 1:
            selector = LLMFrameSelector(api_key=openai_api_key)
            with VideoSession(video_path) as session:
                best_frame = await selector.select_best_frame_async(
                    session,
                    ranked_frames,
                    analysis.speech_end_time,
                    analysis.duration,
                )
            # Move LLM-selected frame to front
            ranked_frames.remove(best_frame)
            ranked_frames.insert(0, best_frame)

        timestamps = [f.timestamp for f in ranked_frames[:n]]
        logger.info("Found top %d cut points: best at %.4fs", len(timestamps), timestamps[0])
        return timestamps
    finally:
        if is_downloaded:
            await cleanup_downloaded_video_async(video_path)


def get_cut_frames(
    source: str,
    n: int = 1,
    ranking_config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> List[float]:
    """Sync wrapper for get_cut_frames_async."""
    return asyncio.run(
        get_cut_frames_async(
            source=source,
            n=n,
            ranking_config=ranking_config,
            sample_rate=sample_rate,
            use_llm_selection=use_llm_selection,
            openai_api_key=openai_api_key,
            use_energy_refinement=use_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
        )
    )


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
    """Find best cut point, save the cut video, and optionally save frames/logs.

    Handles URL downloads automatically. For local videos, use run_analysis_async directly.
    """
    video_path = source
    is_downloaded = False

    if is_url(source):
        video_path = await download_video_async(source)
        is_downloaded = True

    try:
        analysis = await run_analysis_async(
            video_path=video_path,
            ranking_config=ranking_config,
            sample_rate=sample_rate,
            use_energy_refinement=use_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
            save_frames=save_frames,
        )

        best_frame = analysis.ranked_frames[0]

        # Apply LLM selection if requested and we have multiple frames
        if use_llm_selection and len(analysis.ranked_frames) > 1:
            selector = LLMFrameSelector(api_key=openai_api_key)
            with VideoSession(video_path) as session:
                best_frame = await selector.select_best_frame_async(
                    session,
                    analysis.ranked_frames,
                    analysis.speech_end_time,
                    analysis.duration,
                )

        # Handle save_logs separately
        if (
            save_logs
            and analysis.ranking_result
            and analysis.ranking_result.features
            and analysis.ranking_result.scores
        ):
            await asyncio.to_thread(
                save_analysis_logs,
                video_path,
                analysis.ranked_frames,
                analysis.ranking_result.features,
                analysis.ranking_result.scores,
            )

        # Cut the video
        await _cut_video_util_async(video_path, best_frame.timestamp, output_path)

        logger.info(
            "Cut video saved to %s (cut point: %.4fs, score: %.4f)",
            output_path,
            best_frame.timestamp,
            best_frame.score,
        )
        return best_frame.timestamp
    finally:
        if is_downloaded:
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
    """Sync wrapper for cut_video_async."""
    return asyncio.run(
        cut_video_async(
            source=source,
            output_path=output_path,
            ranking_config=ranking_config,
            sample_rate=sample_rate,
            save_frames=save_frames,
            save_logs=save_logs,
            use_llm_selection=use_llm_selection,
            openai_api_key=openai_api_key,
            use_energy_refinement=use_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
        )
    )
