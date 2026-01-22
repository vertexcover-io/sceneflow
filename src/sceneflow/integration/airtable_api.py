"""High-level Airtable integration API for SceneFlow."""

import logging
from typing import Optional, Tuple, List

from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import NoValidFramesError
from sceneflow.integration.airtable import upload_to_airtable
from sceneflow.detection import SpeechDetector
from sceneflow.selection import LLMFrameSelector
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    VideoSession,
    is_url,
    download_video,
    cleanup_downloaded_video,
    cut_video,
)
from sceneflow.utils.output import save_annotated_frames, save_analysis_logs

logger = logging.getLogger(__name__)


def _run_analysis_pipeline(
    source: str,
    config: Optional[RankingConfig],
    sample_rate: int,
    use_llm_selection: bool,
    openai_api_key: Optional[str],
    use_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
    disable_visual_analysis: bool,
):
    """Internal helper to run the common analysis pipeline with VideoSession."""
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    with VideoSession(video_path) as session:
        detector = SpeechDetector()
        speech_end_time, _ = detector.get_speech_end_time(
            session,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
        )

        if disable_visual_analysis:
            raise ValueError("Cannot upload to Airtable when visual analysis is disabled")

        duration = session.properties.duration

        ranker = CutPointRanker(config)
        result = ranker.rank_frames(
            session=session,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
        )

        if not result.ranked_frames:
            raise NoValidFramesError("No valid frames found for ranking")

        if not result.features or not result.scores:
            raise RuntimeError("Failed to extract frame features and scores")

        best_frame = result.ranked_frames[0]

        if use_llm_selection and len(result.ranked_frames) > 1:
            selector = LLMFrameSelector(api_key=openai_api_key)
            best_frame = selector.select_best_frame(
                session,
                result.ranked_frames,
                speech_end_time,
                duration,
            )

        best_score = next(
            (s for s in result.scores if s.frame_index == best_frame.frame_index), None
        )
        best_features = next(
            (f for f in result.features if f.frame_index == best_frame.frame_index), None
        )

        if not best_score or not best_features:
            raise RuntimeError("Could not find score and features for best frame")

    return (
        video_path,
        is_url_source,
        speech_end_time,
        duration,
        best_frame,
        best_score,
        best_features,
        result.ranked_frames,
        result.features,
        result.scores,
        ranker,
    )


def analyze_and_upload_to_airtable(
    source: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[float, str]:
    """Analyze video and upload results to Airtable."""
    try:
        (
            video_path,
            is_url_source,
            speech_end_time,
            duration,
            best_frame,
            best_score,
            best_features,
            *_,
        ) = _run_analysis_pipeline(
            source,
            config,
            sample_rate,
            use_llm_selection,
            openai_api_key,
            use_energy_refinement,
            energy_threshold_db,
            energy_lookback_frames,
            disable_visual_analysis,
        )

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        logger.info(
            "Uploaded to Airtable: timestamp=%.4fs, record_id=%s",
            best_frame.timestamp,
            record_id,
        )
        return best_frame.timestamp, record_id

    finally:
        if "is_url_source" in locals() and is_url_source:
            cleanup_downloaded_video(video_path)


def analyze_ranked_and_upload_to_airtable(
    source: str,
    n: int = 5,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[List[float], str]:
    """Analyze video for top N cut points and upload best result to Airtable."""
    try:
        (
            video_path,
            is_url_source,
            speech_end_time,
            duration,
            best_frame,
            best_score,
            best_features,
            all_ranked_frames,
            *_,
        ) = _run_analysis_pipeline(
            source,
            config,
            sample_rate,
            use_llm_selection=False,
            openai_api_key=None,
            use_energy_refinement=use_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
        )

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        top_timestamps = [frame.timestamp for frame in all_ranked_frames[:n]]
        logger.info(
            "Uploaded to Airtable: top %d cut points, best at %.4fs, record_id=%s",
            len(top_timestamps),
            top_timestamps[0],
            record_id,
        )
        return top_timestamps, record_id

    finally:
        if "is_url_source" in locals() and is_url_source:
            cleanup_downloaded_video(video_path)


def cut_and_upload_to_airtable(
    source: str,
    output_path: str,
    config: Optional[RankingConfig] = None,
    sample_rate: int = 1,
    save_frames: bool = False,
    save_logs: bool = False,
    airtable_access_token: Optional[str] = None,
    airtable_base_id: Optional[str] = None,
    airtable_table_name: Optional[str] = None,
    use_llm_selection: bool = False,
    openai_api_key: Optional[str] = None,
    use_energy_refinement: bool = True,
    energy_threshold_db: float = 8.0,
    energy_lookback_frames: int = 20,
    disable_visual_analysis: bool = False,
) -> Tuple[float, str]:
    """Cut video at optimal point and upload results to Airtable."""
    video_path = source
    is_url_source = is_url(source)

    if is_url_source:
        video_path = download_video(source)

    try:
        with VideoSession(video_path) as session:
            detector = SpeechDetector()
            speech_end_time, _ = detector.get_speech_end_time(
                session,
                use_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames,
            )

            if disable_visual_analysis:
                raise ValueError("Cannot upload to Airtable when visual analysis is disabled")

            duration = session.properties.duration

            ranker = CutPointRanker(config)
            result = ranker.rank_frames(
                session=session,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
            )

            if not result.ranked_frames:
                raise NoValidFramesError("No valid frames found for ranking")

            if not result.features or not result.scores:
                raise RuntimeError("Failed to extract frame features and scores")

            best_frame = result.ranked_frames[0]

            if use_llm_selection and len(result.ranked_frames) > 1:
                selector = LLMFrameSelector(api_key=openai_api_key)
                best_frame = selector.select_best_frame(
                    session,
                    result.ranked_frames,
                    speech_end_time,
                    duration,
                )

            if save_frames:
                save_annotated_frames(session, result.ranked_frames, ranker.extractor)

            if save_logs:
                save_analysis_logs(video_path, result.ranked_frames, result.features, result.scores)

            best_score = next(
                (s for s in result.scores if s.frame_index == best_frame.frame_index), None
            )
            best_features = next(
                (f for f in result.features if f.frame_index == best_frame.frame_index), None
            )

            if not best_score or not best_features:
                raise RuntimeError("Could not find score and features for best frame")

        cut_video(video_path, best_frame.timestamp, output_path)

        config_dict = _build_config_dict(config, sample_rate)

        record_id = upload_to_airtable(
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=speech_end_time,
            duration=duration,
            config_dict=config_dict,
            access_token=airtable_access_token,
            base_id=airtable_base_id,
            table_name=airtable_table_name,
        )

        logger.info(
            "Cut video and uploaded to Airtable: timestamp=%.4fs, record_id=%s",
            best_frame.timestamp,
            record_id,
        )
        return best_frame.timestamp, record_id

    finally:
        if is_url_source:
            cleanup_downloaded_video(video_path)


def _build_config_dict(config: Optional[RankingConfig], sample_rate: int) -> dict:
    """Build configuration dictionary for Airtable metadata."""
    if config is None:
        config = RankingConfig()

    return {
        "sample_rate": sample_rate,
        "weights": {
            "eye_openness": config.eye_openness_weight,
            "motion_stability": config.motion_stability_weight,
            "expression_neutrality": config.expression_neutrality_weight,
            "pose_stability": config.pose_stability_weight,
            "visual_sharpness": config.visual_sharpness_weight,
        },
        "context_window_size": config.context_window_size,
        "quality_gate_percentile": config.quality_gate_percentile,
        "local_stability_window": config.local_stability_window,
    }
