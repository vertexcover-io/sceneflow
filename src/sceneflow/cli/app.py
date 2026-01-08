"""SceneFlow CLI."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import cyclopts

from sceneflow.api import run_analysis_async, AnalysisResult
from sceneflow.shared.config import RankingConfig
from sceneflow.shared.exceptions import VideoDownloadError
from typing import List
from sceneflow.shared.models import RankedFrame
from sceneflow.utils.video import (
    is_url,
    download_video_async,
    cleanup_downloaded_video_async,
    cut_video_async as _cut_video_util_async,
    VideoSession,
)
from sceneflow.utils.output import save_analysis_logs_async
from sceneflow.selection import LLMFrameSelector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = cyclopts.App(
    name="sceneflow",
    help="Find optimal cut points in AI-generated talking head videos",
    version="0.1.0",
)


def _print_results(
    best_frame: RankedFrame,
    ranked_frames: List[RankedFrame],
    top_n: Optional[int],
) -> None:
    """Print results based on mode (top-n or simple timestamp)."""
    if top_n is not None:
        n = min(top_n, len(ranked_frames))
        for i, frame in enumerate(ranked_frames[:n], 1):
            print(f"{i}. {frame.timestamp:.4f}s (score: {frame.score:.4f})")
    else:
        print(f"{best_frame.timestamp:.4f}")


async def _upload_to_airtable(
    video_path: str,
    analysis: AnalysisResult,
    best_frame: RankedFrame,
    sample_rate: int,
) -> None:
    try:
        from sceneflow.integration.airtable import upload_to_airtable

        if (
            not analysis.ranking_result
            or not analysis.ranking_result.scores
            or not analysis.ranking_result.features
        ):
            raise RuntimeError("Missing analysis data for Airtable upload")

        best_score = next(
            (s for s in analysis.ranking_result.scores if s.frame_index == best_frame.frame_index),
            None,
        )
        best_features = next(
            (
                f
                for f in analysis.ranking_result.features
                if f.frame_index == best_frame.frame_index
            ),
            None,
        )

        if not best_score or not best_features:
            raise RuntimeError("Could not find score and features for best frame")

        default_config = RankingConfig()
        config_dict = {
            "sample_rate": sample_rate,
            "weights": {
                "eye_openness": default_config.eye_openness_weight,
                "motion_stability": default_config.motion_stability_weight,
                "expression_neutrality": default_config.expression_neutrality_weight,
                "pose_stability": default_config.pose_stability_weight,
                "visual_sharpness": default_config.visual_sharpness_weight,
            },
        }

        record_id = await asyncio.to_thread(
            upload_to_airtable,
            video_path=video_path,
            best_frame=best_frame,
            frame_score=best_score,
            frame_features=best_features,
            speech_end_time=analysis.speech_end_time,
            duration=analysis.duration,
            config_dict=config_dict,
            access_token=None,
            base_id=None,
            table_name=None,
        )

        logger.info("Uploaded to Airtable: %s", record_id)

    except Exception as e:
        logger.error("Failed to upload to Airtable: %s", e)
        print(f"Error uploading to Airtable: {e}", file=sys.stderr)


@app.default
async def main(
    source: Annotated[str, cyclopts.Parameter(help="Path to video file or URL")],
    sample_rate: Annotated[
        int, cyclopts.Parameter(help="Process every Nth frame (default: 2)")
    ] = 2,
    save_frames: Annotated[
        bool,
        cyclopts.Parameter(
            help="Save annotated frames with InsightFace 106 landmarks to output directory"
        ),
    ] = False,
    output: Annotated[
        Optional[str],
        cyclopts.Parameter(help="Output path for saved video (requires ffmpeg)"),
    ] = None,
    save_logs: Annotated[
        bool,
        cyclopts.Parameter(
            help="Save detailed feature extraction and scoring logs to logs directory"
        ),
    ] = False,
    top_n: Annotated[
        Optional[int],
        cyclopts.Parameter(help="Return top N ranked timestamps in sorted order (shows scores)"),
    ] = None,
    airtable: Annotated[
        bool,
        cyclopts.Parameter(
            help="Upload results to Airtable (requires AIRTABLE_ACCESS_TOKEN and AIRTABLE_BASE_ID env vars)"
        ),
    ] = False,
    use_llm_selection: Annotated[
        bool,
        cyclopts.Parameter(
            help="Use GPT-4o vision to select best frame from top 5 candidates (requires OPENAI_API_KEY env var)"
        ),
    ] = False,
    no_energy_refinement: Annotated[
        bool,
        cyclopts.Parameter(
            help="Disable energy-based refinement of VAD timestamp (uses raw VAD result)"
        ),
    ] = False,
    energy_threshold_db: Annotated[
        float,
        cyclopts.Parameter(
            help="Minimum dB drop to detect speech end for energy refinement (default: 8.0)"
        ),
    ] = 8.0,
    energy_lookback_frames: Annotated[
        int,
        cyclopts.Parameter(
            help="Maximum frames to search backward from VAD timestamp (default: 20)"
        ),
    ] = 20,
    disable_visual_analysis: Annotated[
        bool,
        cyclopts.Parameter(
            help="Disable visual analysis and return speech end time only (faster, no frame ranking)"
        ),
    ] = False,
) -> None:
    """Analyze a talking head video and find the optimal cut point."""
    if not is_url(source) and not Path(source).exists():
        print(f"Error: Video file not found: {source}", file=sys.stderr)
        sys.exit(1)

    video_path = source
    is_downloaded = False

    if is_url(source):
        video_path = await download_video_async(source)
        is_downloaded = True

    try:
        logger.info("Analyzing video: %s", Path(video_path).name)
        should_use_llm = use_llm_selection and not top_n

        analysis = await run_analysis_async(
            video_path=video_path,
            sample_rate=sample_rate,
            use_energy_refinement=not no_energy_refinement,
            energy_threshold_db=energy_threshold_db,
            energy_lookback_frames=energy_lookback_frames,
            disable_visual_analysis=disable_visual_analysis,
            save_frames=save_frames,
        )

        if not analysis.ranked_frames:
            logger.error("No suitable cut points found")
            print("Error: No suitable cut points found", file=sys.stderr)
            sys.exit(1)

        logger.info("Successfully analyzed %d frames", len(analysis.ranked_frames))

        best_frame = analysis.ranked_frames[0]

        if should_use_llm and len(analysis.ranked_frames) > 1:
            selector = LLMFrameSelector(api_key=None)
            async with VideoSession(video_path) as session:
                best_frame = await selector.select_best_frame_async(
                    session,
                    analysis.ranked_frames,
                    analysis.speech_end_time,
                    analysis.duration,
                )
            logger.info("LLM selected frame at %.4fs", best_frame.timestamp)

        logger.info("Best cut point: %.4fs (score: %.4f)", best_frame.timestamp, best_frame.score)

        _print_results(best_frame, analysis.ranked_frames, top_n)

        if save_logs and analysis:
            await save_analysis_logs_async(
                video_path,
                analysis.ranked_frames,
                analysis.ranking_result.features,
                analysis.ranking_result.scores,
            )

        if output and (not top_n):
            await _cut_video_util_async(video_path, best_frame.timestamp, output)
            logger.info("Cut video saved to: %s", output)

        if airtable and analysis:
            await _upload_to_airtable(
                video_path=video_path,
                analysis=analysis,
                best_frame=best_frame,
                sample_rate=sample_rate,
            )

    except VideoDownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if is_downloaded:
            await cleanup_downloaded_video_async(video_path)


if __name__ == "__main__":
    app()
