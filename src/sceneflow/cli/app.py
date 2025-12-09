"""SceneFlow CLI - Find optimal cut points in talking head videos."""

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import cyclopts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from sceneflow.api import get_ranked_cut_frames
from sceneflow.shared.exceptions import VideoDownloadError, VideoNotFoundError
from sceneflow.shared.models import RankedFrame
from sceneflow.core import CutPointRanker
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_duration,
    validate_video_path,
)
from sceneflow.cli._internal import (
    print_verbose_header,
    detect_speech_end_cli,
    rank_frames_cli,
    apply_llm_selection_cli,
    print_results,
    save_json_output,
)

app = cyclopts.App(
    name="sceneflow",
    help="Find optimal cut points in AI-generated talking head videos",
    version="0.1.0",
)


@app.default
def main(
    source: Annotated[str, cyclopts.Parameter(help="Path to video file or URL")],
    verbose: Annotated[bool, cyclopts.Parameter(help="Show detailed analysis information")] = False,
    json_output: Annotated[
        Optional[str],
        cyclopts.Parameter(help="Save detailed analysis to JSON file (directory path)"),
    ] = None,
    sample_rate: Annotated[
        int, cyclopts.Parameter(help="Process every Nth frame (default: 2)")
    ] = 2,
    save_frames: Annotated[
        bool,
        cyclopts.Parameter(
            help="Save annotated frames with InsightFace 106 landmarks to output directory"
        ),
    ] = False,
    save_video: Annotated[
        bool,
        cyclopts.Parameter(help="Save cut video from start to best timestamp (requires ffmpeg)"),
    ] = False,
    output: Annotated[
        Optional[str],
        cyclopts.Parameter(help="Output path for saved video (automatically enables --save-video)"),
    ] = None,
    save_logs: Annotated[
        bool,
        cyclopts.Parameter(help="Save detailed feature extraction and scoring logs to logs directory"),
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
        cyclopts.Parameter(help="Minimum dB drop to detect speech end for energy refinement (default: 8.0)"),
    ] = 8.0,
    energy_lookback_frames: Annotated[
        int,
        cyclopts.Parameter(help="Maximum frames to search backward from VAD timestamp (default: 20)"),
    ] = 20,
    disable_visual_analysis: Annotated[
        bool,
        cyclopts.Parameter(
            help="Disable visual analysis and return speech end time only (faster, no frame ranking)"
        ),
    ] = False,
) -> None:
    """Analyze a talking head video and find the optimal cut point.

    The tool uses speech detection to find where speech ends, then analyzes
    visual features to rank potential cut points based on eye openness,
    motion stability, expression neutrality, pose stability, and visual sharpness.

    Examples:
        sceneflow video.mp4
        sceneflow video.mp4 --top-n 5
        sceneflow video.mp4 --verbose
        sceneflow video.mp4 --json-output ./output
        sceneflow https://example.com/video.mp4 --verbose
        sceneflow video.mp4 --save-frames --save-video --save-logs
        sceneflow video.mp4 --output /path/to/my_output.mp4
        sceneflow video.mp4 --airtable --verbose
        sceneflow video.mp4 --use-llm-selection --verbose
        sceneflow video.mp4 --no-energy-refinement
        sceneflow video.mp4 --energy-threshold-db 10.0 --energy-lookback-frames 25
        sceneflow video.mp4 --disable-visual-analysis

    Environment Variables:
        AIRTABLE_ACCESS_TOKEN   Your Airtable access token
        AIRTABLE_BASE_ID        Your Airtable base ID (e.g., appXXXXXXXXXXXXXX)
        AIRTABLE_TABLE_NAME     Table name (optional, defaults to "SceneFlow Analysis")
        OPENAI_API_KEY          OpenAI API key for LLM-powered frame selection
    """
    video_path = source
    is_downloaded = False

    try:
        if is_url(source):
            if verbose:
                print("Downloading video from URL...")
            logger.info("Source is URL, downloading video...")

            try:
                video_path = download_video(source)
                is_downloaded = True
                if verbose:
                    print(f"Downloaded to: {video_path}")
            except VideoDownloadError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                validate_video_path(source)
            except VideoNotFoundError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        logger.info("Analyzing video: %s", Path(video_path).name)

        if output and not save_video:
            save_video = True
            logger.info("Auto-enabling video save because output path was specified: %s", output)
            if verbose:
                print(f"Note: Automatically enabling video save to: {output}")

        if verbose:
            print_verbose_header(
                video_path,
                use_llm_selection,
                no_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames
            )

        if top_n is not None:
            ranked_timestamps = get_ranked_cut_frames(
                source=video_path,
                n=top_n,
                sample_rate=sample_rate,
                upload_to_airtable=airtable,
                use_energy_refinement=not no_energy_refinement,
                energy_threshold_db=energy_threshold_db,
                energy_lookback_frames=energy_lookback_frames,
                disable_visual_analysis=disable_visual_analysis,
            )

            if disable_visual_analysis:
                speech_end_time, vad_speech_end_time, confidence = detect_speech_end_cli(
                    video_path,
                    no_energy_refinement,
                    energy_threshold_db,
                    energy_lookback_frames,
                    verbose=False
                )
                ranked_frames = [RankedFrame(
                    timestamp=speech_end_time,
                    frame_index=0,
                    score=1.0,
                    rank=1
                )]
            else:
                speech_end_time, vad_speech_end_time, confidence = detect_speech_end_cli(
                    video_path,
                    no_energy_refinement,
                    energy_threshold_db,
                    energy_lookback_frames,
                    verbose=False
                )

                duration = get_video_duration(video_path)

                ranker = CutPointRanker()
                ranked_frames = ranker.rank_frames(
                    video_path=video_path,
                    start_time=speech_end_time,
                    end_time=duration,
                    sample_rate=sample_rate,
                    save_frames=save_frames,
                    save_video=save_video,
                    output_path=output,
                    save_logs=save_logs,
                )
        else:
            speech_end_time, vad_speech_end_time, confidence = detect_speech_end_cli(
                video_path,
                no_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames,
                verbose
            )

            if disable_visual_analysis:
                if verbose:
                    print(f"\n{'=' * 60}")
                    print("Visual analysis disabled - using speech end time")
                    print(f"{'=' * 60}")

                ranked_frames = [RankedFrame(
                    timestamp=speech_end_time,
                    frame_index=0,
                    score=1.0,
                    rank=1
                )]
                features = None
                scores = None
            else:
                duration = get_video_duration(video_path)

                if verbose:
                    print(f"      Video duration: {duration:.4f}s")

                need_internals = use_llm_selection or airtable
                ranked_frames, features, scores = rank_frames_cli(
                    video_path,
                    speech_end_time,
                    duration,
                    sample_rate,
                    save_frames,
                    save_video,
                    output,
                    save_logs,
                    need_internals,
                    verbose
                )

        if not ranked_frames:
            logger.error("No suitable cut points found")
            print("Error: No suitable cut points found", file=sys.stderr)
            sys.exit(1)

        logger.info("Successfully analyzed %d frames", len(ranked_frames))

        best_frame = ranked_frames[0]

        if use_llm_selection and len(ranked_frames) > 1 and not top_n:
            best_frame = apply_llm_selection_cli(
                video_path,
                ranked_frames,
                speech_end_time,
                duration,
                scores,
                features,
                verbose
            )

        if save_video and (use_llm_selection or airtable) and not top_n:
            ranker = CutPointRanker()
            ranker._save_cut_video(video_path, best_frame.timestamp, output_path=output)

        logger.info(
            "Best cut point: %.4fs (score: %.4f)",
            best_frame.timestamp,
            best_frame.score
        )

        print_results(
            best_frame,
            ranked_frames,
            top_n,
            verbose,
            save_frames,
            save_video,
            video_path
        )

        if airtable and not top_n:
            try:
                from sceneflow.api._internal import upload_to_airtable

                if verbose:
                    print(f"\n{'=' * 60}")
                    print("AIRTABLE UPLOAD")
                    print(f"{'=' * 60}")
                    print("Uploading analysis to Airtable...")

                upload_to_airtable(
                    video_path,
                    best_frame,
                    scores,
                    features,
                    speech_end_time,
                    duration,
                    None,
                    sample_rate,
                    None,
                    None,
                    None,
                )

                if verbose:
                    print("Successfully uploaded to Airtable!")

            except Exception as e:
                logger.error("Failed to upload to Airtable: %s", e)
                print(f"Error uploading to Airtable: {e}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()

        if json_output:
            ranker = CutPointRanker()
            save_json_output(
                json_output,
                video_path,
                duration,
                speech_end_time,
                confidence,
                best_frame,
                ranked_frames,
                ranker,
                sample_rate,
                top_n,
                verbose
            )

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        if is_downloaded:
            cleanup_downloaded_video(video_path)


if __name__ == "__main__":
    app()
