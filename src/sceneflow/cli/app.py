"""SceneFlow CLI - Find optimal cut points in talking head videos."""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional, Tuple, List

import cyclopts

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from sceneflow.api import get_ranked_cut_frames
from sceneflow.shared.config import RankingConfig
from sceneflow.detection import EnergyRefiner
from sceneflow.shared.exceptions import VideoDownloadError, VideoNotFoundError
from sceneflow.selection import LLMFrameSelector
from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector
from sceneflow.utils.video import (
    is_url,
    download_video,
    cleanup_downloaded_video,
    get_video_duration,
    validate_video_path,
)

app = cyclopts.App(
    name="sceneflow",
    help="Find optimal cut points in AI-generated talking head videos",
    version="0.1.0",
)


def _print_verbose_header(
    source: str,
    use_llm_selection: bool,
    no_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int
) -> None:
    """Print verbose analysis header."""
    print("=" * 60)
    print("SCENEFLOW - Smart Video Cut Point Detection")
    print("=" * 60)
    print(f"\nAnalyzing: {Path(source).name}")

    if use_llm_selection:
        print("\nUsing LLM-powered frame selection (GPT-4o)")

    if no_energy_refinement:
        print("\nEnergy refinement: DISABLED")
    else:
        print(
            f"\nEnergy refinement: ENABLED "
            f"(threshold={energy_threshold_db} dB, lookback={energy_lookback_frames} frames)"
        )


def _detect_speech_end_cli(
    source: str,
    no_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
    verbose: bool
) -> Tuple[float, float, float]:
    """
    Detect speech end time with CLI-specific logging.

    Returns:
        Tuple of (speech_end_time, vad_speech_end_time, confidence)
    """
    if verbose:
        print("\n[1/2] Detecting speech end time using VAD...")

    detector = SpeechDetector()
    vad_speech_end_time, confidence = detector.get_speech_end_time(source, return_confidence=True)

    if verbose:
        print(f"      Speech ends at: {vad_speech_end_time:.2f}s (confidence: {confidence:.2f})")

    speech_end_time = vad_speech_end_time

    if not no_energy_refinement:
        if verbose:
            print("\n[1.5/2] Refining speech end time with energy analysis...")

        refiner = EnergyRefiner(
            threshold_db=energy_threshold_db,
            lookback_frames=energy_lookback_frames
        )
        result = refiner.refine_speech_end(vad_speech_end_time, source)
        speech_end_time = result.refined_timestamp

        if verbose and result.frames_adjusted > 0:
            print(
                f"      Adjusted by {result.frames_adjusted} frames "
                f"(energy drop: {result.energy_drop_db:.2f} dB)"
            )

    return speech_end_time, vad_speech_end_time, confidence


def _rank_frames_cli(
    source: str,
    speech_end_time: float,
    duration: float,
    sample_rate: int,
    save_frames: bool,
    save_video: bool,
    output: Optional[str],
    save_logs: bool,
    need_internals: bool,
    verbose: bool
) -> Tuple[List[RankedFrame], Optional[List[FrameFeatures]], Optional[List[FrameScore]]]:
    """
    Rank frames with CLI-specific logging.

    Returns:
        Tuple of (ranked_frames, features, scores)
        If need_internals=False, features and scores are None
    """
    if verbose:
        print(f"\n[2/2] Analyzing visual features from {speech_end_time:.2f}s to {duration:.2f}s...")

    ranker = CutPointRanker()

    if need_internals:
        ranked_frames, features, scores = ranker.rank_frames(
            video_path=source,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_frames=save_frames,
            save_video=False,  # Save later with correct timestamp
            output_path=output,
            save_logs=save_logs,
            return_internals=True
        )
        return ranked_frames, features, scores
    else:
        ranked_frames = ranker.rank_frames(
            video_path=source,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_frames=save_frames,
            save_video=save_video,
            output_path=output,
            save_logs=save_logs
        )
        return ranked_frames, None, None


def _apply_llm_selection_cli(
    source: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    verbose: bool
) -> RankedFrame:
    """
    Apply LLM selection with CLI-specific logging.

    Returns:
        Best frame (falls back to top algorithmic result on error)
    """
    if len(ranked_frames) < 2:
        return ranked_frames[0]

    try:
        if verbose:
            print("\n[3/3] Using LLM to select best frame from top 5 candidates...")

        selector = LLMFrameSelector()
        best_frame = selector.select_best_frame(
            video_path=source,
            ranked_frames=ranked_frames[:5],
            speech_end_time=speech_end_time,
            video_duration=duration,
            all_scores=scores,
            all_features=features
        )

        if verbose:
            print(
                f"      LLM selected frame at {best_frame.timestamp:.2f}s "
                f"(frame {best_frame.frame_index})"
            )

        logger.info("LLM selected frame at %.2fs", best_frame.timestamp)
        return best_frame

    except Exception as e:
        logger.warning("LLM selection failed: %s, using top algorithmic result", e)
        if verbose:
            print(f"      LLM selection failed: {e}")
            print("      Falling back to top algorithmic result")
        return ranked_frames[0]


def _print_results(
    best_frame: RankedFrame,
    ranked_frames: List[RankedFrame],
    top_n: Optional[int],
    verbose: bool,
    save_frames: bool,
    save_video: bool,
    source: str
) -> None:
    """Print results based on mode (top-n, verbose, or simple)."""
    if top_n is not None:
        # Mode: -n flag provided
        n = min(top_n, len(ranked_frames))

        if verbose:
            # Full verbose output with top N candidates
            print(f"\n{'=' * 60}")
            print("RESULTS")
            print(f"{'=' * 60}")
            print(f"\nBest cut point: {best_frame.timestamp:.2f}s")
            print(f"Frame: {best_frame.frame_index}")
            print(f"Score: {best_frame.score:.4f}")
            print(f"\nTop {n} candidates:")
            for i, frame in enumerate(ranked_frames[:n], 1):
                print(
                    f"  {i}. {frame.timestamp:.2f}s "
                    f"(frame {frame.frame_index}, score: {frame.score:.4f})"
                )

            if len(ranked_frames) > n:
                print(f"\n  ... and {len(ranked_frames) - n} more candidates")

            # Show output locations
            if save_frames or save_video:
                print(f"\n{'=' * 60}")
                print("OUTPUT FILES")
                print(f"{'=' * 60}")
                video_name = Path(source).stem
                if save_frames:
                    print(f"Annotated frames: output/{video_name}/")
                if save_video:
                    print(f"Cut video: output/{video_name}_cut.mp4")
        else:
            # Compact output: just numbered list
            for i, frame in enumerate(ranked_frames[:n], 1):
                print(f"{i}. {frame.timestamp:.2f}s (score: {frame.score:.4f})")

    elif verbose:
        # Mode: --verbose only (no -n flag)
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"\nBest cut point: {best_frame.timestamp:.2f}s")
        print(f"Frame: {best_frame.frame_index}")
        print(f"Score: {best_frame.score:.4f}")
        print("\nTop 3 candidates:")
        for i, frame in enumerate(ranked_frames[:3], 1):
            print(
                f"  {i}. {frame.timestamp:.2f}s "
                f"(frame {frame.frame_index}, score: {frame.score:.4f})"
            )

        if len(ranked_frames) > 3:
            print(f"\n  ... and {len(ranked_frames) - 3} more candidates")

        # Show output locations
        if save_frames or save_video:
            print(f"\n{'=' * 60}")
            print("OUTPUT FILES")
            print(f"{'=' * 60}")
            video_name = Path(source).stem
            if save_frames:
                print(f"Annotated frames: output/{video_name}/")
            if save_video:
                print(f"Cut video: output/{video_name}_cut.mp4")
    else:
        # Mode: Default (no -n, no --verbose)
        # Just print the single best timestamp
        print(f"{best_frame.timestamp:.2f}")


def _save_json_output(
    json_output: str,
    source: str,
    duration: float,
    speech_end_time: float,
    confidence: float,
    best_frame: RankedFrame,
    ranked_frames: List[RankedFrame],
    ranker: CutPointRanker,
    sample_rate: int,
    top_n: Optional[int],
    verbose: bool
) -> None:
    """Save detailed analysis to JSON file."""
    json_dir = Path(json_output)
    json_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(source).stem
    json_path = json_dir / f"{video_name}_analysis.json"

    # Get detailed scores
    detailed_scores = ranker.get_detailed_scores(
        video_path=source,
        start_time=speech_end_time,
        end_time=duration,
        sample_rate=sample_rate
    )

    # Determine how many candidates to include in JSON
    json_candidate_count = min(top_n, len(ranked_frames)) if top_n is not None else 10

    analysis_data = {
        "video_path": str(source),
        "video_name": video_name,
        "duration": duration,
        "speech_detection": {
            "speech_end_time": speech_end_time,
            "confidence": confidence,
            "method": "Silero VAD",
        },
        "best_cut_point": {
            "timestamp": best_frame.timestamp,
            "frame_index": best_frame.frame_index,
            "score": best_frame.score,
        },
        "top_candidates": [
            {
                "rank": frame.rank,
                "timestamp": frame.timestamp,
                "frame_index": frame.frame_index,
                "score": frame.score,
            }
            for frame in ranked_frames[:json_candidate_count]
        ],
        "detailed_scores": [
            {
                "frame_index": score.frame_index,
                "timestamp": score.timestamp,
                "final_score": score.final_score,
                "composite_score": score.composite_score,
                "quality_penalty": score.quality_penalty,
                "stability_boost": score.stability_boost,
                "component_scores": {
                    "eye_openness": score.eye_openness_score,
                    "motion_stability": score.motion_stability_score,
                    "expression_neutrality": score.expression_neutrality_score,
                    "pose_stability": score.pose_stability_score,
                    "visual_sharpness": score.visual_sharpness_score,
                },
            }
            for score in detailed_scores[:json_candidate_count]
        ],
        "config": {
            "eye_openness_weight": 0.20,
            "motion_stability_weight": 0.25,
            "expression_neutrality_weight": 0.30,
            "pose_stability_weight": 0.15,
            "visual_sharpness_weight": 0.10,
            "sample_rate": sample_rate,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)

    if verbose:
        print(f"\nDetailed analysis saved to: {json_path}")


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
) -> None:
    """
    Analyze a talking head video and find the optimal cut point.

    The tool uses speech detection to find where speech ends, then analyzes
    visual features to rank potential cut points based on eye openness,
    motion stability, expression neutrality, pose stability, and visual sharpness.

    Examples:
        # Basic usage (just get the timestamp)
        sceneflow video.mp4

        # Get top 5 cut points with scores
        sceneflow video.mp4 --top-n 5

        # With detailed output
        sceneflow video.mp4 --verbose

        # Save analysis to JSON
        sceneflow video.mp4 --json-output ./output

        # From URL
        sceneflow https://example.com/video.mp4 --verbose

        # Save annotated frames, cut video, and logs
        sceneflow video.mp4 --save-frames --save-video --save-logs

        # Save video to custom output path (auto-enables --save-video)
        sceneflow video.mp4 --output /path/to/my_output.mp4

        # Upload results to Airtable (requires environment variables)
        sceneflow video.mp4 --airtable --verbose

        # Use LLM to select best frame from top candidates
        sceneflow video.mp4 --use-llm-selection --verbose

        # Disable energy refinement (use raw VAD timestamp)
        sceneflow video.mp4 --no-energy-refinement

        # Fine-tune energy refinement
        sceneflow video.mp4 --energy-threshold-db 10.0 --energy-lookback-frames 25

    Environment Variables:
        AIRTABLE_ACCESS_TOKEN   Your Airtable access token
        AIRTABLE_BASE_ID        Your Airtable base ID (e.g., appXXXXXXXXXXXXXX)
        AIRTABLE_TABLE_NAME     Table name (optional, defaults to "SceneFlow Analysis")
        OPENAI_API_KEY          OpenAI API key for LLM-powered frame selection
    """
    video_path = source
    is_downloaded = False

    try:
        # Download video if URL
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
            # Validate local file
            try:
                validate_video_path(source)
            except VideoNotFoundError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        logger.info("Analyzing video: %s", Path(video_path).name)

        # Auto-enable save_video if output path is specified
        if output and not save_video:
            save_video = True
            logger.info("Auto-enabling video save because output path was specified: %s", output)
            if verbose:
                print(f"Note: Automatically enabling video save to: {output}")

        # Print verbose header
        if verbose:
            _print_verbose_header(
                video_path,
                use_llm_selection,
                no_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames
            )

        # Use high-level API for top-n mode
        if top_n is not None:
            ranked_timestamps = get_ranked_cut_frames(
                source=video_path,
                n=top_n,
                sample_rate=sample_rate,
                save_frames=save_frames,
                upload_to_airtable=airtable,
                use_energy_refinement=not no_energy_refinement,
                energy_threshold_db=energy_threshold_db,
                energy_lookback_frames=energy_lookback_frames,
            )

            # Re-run with internals for detailed output
            speech_end_time, vad_speech_end_time, confidence = _detect_speech_end_cli(
                video_path,
                no_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames,
                verbose=False  # Already processed, don't print again
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
            # Single best result - manual flow
            speech_end_time, vad_speech_end_time, confidence = _detect_speech_end_cli(
                video_path,
                no_energy_refinement,
                energy_threshold_db,
                energy_lookback_frames,
                verbose
            )

            duration = get_video_duration(video_path)

            if verbose:
                print(f"      Video duration: {duration:.2f}s")

            # Rank frames
            need_internals = use_llm_selection or airtable
            ranked_frames, features, scores = _rank_frames_cli(
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

        # Apply LLM selection if requested (not for top-n mode)
        if use_llm_selection and len(ranked_frames) > 1 and not top_n:
            best_frame = _apply_llm_selection_cli(
                video_path,
                ranked_frames,
                speech_end_time,
                duration,
                scores,
                features,
                verbose
            )

        # Save video with correct timestamp (after LLM selection)
        if save_video and (use_llm_selection or airtable) and not top_n:
            ranker = CutPointRanker()
            ranker._save_cut_video(video_path, best_frame.timestamp, output_path=output)

        logger.info(
            "Best cut point: %.2fs (score: %.4f)",
            best_frame.timestamp,
            best_frame.score
        )

        # Print results
        _print_results(
            best_frame,
            ranked_frames,
            top_n,
            verbose,
            save_frames,
            save_video,
            video_path
        )

        # Upload to Airtable if requested (handled by API for top-n mode)
        if airtable and not top_n:
            try:
                from sceneflow.api import _upload_to_airtable

                if verbose:
                    print(f"\n{'=' * 60}")
                    print("AIRTABLE UPLOAD")
                    print(f"{'=' * 60}")
                    print("Uploading analysis to Airtable...")

                _upload_to_airtable(
                    video_path,
                    best_frame,
                    scores,
                    features,
                    speech_end_time,
                    duration,
                    None,  # config
                    sample_rate,
                    None,  # airtable_access_token
                    None,  # airtable_base_id
                    None,  # airtable_table_name
                )

                if verbose:
                    print("Successfully uploaded to Airtable!")

            except Exception as e:
                logger.error("Failed to upload to Airtable: %s", e)
                print(f"Error uploading to Airtable: {e}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()

        # Save JSON if requested
        if json_output:
            ranker = CutPointRanker()
            _save_json_output(
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
        # Clean up downloaded video
        if is_downloaded:
            cleanup_downloaded_video(video_path)


if __name__ == "__main__":
    app()
