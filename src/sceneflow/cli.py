"""SceneFlow CLI - Find optimal cut points in talking head videos."""

import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Annotated, Optional

import cv2
import cyclopts
import requests

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from .config import RankingConfig
from .ranker import CutPointRanker
from .speech_detector import SpeechDetector

app = cyclopts.App(
    name="sceneflow",
    help="Find optimal cut points in AI-generated talking head videos",
    version="0.1.0",
)


def _download_video(url: str) -> str:
    """
    Download video from URL to temporary directory using HTTP GET.

    Args:
        url: Direct video URL (e.g., .mp4, .avi file URLs)

    Returns:
        Path to downloaded video file
    """
    logger.info(f"Downloading video from URL: {url}")

    # Create temp directory for downloaded video
    temp_dir = Path(tempfile.mkdtemp(prefix="sceneflow_"))

    # Extract filename from URL or use default
    url_path = Path(url.split("?")[0])  # Remove query params
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Write to file
        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")

        logger.info(f"Video downloaded successfully to: {output_path}")
        return str(output_path)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download video: {str(e)}")
        print(f"Error: Failed to download video: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        print(f"Error downloading video: {str(e)}", file=sys.stderr)
        sys.exit(1)


def _is_url(path: str) -> bool:
    """Check if the path is a URL."""
    return path.startswith(("http://", "https://", "www."))


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


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
    # Download video if URL
    if _is_url(source):
        if verbose:
            print(f"Downloading video from URL...")
        logger.info("Source is URL, downloading video...")
        source = _download_video(source)
        if verbose:
            print(f"Downloaded to: {source}")

    # Validate video path
    if not Path(source).exists():
        logger.error(f"Video file not found: {source}")
        print(f"Error: Video file not found: {source}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Analyzing video: {Path(source).name}")

    # Auto-enable save_video if output path is specified
    if output and not save_video:
        save_video = True
        logger.info(f"Auto-enabling video save because output path was specified: {output}")
        if verbose:
            print(f"Note: Automatically enabling video save to: {output}")

    try:
        # Print verbose header
        if verbose:
            print("=" * 60)
            print("SCENEFLOW - Smart Video Cut Point Detection")
            print("=" * 60)
            print(f"\nAnalyzing: {Path(source).name}")
            if use_llm_selection:
                print("\nUsing LLM-powered frame selection (GPT-4o)")
            if no_energy_refinement:
                print("\nEnergy refinement: DISABLED")
            else:
                print(f"\nEnergy refinement: ENABLED (threshold={energy_threshold_db} dB, lookback={energy_lookback_frames} frames)")

        # Use high-level API for ranked results
        if top_n is not None:
            from sceneflow.api import get_ranked_cut_frames

            ranked_timestamps = get_ranked_cut_frames(
                source=source,
                n=top_n,
                sample_rate=sample_rate,
                save_frames=save_frames,
                upload_to_airtable=airtable,
                use_energy_refinement=not no_energy_refinement,
                energy_threshold_db=energy_threshold_db,
                energy_lookback_frames=energy_lookback_frames,
            )

            # Need to get the frames for detailed output
            # Re-run with internals to get frame details
            from sceneflow.speech_detector import SpeechDetector
            from sceneflow.energy_refiner import EnergyRefiner

            detector = SpeechDetector()
            vad_speech_end = detector.get_speech_end_time(source)

            speech_end_time = vad_speech_end
            if not no_energy_refinement:
                refiner = EnergyRefiner(
                    threshold_db=energy_threshold_db,
                    lookback_frames=energy_lookback_frames
                )
                speech_end_time, _ = refiner.refine_speech_end(vad_speech_end, source)

            duration = _get_video_duration(source)

            ranker = CutPointRanker()
            ranked_frames = ranker.rank_frames(
                video_path=source,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
                save_frames=save_frames,
                save_video=save_video,
                output_path=output,
                save_logs=save_logs,
            )
        else:
            # Single best result - use simplified flow
            from sceneflow.speech_detector import SpeechDetector
            from sceneflow.energy_refiner import EnergyRefiner

            if verbose:
                print(f"\n[1/2] Detecting speech end time using VAD...")

            detector = SpeechDetector()
            speech_end_time, confidence = detector.get_speech_end_time(source, return_confidence=True)

            if verbose:
                print(f"      Speech ends at: {speech_end_time:.2f}s (confidence: {confidence:.2f})")

            # Energy refinement
            vad_speech_end = speech_end_time
            if not no_energy_refinement:
                if verbose:
                    print(f"\n[1.5/2] Refining speech end time with energy analysis...")

                refiner = EnergyRefiner(
                    threshold_db=energy_threshold_db,
                    lookback_frames=energy_lookback_frames
                )
                speech_end_time, metadata = refiner.refine_speech_end(vad_speech_end, source)

                if verbose and metadata['frames_adjusted'] > 0:
                    print(f"      Adjusted by {metadata['frames_adjusted']} frames (energy drop: {metadata['energy_drop_db']:.2f} dB)")

            # Get video duration
            duration = _get_video_duration(source)

            if verbose:
                print(f"      Video duration: {duration:.2f}s")
                print(f"\n[2/2] Analyzing visual features from {speech_end_time:.2f}s to {duration:.2f}s...")

            # Visual Frame Ranking
            ranker = CutPointRanker()

            # Get internals if we need them for LLM or Airtable
            if use_llm_selection or airtable:
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
            else:
                ranked_frames = ranker.rank_frames(
                    video_path=source,
                    start_time=speech_end_time,
                    end_time=duration,
                    sample_rate=sample_rate,
                    save_frames=save_frames,
                    save_video=save_video,
                    output_path=output,
                    save_logs=save_logs,
                )

        if not ranked_frames:
            logger.error("No suitable cut points found")
            print("Error: No suitable cut points found", file=sys.stderr)
            sys.exit(1)

        logger.info(f"Successfully analyzed {len(ranked_frames)} frames")

        best_frame = ranked_frames[0]

        # Apply LLM selection if requested
        if use_llm_selection and len(ranked_frames) > 1 and not top_n:
            try:
                if verbose:
                    print(f"\n[3/3] Using LLM to select best frame from top 5 candidates...")

                from sceneflow.llm_selector import LLMFrameSelector

                selector = LLMFrameSelector()
                best_frame = selector.select_best_frame(
                    video_path=source,
                    ranked_frames=ranked_frames[:5],
                    speech_end_time=speech_end_time,
                    video_duration=duration,
                    all_scores=scores if (use_llm_selection or airtable) else None,
                    all_features=features if (use_llm_selection or airtable) else None
                )

                if verbose:
                    print(f"      LLM selected frame at {best_frame.timestamp:.2f}s (frame {best_frame.frame_index})")

                logger.info(f"LLM selected frame at {best_frame.timestamp:.2f}s")
            except Exception as e:
                logger.warning(f"LLM selection failed: {str(e)}, using top algorithmic result")
                if verbose:
                    print(f"      LLM selection failed: {str(e)}")
                    print(f"      Falling back to top algorithmic result")
                best_frame = ranked_frames[0]

        # Save video with correct timestamp (after LLM selection if enabled)
        if save_video and (use_llm_selection or airtable):
            ranker._save_cut_video(source, best_frame.timestamp, output_path=output)

        logger.info(f"Best cut point: {best_frame.timestamp:.2f}s (score: {best_frame.score:.4f})")

        # Output results based on mode
        if top_n is not None:
            # Mode: -n flag provided (with or without --verbose)
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
                        f"  {i}. {frame.timestamp:.2f}s (frame {frame.frame_index}, score: {frame.score:.4f})"
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
                # Compact output: just numbered list with timestamps and scores
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
            print(f"\nTop 3 candidates:")
            for i, frame in enumerate(ranked_frames[:3], 1):
                print(
                    f"  {i}. {frame.timestamp:.2f}s (frame {frame.frame_index}, score: {frame.score:.4f})"
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

        # Upload to Airtable if requested
        if airtable:
            try:
                if verbose:
                    print(f"\n{'=' * 60}")
                    print("AIRTABLE UPLOAD")
                    print(f"{'=' * 60}")
                    print("Uploading analysis to Airtable...")

                logger.info("Uploading results to Airtable...")

                from .airtable_uploader import upload_to_airtable

                # Use already-computed scores and features if available
                if use_llm_selection or airtable:
                    # Already have scores and features from earlier
                    best_score = next(
                        (s for s in scores if s.frame_index == best_frame.frame_index), None
                    )
                    best_features = next(
                        (f for f in features if f.frame_index == best_frame.frame_index), None
                    )
                else:
                    # Need to get scores and features
                    detailed_scores = ranker.get_detailed_scores(
                        video_path=source,
                        start_time=speech_end_time,
                        end_time=duration,
                        sample_rate=sample_rate,
                    )
                    best_score = next(
                        (s for s in detailed_scores if s.frame_index == best_frame.frame_index), None
                    )

                    # Get features for the best frame
                    all_features = ranker._extract_features(
                        source, speech_end_time, duration, sample_rate
                    )
                    best_features = next(
                        (f for f in all_features if f.frame_index == best_frame.frame_index), None
                    )

                if not best_score:
                    logger.error("Could not find score for best frame")
                    print(
                        "Warning: Could not upload to Airtable - missing score data",
                        file=sys.stderr,
                    )
                else:

                    if not best_features:
                        logger.error("Could not find features for best frame")
                        print(
                            "Warning: Could not upload to Airtable - missing feature data",
                            file=sys.stderr,
                        )
                    else:
                        # Prepare config dict
                        config_dict = {
                            "sample_rate": sample_rate,
                            "weights": {
                                "eye_openness": 0.30,
                                "motion_stability": 0.25,
                                "expression_neutrality": 0.20,
                                "pose_stability": 0.15,
                                "visual_sharpness": 0.10,
                            },
                        }

                        # Upload to Airtable
                        record_id = upload_to_airtable(
                            video_path=source,
                            best_frame=best_frame,
                            frame_score=best_score,
                            frame_features=best_features,
                            speech_end_time=speech_end_time,
                            duration=duration,
                            config_dict=config_dict,
                        )

                        logger.info(f"Successfully uploaded to Airtable! Record ID: {record_id}")

                        if verbose:
                            print(f"Successfully uploaded to Airtable!")
                            print(f"Record ID: {record_id}")

            except ImportError:
                logger.error("pyairtable is not installed")
                print(
                    "Error: pyairtable is not installed. Install with: pip install pyairtable>=3.2.0",
                    file=sys.stderr,
                )
            except Exception as e:
                logger.error(f"Failed to upload to Airtable: {str(e)}")
                print(f"Error uploading to Airtable: {str(e)}", file=sys.stderr)
                if verbose:
                    import traceback

                    traceback.print_exc()

        # Save JSON if requested
        if json_output:
            json_dir = Path(json_output)
            json_dir.mkdir(parents=True, exist_ok=True)

            video_name = Path(source).stem
            json_path = json_dir / f"{video_name}_analysis.json"

            # Get detailed scores
            detailed_scores = ranker.get_detailed_scores(
                video_path=source,
                start_time=speech_end_time,
                end_time=duration,
                sample_rate=sample_rate,
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
                    "eye_openness_weight": 0.30,
                    "motion_stability_weight": 0.25,
                    "expression_neutrality_weight": 0.20,
                    "pose_stability_weight": 0.15,
                    "visual_sharpness_weight": 0.10,
                    "sample_rate": sample_rate,
                },
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2)

            if verbose:
                print(f"\nDetailed analysis saved to: {json_path}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app()
