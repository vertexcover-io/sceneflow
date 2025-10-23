"""SceneFlow CLI - Find optimal cut points in talking head videos."""

import sys
import json
import tempfile
import warnings
import os
import logging
from pathlib import Path
from typing import Optional, Annotated
import cv2
import cyclopts
import requests

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

from .speech_detector import SpeechDetector
from .ranker import CutPointRanker
from .config import RankingConfig


app = cyclopts.App(
    name="sceneflow",
    help="Find optimal cut points in AI-generated talking head videos",
    version="0.1.0"
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
    url_path = Path(url.split('?')[0])  # Remove query params
    filename = url_path.name if url_path.suffix else "video.mp4"
    output_path = temp_dir / filename

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Write to file
        with open(output_path, 'wb') as f:
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
    return path.startswith(('http://', 'https://', 'www.'))


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
    json_output: Annotated[Optional[str], cyclopts.Parameter(help="Save detailed analysis to JSON file (directory path)")] = None,
    sample_rate: Annotated[int, cyclopts.Parameter(help="Process every Nth frame (default: 2)")] = 2,
    save_frames: Annotated[bool, cyclopts.Parameter(help="Save annotated frames with MediaPipe landmarks to output directory")] = False,
    save_video: Annotated[bool, cyclopts.Parameter(help="Save cut video from start to best timestamp (requires ffmpeg)")] = False,
    top_n: Annotated[Optional[int], cyclopts.Parameter(help="Return top N ranked timestamps in sorted order (shows scores)")] = None,
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

        # Get top 5 with full verbose details
        sceneflow video.mp4 --top-n 5 --verbose

        # With detailed output
        sceneflow video.mp4 --verbose

        # Save analysis to JSON
        sceneflow video.mp4 --json ./output

        # From URL
        sceneflow https://www.youtube.com/watch?v=... --verbose

        # Save annotated frames and cut video
        sceneflow video.mp4 --save-frames --save-video
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

    try:
        # Step 1: Speech Detection
        if verbose:
            print("=" * 60)
            print("SCENEFLOW - Smart Video Cut Point Detection")
            print("=" * 60)
            print(f"\nAnalyzing: {Path(source).name}")
            print(f"\n[1/2] Detecting speech end time using VAD...")

        logger.info("Stage 1/2: Detecting speech end time...")
        detector = SpeechDetector()
        speech_end_time, confidence = detector.get_speech_end_time(
            source,
            return_confidence=True
        )
        logger.info(f"Speech ends at: {speech_end_time:.2f}s (confidence: {confidence:.2f})")

        if verbose:
            print(f"      Speech ends at: {speech_end_time:.2f}s (confidence: {confidence:.2f})")

        # Get video duration
        duration = _get_video_duration(source)
        logger.info(f"Video duration: {duration:.2f}s")

        if verbose:
            print(f"      Video duration: {duration:.2f}s")
            print(f"\n[2/2] Analyzing visual features from {speech_end_time:.2f}s to {duration:.2f}s...")

        # Step 2: Visual Frame Ranking (using default weights)
        logger.info("Stage 2/2: Ranking frames based on visual quality...")
        ranker = CutPointRanker()
        ranked_frames = ranker.rank_frames(
            video_path=source,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_frames=save_frames,
            save_video=save_video
        )

        if not ranked_frames:
            logger.error("No suitable cut points found")
            print("Error: No suitable cut points found", file=sys.stderr)
            sys.exit(1)

        logger.info(f"Successfully analyzed {len(ranked_frames)} frames")

        best_frame = ranked_frames[0]
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
                    print(f"  {i}. {frame.timestamp:.2f}s (frame {frame.frame_index}, score: {frame.score:.4f})")

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
                print(f"  {i}. {frame.timestamp:.2f}s (frame {frame.frame_index}, score: {frame.score:.4f})")

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
                sample_rate=sample_rate
            )

            # Determine how many candidates to include in JSON
            json_candidate_count = min(top_n, len(ranked_frames)) if top_n is not None else 10

            analysis_data = {
                'video_path': str(source),
                'video_name': video_name,
                'duration': duration,
                'speech_detection': {
                    'speech_end_time': speech_end_time,
                    'confidence': confidence,
                    'method': 'Silero VAD'
                },
                'best_cut_point': {
                    'timestamp': best_frame.timestamp,
                    'frame_index': best_frame.frame_index,
                    'score': best_frame.score
                },
                'top_candidates': [
                    {
                        'rank': frame.rank,
                        'timestamp': frame.timestamp,
                        'frame_index': frame.frame_index,
                        'score': frame.score
                    }
                    for frame in ranked_frames[:json_candidate_count]
                ],
                'detailed_scores': [
                    {
                        'frame_index': score.frame_index,
                        'timestamp': score.timestamp,
                        'final_score': score.final_score,
                        'composite_score': score.composite_score,
                        'quality_penalty': score.quality_penalty,
                        'stability_boost': score.stability_boost,
                        'component_scores': {
                            'eye_openness': score.eye_openness_score,
                            'motion_stability': score.motion_stability_score,
                            'expression_neutrality': score.expression_neutrality_score,
                            'pose_stability': score.pose_stability_score,
                            'visual_sharpness': score.visual_sharpness_score
                        }
                    }
                    for score in detailed_scores[:json_candidate_count]
                ],
                'config': {
                    'eye_openness_weight': 0.30,
                    'motion_stability_weight': 0.25,
                    'expression_neutrality_weight': 0.20,
                    'pose_stability_weight': 0.15,
                    'visual_sharpness_weight': 0.10,
                    'sample_rate': sample_rate
                }
            }

            with open(json_path, 'w', encoding='utf-8') as f:
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
