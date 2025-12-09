"""Internal helper functions for SceneFlow CLI.

This module contains implementation details and should not be imported directly.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from sceneflow.shared.models import RankedFrame, FrameScore, FrameFeatures
from sceneflow.core import CutPointRanker
from sceneflow.detection import SpeechDetector, EnergyRefiner
from sceneflow.selection import LLMFrameSelector

logger = logging.getLogger(__name__)


def print_verbose_header(
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


def detect_speech_end_cli(
    source: str,
    no_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
    verbose: bool
) -> Tuple[float, float, float]:
    """Detect speech end time with CLI-specific logging."""
    if verbose:
        print("\n[1/2] Detecting speech end time using VAD...")

    detector = SpeechDetector()
    vad_speech_end_time, vad_timestamps = detector.get_speech_timestamps(source)

    confidence = 0.0
    if vad_timestamps:
        confidence = 1.0

    if verbose:
        print(f"      Speech ends at: {vad_speech_end_time:.4f}s (confidence: {confidence:.2f})")

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


def rank_frames_cli(
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
    """Rank frames with CLI-specific logging."""
    if verbose:
        print(f"\n[2/2] Analyzing visual features from {speech_end_time:.4f}s to {duration:.4f}s...")

    ranker = CutPointRanker()

    if need_internals:
        ranked_frames, features, scores = ranker.rank_frames(
            video_path=source,
            start_time=speech_end_time,
            end_time=duration,
            sample_rate=sample_rate,
            save_frames=save_frames,
            save_video=False,
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


def apply_llm_selection_cli(
    source: str,
    ranked_frames: List[RankedFrame],
    speech_end_time: float,
    duration: float,
    scores: List[FrameScore],
    features: List[FrameFeatures],
    verbose: bool
) -> RankedFrame:
    """Apply LLM selection with CLI-specific logging."""
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
        )

        if verbose:
            print(
                f"      LLM selected frame at {best_frame.timestamp:.4f}s "
                f"(frame {best_frame.frame_index})"
            )

        logger.info("LLM selected frame at %.4fs", best_frame.timestamp)
        return best_frame

    except Exception as e:
        logger.warning("LLM selection failed: %s, using top algorithmic result", e)
        if verbose:
            print(f"      LLM selection failed: {e}")
            print("      Falling back to top algorithmic result")
        return ranked_frames[0]


def print_results(
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
        n = min(top_n, len(ranked_frames))

        if verbose:
            print(f"\n{'=' * 60}")
            print("RESULTS")
            print(f"{'=' * 60}")
            print(f"\nBest cut point: {best_frame.timestamp:.4f}s")
            print(f"Frame: {best_frame.frame_index}")
            print(f"Score: {best_frame.score:.4f}")
            print(f"\nTop {n} candidates:")
            for i, frame in enumerate(ranked_frames[:n], 1):
                print(
                    f"  {i}. {frame.timestamp:.4f}s "
                    f"(frame {frame.frame_index}, score: {frame.score:.4f})"
                )

            if len(ranked_frames) > n:
                print(f"\n  ... and {len(ranked_frames) - n} more candidates")

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
            for i, frame in enumerate(ranked_frames[:n], 1):
                print(f"{i}. {frame.timestamp:.4f}s (score: {frame.score:.4f})")

    elif verbose:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"\nBest cut point: {best_frame.timestamp:.4f}s")
        print(f"Frame: {best_frame.frame_index}")
        print(f"Score: {best_frame.score:.4f}")
        print("\nTop 3 candidates:")
        for i, frame in enumerate(ranked_frames[:3], 1):
            print(
                f"  {i}. {frame.timestamp:.4f}s "
                f"(frame {frame.frame_index}, score: {frame.score:.4f})"
            )

        if len(ranked_frames) > 3:
            print(f"\n  ... and {len(ranked_frames) - 3} more candidates")

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
        print(f"{best_frame.timestamp:.4f}")


def save_json_output(
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

    detailed_scores = ranker.get_detailed_scores(
        video_path=source,
        start_time=speech_end_time,
        end_time=duration,
        sample_rate=sample_rate
    )

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
