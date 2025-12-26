"""CLI formatting and output utilities for SceneFlow."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from sceneflow.shared.models import RankedFrame
from sceneflow.core import CutPointRanker

logger = logging.getLogger(__name__)


def print_verbose_header(
    source: str,
    use_llm_selection: bool,
    no_energy_refinement: bool,
    energy_threshold_db: float,
    energy_lookback_frames: int,
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


def print_results(
    best_frame: RankedFrame,
    ranked_frames: List[RankedFrame],
    top_n: Optional[int],
    verbose: bool,
    save_frames: bool,
    output: Optional[str],
    source: str,
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

            if save_frames or output:
                print(f"\n{'=' * 60}")
                print("OUTPUT FILES")
                print(f"{'=' * 60}")
                video_name = Path(source).stem
                if save_frames:
                    print(f"Annotated frames: output/{video_name}/")
                if output:
                    print(f"Cut video: {output}")
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

        if save_frames or output:
            print(f"\n{'=' * 60}")
            print("OUTPUT FILES")
            print(f"{'=' * 60}")
            video_name = Path(source).stem
            if save_frames:
                print(f"Annotated frames: output/{video_name}/")
            if output:
                print(f"Cut video: {output}")
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
    verbose: bool,
) -> None:
    """Save detailed analysis to JSON file."""
    json_dir = Path(json_output)
    json_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(source).stem
    json_path = json_dir / f"{video_name}_analysis.json"

    detailed_scores = ranker.get_detailed_scores(
        video_path=source, start_time=speech_end_time, end_time=duration, sample_rate=sample_rate
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
