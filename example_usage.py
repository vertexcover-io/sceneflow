from cut_point_ranker import CutPointRanker, RankingConfig
import os
from pathlib import Path


def basic_usage():
    """Basic usage with default configuration"""
    ranker = CutPointRanker()

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/018_hook.mp4"
    start_time = 7.0
    end_time = 8.0

    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        sample_rate=2
    )

    print("Top 5 Cut Point Candidates:")
    for frame in ranked_frames[:5]:
        print(f"Rank {frame.rank}: Frame {frame.frame_index} at {frame.timestamp:.2f}s (score: {frame.score:.4f})")


def detailed_analysis():
    """Get detailed scoring breakdown for debugging"""
    ranker = CutPointRanker()

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/011_explainer.mp4"
    start_time = 5.292
    end_time = 8.0

    detailed_scores = ranker.get_detailed_scores(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        sample_rate=2
    )

    print("\nDetailed Score Breakdown (Top 3):")
    for i, score in enumerate(detailed_scores[:3], 1):
        print(f"\n=== Rank {i}: Frame {score.frame_index} at {score.timestamp:.2f}s ===")
        print(f"Final Score: {score.final_score:.4f}")
        print(f"Composite Score: {score.composite_score:.4f}")
        print(f"Quality Penalty: {score.quality_penalty:.4f}")
        print(f"Stability Boost: {score.stability_boost:.4f}")
        print(f"\nComponent Scores:")
        print(f"  - Eye Openness: {score.eye_openness_score:.4f}")
        print(f"  - Motion Stability: {score.motion_stability_score:.4f}")
        print(f"  - Expression Neutrality: {score.expression_neutrality_score:.4f}")
        print(f"  - Pose Stability: {score.pose_stability_score:.4f}")
        print(f"  - Visual Sharpness: {score.visual_sharpness_score:.4f}")


def custom_weights():
    """Use custom weighting and multi-stage parameters"""
    config = RankingConfig(
        eye_openness_weight=0.40,
        motion_stability_weight=0.30,
        expression_neutrality_weight=0.15,
        pose_stability_weight=0.10,
        visual_sharpness_weight=0.05,
        context_window_size=7,
        quality_gate_percentile=80.0,
        local_stability_window=7
    )

    ranker = CutPointRanker(config)

    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/003_explainer.mp4"
    start_time = 5.151
    end_time = 8.0

    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time
    )

    best_frame = ranked_frames[0]
    print(f"\nBest cut point: Frame {best_frame.frame_index} at {best_frame.timestamp:.2f}s")


def batch_usage(video_dir, start_time=5.0, end_time=8.0, sample_rate=2):
    """Process all videos in a directory

    Args:
        video_dir: Directory containing video files
        start_time: Start time for analysis (default: 5.0s)
        end_time: End time for analysis (default: 8.0s)
        sample_rate: Frame sampling rate (default: 2)
    """
    ranker = CutPointRanker()

    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    # Get all video files from directory
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory '{video_dir}' does not exist")
        return

    video_files = [f for f in video_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in '{video_dir}'")
        return

    print(f"Found {len(video_files)} video(s) to process\n")

    results = {}

    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 60)

        try:
            ranked_frames = ranker.rank_frames(
                video_path=str(video_path),
                start_time=start_time,
                end_time=end_time,
                sample_rate=sample_rate,
                save_video=True,
            )

            if ranked_frames:
                best_frame = ranked_frames[0]
                print(f"  Best cut point: Frame {best_frame.frame_index} at {best_frame.timestamp:.2f}s (score: {best_frame.score:.4f})")
                print(f"  Top 3 candidates:")
                for frame in ranked_frames[:3]:
                    print(f"    Rank {frame.rank}: Frame {frame.frame_index} at {frame.timestamp:.2f}s (score: {frame.score:.4f})")

                results[video_path.name] = {
                    'best_frame': best_frame.frame_index,
                    'best_timestamp': best_frame.timestamp,
                    'best_score': best_frame.score,
                    'top_frames': ranked_frames[:5]
                }
            else:
                print(f"  No frames found in the specified time range")
                results[video_path.name] = None

        except Exception as e:
            print(f"  Error processing video: {str(e)}")
            results[video_path.name] = None

        print()

    # Summary
    print("=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    successful = sum(1 for v in results.values() if v is not None)
    print(f"Successfully processed: {successful}/{len(video_files)} videos")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CUT POINT RANKER - EXAMPLE USAGE")
    print("=" * 60)

    # print("\n1. Basic Usage")
    # print("-" * 60)
    # basic_usage()

    # print("\n2. Detailed Analysis")
    # print("-" * 60)
    # detailed_analysis()

    # print("\n3. Custom Weights")
    # print("-" * 60)
    # custom_weights()

    print("\n4. Batch Processing")
    print("-" * 60)
    batch_usage(
        video_dir="D:/vertexcover/ai-video-cutter/dataset/AI",
        start_time=6.0,
        end_time=8.0,
        sample_rate=2
    )
