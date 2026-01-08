"""
Custom Configuration Example - Async Version

This example shows how to customize the ranking algorithm asynchronously:
- Adjust scoring weights for different priorities
- Control processing speed with sample_rate
- Save outputs (frames, cut video)
- Non-blocking execution for async applications
"""

import asyncio
import logging
from sceneflow import (
    get_cut_frame_async,
    get_ranked_cut_frames_async,
    cut_video_async,
    RankingConfig,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def example_custom_weights():
    """Example 1: Custom scoring weights (async)"""
    print("\n" + "=" * 60)
    print("Example 1: Custom Scoring Weights (Async)")
    print("=" * 60)

    video_path = "your_video.mp4"

    config = RankingConfig(
        eye_openness_weight=0.40,  # 40% weight on eyes (default: 30%)
        motion_stability_weight=0.30,  # 30% weight on motion (default: 25%)
        expression_neutrality_weight=0.15,  # 15% weight on expression (default: 20%)
        pose_stability_weight=0.10,  # 10% weight on pose (default: 15%)
        visual_sharpness_weight=0.05,  # 5% weight on sharpness (default: 10%)
    )
    # Note: Weights must sum to 1.0

    print("\nUsing custom weights (prioritizing eyes and motion)...")
    best_time = await get_cut_frame_async(video_path, config=config)
    print(f"\n✓ Best cut point: {best_time:.2f}s")


async def example_performance_tuning():
    """Example 2: Performance tuning with sample_rate (async)"""
    print("\n" + "=" * 60)
    print("Example 2: Performance Tuning (Async)")
    print("=" * 60)

    video_path = "your_video.mp4"

    # sample_rate controls how many frames to process
    # sample_rate=1: Process every frame (slowest, most accurate)
    # sample_rate=2: Process every 2nd frame (default, good balance)
    # sample_rate=3: Process every 3rd frame (faster, still good quality)

    print("\nUsing sample_rate=3 for faster processing...")
    best_time = await get_cut_frame_async(video_path, sample_rate=3)
    print(f"\n✓ Best cut point: {best_time:.2f}s")


async def example_save_outputs():
    """Example 3: Save annotated frames and cut video (async)"""
    print("\n" + "=" * 60)
    print("Example 3: Save Outputs (Async)")
    print("=" * 60)

    video_path = "your_video.mp4"
    output_path = "output/your_video_cut_async.mp4"

    print("\nAnalyzing video and saving outputs...")
    print("- Saving annotated frames with facial landmarks")
    print("- Saving cut video (requires ffmpeg)")

    best_time = await cut_video_async(
        video_path,
        output_path,
        save_frames=True,  # Save frames with InsightFace landmarks
    )

    print(f"\n✓ Best cut point: {best_time:.2f}s")
    print("\nOutputs saved:")
    print("  - Annotated frames: output/<video_name>/")
    print(f"  - Cut video: {output_path}")


async def example_combined():
    """Example 4: Combine custom config with ranked results (async)"""
    print("\n" + "=" * 60)
    print("Example 4: Combined - Custom Config + Ranked Results (Async)")
    print("=" * 60)

    video_path = "your_video.mp4"

    config = RankingConfig(
        eye_openness_weight=0.35,
        motion_stability_weight=0.35,
        expression_neutrality_weight=0.15,
        pose_stability_weight=0.10,
        visual_sharpness_weight=0.05,
    )

    print("\nGetting top 3 cut points with custom weights...")
    top_3 = await get_ranked_cut_frames_async(video_path, n=3, config=config, sample_rate=2)

    print("\n✓ Top 3 cut points:")
    for i, timestamp in enumerate(top_3, 1):
        print(f"  {i}. {timestamp:.2f}s")


async def main():
    print("SceneFlow - Custom Configuration Async Examples")
    print("=" * 60)
    print("\nThis script demonstrates various async configuration options.")
    print("Uncomment the examples you want to run.\n")

    # Uncomment the examples you want to run:

    # await example_custom_weights()
    # await example_performance_tuning()
    # await example_save_outputs()
    # await example_combined()

    print("\nUncomment examples in the script to run them!")


if __name__ == "__main__":
    asyncio.run(main())
