"""
Basic Usage Example - cut_video_async()

This example shows how to use SceneFlow async API to cut a video:
- Async API for use in async applications
- Non-blocking execution (event loop never blocks)
- Same configuration options as sync version
- Perfect for web servers, bots, and concurrent processing
"""

import asyncio
import logging
from sceneflow import cut_video_async

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def basic_example_async():
    """
    Async example: Non-blocking video processing.

    Demonstrates:
    - Async API for use in async applications
    - Non-blocking execution (event loop never blocks)
    - Same configuration options as sync version
    - Perfect for web servers, bots, and concurrent processing
    """
    video_path = "new_dataset/voice_change_videos/video1.mp4"
    output_path = "output/demo_async.mp4"

    print("\nSceneFlow - Async Example")
    print("=" * 60)
    print(f"Finding cut point (async) in: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print()

    best_time = await cut_video_async(
        video_path,
        output_path,
        save_frames=True,
        save_logs=True,
        use_energy_refinement=True,
        energy_threshold_db=10,
        sample_rate=1,
    )

    print()
    print("=" * 60)
    print(f"✓ Cut point found: {best_time:.2f} seconds")
    print(f"✓ Cut video saved to: {output_path}")
    print("=" * 60)
    print("\nAsync API allows non-blocking execution in async applications")


if __name__ == "__main__":
    asyncio.run(basic_example_async())
