"""
Ranked Results Example - get_ranked_cut_frames_async()

This example shows how to get multiple cut point options asynchronously:
- Get top N best cut points
- Compare multiple options
- Choose the one that works best for your needs
- Non-blocking execution for async applications
"""

import asyncio
import logging
from sceneflow import get_ranked_cut_frames_async

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main():
    video_path = "new_dataset/videos/scene1.mp4"

    n_results = 5

    print("SceneFlow - Ranked Results Async Example")
    print("=" * 60)
    print(f"Finding top {n_results} cut points in: {video_path}")
    print()

    top_cuts = await get_ranked_cut_frames_async(
        video_path, n=n_results, disable_visual_analysis=True
    )

    print()
    print("=" * 60)
    print(f"✓ Top {len(top_cuts)} cut points found:")
    print("=" * 60)
    for i, timestamp in enumerate(top_cuts, 1):
        print(f"  {i}. {timestamp:.2f}s")
    print()
    print("Choose any of these timestamps for your cut!")
    print("The first one is the best, but others are good alternatives.")


if __name__ == "__main__":
    asyncio.run(main())
