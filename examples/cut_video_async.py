"""
Cut Video Example - Async Version

This example shows how to use the cut_video_async() function:
- Automatically finds the best cut point and saves the cut video
- Works with local files and URLs
- Non-blocking execution for async applications
- Optional LLM selection for improved accuracy
- Customizable speech detection and visual analysis
"""

import asyncio
import logging
from sceneflow import cut_video_async

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def example_basic_async(video_path: str, output_path: str):
    """Example: Basic async usage - cut video at optimal point"""
    print("\n" + "=" * 60)
    print("Example: Async Cut Video")
    print("=" * 60)

    print(f"\nAnalyzing and cutting: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print()

    best_time = await cut_video_async(
        video_path,
        output_path,
    )

    print()
    print("=" * 60)
    print(f"✓ Video cut at: {best_time:.2f}s")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)


async def main():
    print("SceneFlow - cut_video_async() Example")
    print("=" * 60)
    print("\nThis script demonstrates the async cut_video() function.")
    print("Replace 'your_video.mp4' with your actual video path.")
    print()

    video_path = "https://cco-public.s3.us-west-1.amazonaws.com/cmirs9amy0000lnq5njqv1lb2%2F65e65f58-3021-46eb-9564-33dfcc2abc52%2F7d0cc00d-903a-4c64-a5cf-07a1077fc743%2Fvideo%2Fvoice_change_0e00fadb.mp4"
    output_path = "output/cut_video_async.mp4"

    await example_basic_async(video_path, output_path)

    print("\nAsync execution allows non-blocking video processing!")


if __name__ == "__main__":
    asyncio.run(main())
