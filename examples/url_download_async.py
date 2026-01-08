"""
URL Download Example - Async Version

This example shows how to analyze videos from URLs asynchronously:
- Automatically downloads videos from direct URLs
- Works with .mp4, .avi, and other direct video file URLs
- Non-blocking execution for async applications
- Note: YouTube/platform URLs are NOT supported (only direct file URLs)
"""

import asyncio
import logging
from sceneflow import get_cut_frames_async

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def example_single_url():
    """Example 1: Get best cut point from URL (async)"""
    print("\n" + "=" * 60)
    print("Example 1: Analyze Video from URL (Async)")
    print("=" * 60)

    # Direct video URL (must be a direct link to .mp4, .avi, etc.)
    video_url = "https://example.com/path/to/video.mp4"

    print(f"\nDownloading and analyzing: {video_url}")
    print("Note: This only works with direct video file URLs")
    print()

    try:
        # SceneFlow will automatically:
        # 1. Download the video to a temporary directory
        # 2. Detect speech end time
        # 3. Analyze frames
        # 4. Return the best cut point
        # 5. Clean up the downloaded video after processing
        best_time = (await get_cut_frames_async(video_url))[0]

        print()
        print("=" * 60)
        print(f"✓ Best cut point: {best_time:.2f}s")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. The URL is a direct link to a video file (.mp4, .avi, etc.)")
        print("  2. The URL is publicly accessible")
        print("  3. You have internet connection")


async def example_ranked_url():
    """Example 2: Get top 5 results from URL (async)"""
    print("\n" + "=" * 60)
    print("Example 2: Get Ranked Results from URL (Async)")
    print("=" * 60)

    video_url = "https://example.com/path/to/video.mp4"

    print(f"\nDownloading and analyzing: {video_url}")
    print()

    try:
        top_5 = await get_cut_frames_async(video_url, n=5)

        print()
        print("=" * 60)
        print(f"✓ Top {len(top_5)} cut points found:")
        print("=" * 60)
        for i, timestamp in enumerate(top_5, 1):
            print(f"  {i}. {timestamp:.2f}s")

    except Exception as e:
        print(f"\n✗ Error: {e}")


async def main():
    print("SceneFlow - URL Download Async Examples")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("\nTo run the examples:")
    print("1. Replace the example URLs with real direct video URLs")
    print("2. Uncomment the examples you want to run")
    print()

    await example_single_url()
    await example_ranked_url()

    print("\nUpdate the URLs in the script and uncomment to run!")


if __name__ == "__main__":
    asyncio.run(main())
