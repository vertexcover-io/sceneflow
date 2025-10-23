"""
URL Download Example

This example shows how to analyze videos from URLs:
- Automatically downloads videos from direct URLs
- Works with .mp4, .avi, and other direct video file URLs
- Note: YouTube/platform URLs are NOT supported (only direct file URLs)
"""

import logging
from sceneflow import get_cut_frame, get_ranked_cut_frames
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def example_single_url():
    """Example 1: Get best cut point from URL"""
    print("\n" + "=" * 60)
    print("Example 1: Analyze Video from URL")
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
        best_time = get_cut_frame(video_url)

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


def example_ranked_url():
    """Example 2: Get top 5 results from URL"""
    print("\n" + "=" * 60)
    print("Example 2: Get Ranked Results from URL")
    print("=" * 60)

    video_url = "https://example.com/path/to/video.mp4"

    print(f"\nDownloading and analyzing: {video_url}")
    print()

    try:
        top_5 = get_ranked_cut_frames(video_url, n=5)

        print()
        print("=" * 60)
        print(f"✓ Top {len(top_5)} cut points found:")
        print("=" * 60)
        for i, timestamp in enumerate(top_5, 1):
            print(f"  {i}. {timestamp:.2f}s")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    print("SceneFlow - URL Download Examples")
    print("=" * 60)


    print("\n" + "=" * 60)
    print("\nTo run the examples:")
    print("1. Replace the example URLs with real direct video URLs")
    print("2. Uncomment the examples you want to run")
    print()

    example_single_url()
    example_ranked_url()

    print("Update the URLs in the script and uncomment to run!")
