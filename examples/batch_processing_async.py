"""
Batch Processing Example - Async Version

This example shows how to process multiple videos concurrently:
- Process videos in parallel using asyncio
- Handle errors gracefully for individual videos
- Save annotated frames and cut videos for each
- Much faster than sequential processing
"""

import asyncio
import logging
from pathlib import Path
from sceneflow import cut_video_async

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def process_single_video(video_path: Path, index: int, total: int):
    """Process a single video asynchronously.

    Args:
        video_path: Path to the video file
        index: Current video index (1-based)
        total: Total number of videos

    Returns:
        Dictionary with processing results
    """
    print(f"[{index}/{total}] Processing: {video_path.name}")
    print("-" * 80)

    try:
        # Generate output path for the cut video
        output_path = f"output/{video_path.stem}_cut_async.mp4"

        # Process video and save cut video, frames, and logs
        best_time = await cut_video_async(
            str(video_path),
            output_path,
            sample_rate=1,  # Skip every other frame for faster processing
            save_frames=True,
            save_logs=True,
            use_energy_refinement=True,
            # energy_threshold_db=8.0,
            # use_llm_selection=True,
            # disable_visual_analysis=True,
        )

        print(f"✓ Success! Best cut point: {best_time:.2f}s")
        print(f"  - Annotated frames saved to: output/{video_path.stem}/")
        print(f"  - Cut video saved to: {output_path}")

        return {"video": video_path.name, "cut_time": best_time, "status": "success"}

    except Exception as e:
        print(f"✗ Failed: {e}")
        return {"video": video_path.name, "error": str(e), "status": "failed"}


async def main():
    # Configure the folder containing videos
    video_folder = Path("new_dataset/voice_change_videos")

    # Common video file extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    print("SceneFlow - Batch Processing Async Example")
    print("=" * 80)
    print(f"Processing videos from: {video_folder}")
    print()

    # Find all video files in the folder
    video_files = [
        f for f in video_folder.iterdir() if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {video_folder}")
        return

    print(f"Found {len(video_files)} video file(s)")
    print("=" * 80)
    print("\nProcessing videos concurrently...")
    print()

    # Process all videos concurrently
    tasks = [
        process_single_video(video_path, i, len(video_files))
        for i, video_path in enumerate(video_files, 1)
    ]
    results = await asyncio.gather(*tasks)

    print()

    # Print summary
    print("=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"Total videos processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        print("Successful videos:")
        for r in successful:
            print(f"  ✓ {r['video']}: cut at {r['cut_time']:.2f}s")
        print()

    if failed:
        print("Failed videos:")
        for r in failed:
            print(f"  ✗ {r['video']}: {r['error']}")
        print()

    print("=" * 80)
    print("All outputs saved to the 'output/' directory")
    print("\nNote: Async processing is much faster than sequential processing!")


if __name__ == "__main__":
    asyncio.run(main())
