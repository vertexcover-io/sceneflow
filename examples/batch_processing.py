"""
Batch Processing Example - Process Multiple Videos

This example shows how to:
- Iterate over all video files in a folder
- Process each video with get_cut_frame()
- Save annotated frames and cut videos for each
- Handle errors gracefully for individual videos
"""

import logging
from pathlib import Path
from sceneflow import get_cut_frame, RankingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    # Configure the folder containing videos
    video_folder = Path("D:/vertexcover/ai-video-cutter/dataset/AI")

    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

    # Optional: Customize ranking configuration
    # config = RankingConfig(
    #     eye_openness_weight=0.30,
    #     motion_stability_weight=0.25,
    #     expression_neutrality_weight=0.20,
    #     pose_stability_weight=0.15,
    #     visual_sharpness_weight=0.10
    # )

    print("SceneFlow - Batch Processing Example")
    print("=" * 80)
    print(f"Processing videos from: {video_folder}")
    print()

    # Find all video files in the folder
    video_files = [
        f for f in video_folder.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {video_folder}")
        return

    print(f"Found {len(video_files)} video file(s)")
    print("=" * 80)
    print()

    results = []

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 80)

        try:
            # Process video with save_frames and save_video enabled
            best_time = get_cut_frame(
                str(video_path),
                sample_rate=2,  # Skip every other frame for faster processing
                save_frames=True,  # Save annotated frames with landmarks
                save_video=True,    # Save cut video (requires ffmpeg)
                save_logs=True
            )

            results.append({
                'video': video_path.name,
                'cut_time': best_time,
                'status': 'success'
            })

            print(f"✓ Success! Best cut point: {best_time:.2f}s")
            print(f"  - Annotated frames saved to: output/{video_path.stem}/")
            print(f"  - Cut video saved to: output/{video_path.stem}_cut.mp4")

        except Exception as e:
            results.append({
                'video': video_path.name,
                'error': str(e),
                'status': 'failed'
            })

            print(f"✗ Failed: {e}")

        print()

    # Print summary
    print("=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

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


if __name__ == "__main__":
    main()
