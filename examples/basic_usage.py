"""
Basic Usage Example - cut_video()

This example shows how to use SceneFlow to cut a video:
- Finds the best cut point and saves the video
- Saves annotated frames and detailed logs
- Works with local video files
- Demonstrates custom configuration options
"""

import logging
from sceneflow import cut_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def basic_example():
    """
    Basic example: Using custom energy threshold and sample rate.

    Demonstrates:
    - Custom energy threshold for speech end detection
    - Higher sample rate (1) for more precise frame analysis
    - Full frame annotation and logging
    """
    video_path = "new_dataset/voice_change_videos/video1.mp4"
    output_path = "output/demo.mp4"

    print("\nSceneFlow - Advanced Configuration Example")
    print("=" * 60)
    print(f"Finding cut point with custom settings in: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print()

    best_time = cut_video(
        video_path,
        output_path,
        save_frames=True,
        save_logs=True,
        use_energy_refinement=True,
        # use_llm_selection=True,
        # disable_visual_analysis=True,
        energy_threshold_db=10,
        sample_rate=1,
    )

    print()
    print("=" * 60)
    print(f"✓ Cut point found: {best_time:.2f} seconds")
    print(f"✓ Cut video saved to: {output_path}")
    print("=" * 60)
    print("\nCustom settings provide more control over speech detection and frame analysis")


if __name__ == "__main__":
    basic_example()
