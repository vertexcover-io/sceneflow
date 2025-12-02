"""
Basic Usage Example - cut_video()

This example shows how to use SceneFlow to cut a video:
- Finds the best cut point and saves the video
- Saves annotated frames and detailed logs
- Works with local video files
"""

import logging
from sceneflow import cut_video

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    video_path = "new_dataset/videos/scene4.mp4"
    output_path = "output/scene4_cut.mp4"

    print("SceneFlow - Basic Usage Example")
    print("=" * 60)
    print(f"Finding best cut point in: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print()
    best_time = cut_video(
        video_path,
        output_path,
        save_frames=True,
        save_logs=True,
        use_energy_refinement=True,
        use_llm_selection=True
    )
    print()
    print("=" * 60)
    print(f"✓ Best cut point: {best_time:.2f} seconds")
    print(f"✓ Cut video saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
