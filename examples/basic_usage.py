"""
Basic Usage Example - get_cut_frame()

This example shows the simplest way to use SceneFlow:
- Get the single best cut point from a video
- Works with local video files
"""

import logging
from sceneflow import get_cut_frame

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    video_path = "new_dataset/scene4.mp4"

    print("SceneFlow - Basic Usage Example")
    print("=" * 60)
    print(f"Finding best cut point in: {video_path}")
    print()
    best_time = get_cut_frame(video_path, save_frames=True, save_logs=True, output="output/scene4_cut.mp4", use_energy_refinement=True, use_llm_selection=True)
    print()
    print("=" * 60)
    print(f"✓ Best cut point: {best_time:.2f} seconds")
    print("=" * 60)
    print()
    print("You can now cut your video at this timestamp!")


if __name__ == "__main__":
    main()
