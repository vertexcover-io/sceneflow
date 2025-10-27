"""
Ranked Results Example - get_ranked_cut_frames()

This example shows how to get multiple cut point options:
- Get top N best cut points
- Compare multiple options
- Choose the one that works best for your needs
"""

import logging
from sceneflow import get_ranked_cut_frames

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"

    n_results = 5

    print("SceneFlow - Ranked Results Example")
    print("=" * 60)
    print(f"Finding top {n_results} cut points in: {video_path}")
    print()
    top_cuts = get_ranked_cut_frames(video_path, n=n_results)

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
    main()
