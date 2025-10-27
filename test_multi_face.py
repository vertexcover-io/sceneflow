"""
Test Multi-Face Detection

This script tests the multi-face detection and center-weighted scoring implementation.
"""

import logging
from sceneflow import CutPointRanker, RankingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    # Test video path - replace with your test video
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"

    print("=" * 70)
    print("SceneFlow - Multi-Face Detection Test")
    print("=" * 70)
    print()

    # Create config with custom multi-face parameters
    config = RankingConfig(
        center_weighting_strength=1.0,  # Default: faces closer to center have more weight
        min_face_confidence=0.5         # Minimum confidence threshold
    )

    # Initialize ranker
    ranker = CutPointRanker(config)

    # Test on a 5-second window
    print("Testing multi-face detection on video...")
    print(f"Video: {video_path}")
    print(f"Analyzing frames from 3.0s to 8.0s")
    print()

    # Get detailed scores to see per-frame face detection
    detailed_scores = ranker.get_detailed_scores(
        video_path=video_path,
        start_time=3.0,
        end_time=8.0,
        sample_rate=5  # Sample every 5th frame for faster testing
    )

    print(f"Analyzed {len(detailed_scores)} frames")
    print()
    print("Top 5 frames:")
    print("-" * 70)

    for i, score in enumerate(detailed_scores[:5], 1):
        print(f"{i}. Frame {score.frame_index} @ {score.timestamp:.2f}s")
        print(f"   Final Score: {score.final_score:.4f}")
        print(f"   - Eye Openness: {score.eye_openness_score:.3f}")
        print(f"   - Motion Stability: {score.motion_stability_score:.3f}")
        print(f"   - Expression Neutrality: {score.expression_neutrality_score:.3f}")
        print(f"   - Pose Stability: {score.pose_stability_score:.3f}")
        print(f"   - Visual Sharpness: {score.visual_sharpness_score:.3f}")
        print()

    print("=" * 70)
    print("Testing with save_frames=True to visualize multi-face detection...")
    print("=" * 70)

    # Now test with visualization
    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=3.0,
        end_time=8.0,
        sample_rate=5,
        save_frames=True,  # Save annotated frames
        save_video=False
    )

    print()
    print("SUCCESS: Saved annotated frames to output/ directory")
    print("Check the frames to see all detected faces with landmarks")
    print()
    print("Best cut point:")
    print(f"  Time: {ranked_frames[0].timestamp:.2f}s")
    print(f"  Score: {ranked_frames[0].score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
