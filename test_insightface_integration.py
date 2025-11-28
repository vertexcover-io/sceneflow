"""
Quick integration test for InsightFace in SceneFlow pipeline.
Tests that all new features can be extracted from a sample frame.
"""

import cv2
import numpy as np
from sceneflow.extractors import FeatureExtractor
from sceneflow.models import FrameFeatures
from sceneflow.config import RankingConfig
from sceneflow.scorer import FrameScorer

def test_extractor():
    """Test that FeatureExtractor can extract all features."""
    print("Testing FeatureExtractor...")

    # Create a simple test frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    extractor = FeatureExtractor()

    # Test all extraction methods
    eye_openness = extractor.extract_eye_openness(test_frame)
    print(f"  Eye openness: {eye_openness:.3f}")

    motion = extractor.extract_motion_magnitude(test_frame)
    print(f"  Motion: {motion:.3f}")

    expression = extractor.extract_expression_activity(test_frame)
    print(f"  Expression: {expression:.3f}")

    pose = extractor.extract_pose_deviation(test_frame)
    print(f"  Pose deviation: {pose:.3f}")

    sharpness = extractor.extract_visual_sharpness(test_frame)
    print(f"  Sharpness: {sharpness:.3f}")

    eyebrow = extractor.extract_eyebrow_position(test_frame)
    print(f"  Eyebrow position: {eyebrow:.3f}")

    jaw = extractor.extract_jaw_openness(test_frame)
    print(f"  Jaw openness: {jaw:.3f}")

    print("  All extraction methods working!")
    return True

def test_feature_model():
    """Test that FrameFeatures can be created with all fields."""
    print("\nTesting FrameFeatures model...")

    features = FrameFeatures(
        frame_index=0,
        timestamp=0.0,
        eye_openness=0.3,
        motion_magnitude=0.5,
        expression_activity=0.2,
        pose_deviation=0.1,
        sharpness=100.0,
        eyebrow_position=0.08,
        jaw_openness=20.0
    )

    print(f"  Created FrameFeatures: {features.frame_index}, timestamp={features.timestamp}")
    print(f"  All fields present: eyebrow={features.eyebrow_position}, jaw={features.jaw_openness}")
    return True

def test_scorer():
    """Test that FrameScorer can score features with new weights."""
    print("\nTesting FrameScorer...")

    config = RankingConfig()
    config.validate()
    print(f"  Config validated (weights sum to 1.0)")

    scorer = FrameScorer(config)

    # Create sample features
    features = [
        FrameFeatures(
            frame_index=i,
            timestamp=i * 0.033,
            eye_openness=0.3,
            motion_magnitude=0.5,
            expression_activity=0.2,
            pose_deviation=0.1,
            sharpness=100.0,
            eyebrow_position=0.08,
            jaw_openness=20.0
        )
        for i in range(10)
    ]

    scores = scorer.compute_scores(features)
    print(f"  Scored {len(scores)} frames")

    if scores:
        score = scores[0]
        print(f"  Sample score breakdown:")
        print(f"    Eye openness: {score.eye_openness_score:.3f}")
        print(f"    Motion: {score.motion_stability_score:.3f}")
        print(f"    Expression: {score.expression_neutrality_score:.3f}")
        print(f"    Pose: {score.pose_stability_score:.3f}")
        print(f"    Sharpness: {score.visual_sharpness_score:.3f}")
        print(f"    Eyebrow: {score.eyebrow_stability_score:.3f}")
        print(f"    Jaw: {score.jaw_stability_score:.3f}")
        print(f"    Final score: {score.final_score:.3f}")

    return True

def main():
    print("=" * 60)
    print("InsightFace Integration Test")
    print("=" * 60)

    try:
        test_extractor()
        test_feature_model()
        test_scorer()

        print("\n" + "=" * 60)
        print("SUCCESS: All integration tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
