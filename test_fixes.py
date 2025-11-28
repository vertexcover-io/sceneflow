"""
Test script to verify InsightFace fixes for EAR/MAR calculations.

This script:
1. Extracts features from multiple frames
2. Displays EAR and MAR values for top-ranked frames
3. Verifies that awkward frames (blinks, open mouth) are NOT ranked high
"""

import logging
from sceneflow import CutPointRanker, RankingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def test_video_ranking(video_path: str):
    """Test the ranking system with fixed EAR/MAR calculations."""

    print("="*70)
    print("Testing InsightFace Fixes")
    print("="*70)
    print(f"Video: {video_path}")
    print()

    # Initialize ranker with new config
    config = RankingConfig()
    print("Current weights:")
    print(f"  Eye openness: {config.eye_openness_weight:.2f}")
    print(f"  Motion stability: {config.motion_stability_weight:.2f}")
    print(f"  Expression neutrality: {config.expression_neutrality_weight:.2f}")
    print(f"  Pose stability: {config.pose_stability_weight:.2f}")
    print(f"  Visual sharpness: {config.visual_sharpness_weight:.2f}")
    print()

    ranker = CutPointRanker(config)

    # Rank frames - extract features separately to access raw data
    print("Extracting features...")
    features = ranker._extract_features(video_path, 5.0, 15.0, sample_rate=2)

    print("Ranking frames...")
    from sceneflow.scorer import FrameScorer
    scorer = FrameScorer(config)
    frame_scores = scorer.compute_scores(features)

    # Create ranked results
    from sceneflow.models import RankedFrame
    ranked_frames_temp = []
    for score in frame_scores:
        ranked_frames_temp.append({
            'frame_index': score.frame_index,
            'timestamp': score.timestamp,
            'score': score.final_score
        })

    # Sort by score
    ranked_frames_temp.sort(key=lambda x: x['score'], reverse=True)

    # Create RankedFrame objects with rank
    ranked_frames = []
    for rank, item in enumerate(ranked_frames_temp, 1):
        ranked_frames.append(RankedFrame(
            rank=rank,
            frame_index=item['frame_index'],
            timestamp=item['timestamp'],
            score=item['score']
        ))

    # Save annotated frames
    if True:  # save_frames
        print("Saving annotated frames...")
        ranker._save_annotated_frames(video_path, features, ranked_frames, 5.0, 2)

    print()
    print("="*70)
    print("TOP 10 RANKED FRAMES")
    print("="*70)
    print(f"{'Rank':<6} {'Time':<8} {'Score':<8} {'EAR':<8} {'MAR':<8} {'Motion':<8} {'Status'}")
    print("-"*70)

    for i, frame_result in enumerate(ranked_frames[:10], 1):
        # Get feature data for this frame
        frame_feat = None
        for feat in features:
            if abs(feat.timestamp - frame_result.timestamp) < 0.01:
                frame_feat = feat
                break

        if frame_feat:
            ear = frame_feat.eye_openness
            mar = frame_feat.expression_activity
            motion = frame_feat.motion_magnitude

            # Determine status
            status_parts = []
            if ear < 0.20:
                status_parts.append("BLINK")
            elif ear > 0.40:
                status_parts.append("WIDE-EYES")

            if mar > 0.30:
                status_parts.append("MOUTH-OPEN")

            status = " ".join(status_parts) if status_parts else "GOOD"

            print(f"{i:<6} {frame_result.timestamp:<8.2f} {frame_result.score:<8.4f} "
                  f"{ear:<8.4f} {mar:<8.4f} {motion:<8.2f} {status}")
        else:
            print(f"{i:<6} {frame_result.timestamp:<8.2f} {frame_result.score:<8.4f} "
                  f"{'N/A':<8} {'N/A':<8} {'N/A':<8} N/A")

    print()
    print("="*70)
    print("VALIDATION")
    print("="*70)

    # Check top 5 frames
    top_5 = ranked_frames[:5]
    issues = []

    for i, frame_result in enumerate(top_5, 1):
        frame_feat = None
        for feat in features:
            if abs(feat.timestamp - frame_result.timestamp) < 0.01:
                frame_feat = feat
                break

        if frame_feat:
            ear = frame_feat.eye_openness
            mar = frame_feat.expression_activity

            if ear < 0.20:
                issues.append(f"Frame #{i} (t={frame_result.timestamp:.2f}s): BLINKING (EAR={ear:.3f})")
            if mar > 0.30:
                issues.append(f"Frame #{i} (t={frame_result.timestamp:.2f}s): MOUTH OPEN (MAR={mar:.3f})")

    if issues:
        print("WARNING: Found awkward frames in top 5:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("The fixes may not be working correctly. Check annotated frames in output/")
    else:
        print("SUCCESS: Top 5 frames all have good quality!")
        print("  - No blinking (EAR >= 0.20)")
        print("  - No open mouths (MAR <= 0.30)")

    print()
    print("Annotated frames saved to: output/")
    print("Review the top frames to verify quality visually.")


if __name__ == "__main__":
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"
    test_video_ranking(video_path)
