"""
Simple test script to verify InsightFace fixes.
"""

import logging
import cv2
from sceneflow import CutPointRanker, RankingConfig
from sceneflow.extractors import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_ear_mar_on_frames(video_path: str):
    """Test EAR/MAR values on sample frames."""

    print("="*70)
    print("Testing EAR/MAR Calculations")
    print("="*70)
    print(f"Video: {video_path}\n")

    # Initialize extractor
    extractor = FeatureExtractor()

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Test on multiple frames
    test_times = [5.0, 7.0, 9.0, 11.0, 13.0]

    print(f"{'Time':<8} {'EAR':<10} {'MAR':<10} {'Status'}")
    print("-"*70)

    for time_sec in test_times:
        frame_num = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        ear = extractor.extract_eye_openness(frame)
        mar = extractor.extract_expression_activity(frame)

        # Determine status
        status_parts = []
        if ear < 0.20:
            status_parts.append("BLINK")
        elif ear > 0.40:
            status_parts.append("WIDE-EYES")
        else:
            status_parts.append("eyes-OK")

        if mar > 0.30:
            status_parts.append("MOUTH-OPEN")
        elif mar < 0.15:
            status_parts.append("mouth-closed")
        else:
            status_parts.append("mouth-neutral")

        status = " ".join(status_parts)

        print(f"{time_sec:<8.2f} {ear:<10.4f} {mar:<10.4f} {status}")

    cap.release()

    print("\n" + "="*70)
    print("EXPECTED RANGES:")
    print("="*70)
    print("EAR (Eye Aspect Ratio):")
    print("  - 0.20-0.35: Normal open eyes (GOOD)")
    print("  - <0.20: Blinking/closed (BAD)")
    print("  - >0.40: Wide open (BAD)")
    print()
    print("MAR (Mouth Aspect Ratio):")
    print("  - <0.15: Closed mouth (BEST)")
    print("  - 0.15-0.30: Slightly open (OK)")
    print("  - >0.30: Open mouth (BAD)")


def test_ranking(video_path: str):
    """Test the full ranking system."""

    print("\n" + "="*70)
    print("Testing Full Ranking System")
    print("="*70)
    print(f"Video: {video_path}\n")

    config = RankingConfig()
    print("Weights:")
    print(f"  Eye openness: {config.eye_openness_weight}")
    print(f"  Expression neutrality: {config.expression_neutrality_weight}")
    print()

    ranker = CutPointRanker(config)

    print("Ranking frames (5s - 15s)...")
    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=5.0,
        end_time=15.0,
        sample_rate=2,
        save_frames=True,  # Save to output/
        save_video=False
    )

    print()
    print("TOP 5 FRAMES:")
    print(f"{'Rank':<6} {'Time':<8} {'Score':<10}")
    print("-"*40)

    for i, frame in enumerate(ranked_frames[:5], 1):
        print(f"{i:<6} {frame.timestamp:<8.2f} {frame.score:<10.4f}")

    print()
    print("Review annotated frames in output/ folder to verify quality.")
    print("Look for frames without blinking or open mouths in the top results.")


if __name__ == "__main__":
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"

    # Test 1: Check EAR/MAR values
    test_ear_mar_on_frames(video_path)

    # Test 2: Full ranking
    test_ranking(video_path)

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("If EAR and MAR values look reasonable, the fix is working!")
