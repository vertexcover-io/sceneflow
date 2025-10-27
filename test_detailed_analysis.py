"""
Detailed analysis of top-ranked frames to verify quality.
"""

import cv2
from sceneflow import CutPointRanker, RankingConfig
from sceneflow.extractors import FeatureExtractor
from sceneflow.speech_detector import SpeechDetector


def analyze_top_frames(video_path: str):
    """Analyze top ranked frames in detail."""

    print("="*80)
    print("DETAILED ANALYSIS OF TOP FRAMES")
    print("="*80)

    # Rank frames
    config = RankingConfig()
    ranker = CutPointRanker(config)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    detector = SpeechDetector()
    speech_end_time = detector.get_speech_end_time(video_path)
    print(f"speech end at : {speech_end_time}")

    print("Ranking frames...")
    ranked_frames = ranker.rank_frames(
        video_path=video_path,
        start_time=speech_end_time,
        end_time=duration,
        sample_rate=2,
        save_frames=True,
        save_video=True
    )

    # Now extract features for top 10 to get EAR/MAR
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print()
    print(f"{'Rank':<6} {'Time':<8} {'Score':<10} {'EAR':<10} {'MAR':<10} {'Quality'}")
    print("-"*80)

    for i, frame_result in enumerate(ranked_frames[:15], 1):
        # Extract frame
        frame_num = int(frame_result.timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Get EAR/MAR
        ear = extractor.extract_eye_openness(frame)
        mar = extractor.extract_expression_activity(frame)

        # Determine quality assessment
        quality_issues = []
        if ear < 0.20:
            quality_issues.append("BLINK")
        elif ear > 0.40:
            quality_issues.append("WIDE-EYES")

        if mar > 0.30:
            quality_issues.append("MOUTH-OPEN")

        if not quality_issues:
            quality = "GOOD"
        else:
            quality = " ".join(quality_issues)

        print(f"{i:<6} {frame_result.timestamp:<8.2f} {frame_result.score:<10.4f} "
              f"{ear:<10.4f} {mar:<10.4f} {quality}")

    cap.release()

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    # Count issues in top 5
    cap = cv2.VideoCapture(video_path)
    top_5_issues = 0

    for i, frame_result in enumerate(ranked_frames[:5], 1):
        frame_num = int(frame_result.timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        ear = extractor.extract_eye_openness(frame)
        mar = extractor.extract_expression_activity(frame)

        if ear < 0.20 or ear > 0.40 or mar > 0.30:
            top_5_issues += 1

    cap.release()

    print(f"Top 5 frames with issues: {top_5_issues} / 5")

    if top_5_issues == 0:
        print("✓ SUCCESS: All top 5 frames have good quality!")
    elif top_5_issues <= 1:
        print("~ MOSTLY GOOD: Only 1 frame with issues in top 5")
    else:
        print("✗ NEEDS IMPROVEMENT: Multiple awkward frames in top 5")


if __name__ == "__main__":
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/007_explainer.mp4"
    analyze_top_frames(video_path)
