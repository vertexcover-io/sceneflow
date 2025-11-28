"""
Analyze Per-Face Data

This script extracts features and displays per-face information to verify
that multi-face detection is working correctly.
"""

import cv2
import logging
from sceneflow import CutPointRanker, RankingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    # Test video path
    video_path = "D:/vertexcover/ai-video-cutter/dataset/AI/002_explainer.mp4"

    print("=" * 70)
    print("Multi-Face Detection - Per-Face Analysis")
    print("=" * 70)
    print()

    # Create ranker
    config = RankingConfig(center_weighting_strength=1.0)
    ranker = CutPointRanker(config)

    # Extract features from a few frames
    print(f"Extracting features from video: {video_path}")
    print("Analyzing frames 3.0s to 5.0s")
    print()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(3.0 * fps)
    end_frame = int(5.0 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ranker.extractor.reset()

    frame_count = 0
    multi_face_count = 0

    print("Frame Analysis:")
    print("-" * 70)

    for frame_idx in range(start_frame, end_frame, 10):  # Sample every 10th frame
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Extract all faces
        num_faces, face_features_list, (eye_openness, expression, pose) = \
            ranker.extractor.extract_all_faces(frame)

        frame_count += 1
        if num_faces > 1:
            multi_face_count += 1

        print(f"\nFrame {frame_idx} @ {timestamp:.2f}s:")
        print(f"  Faces detected: {num_faces}")

        if num_faces > 0:
            print(f"  Aggregated metrics (center-weighted):")
            print(f"    Eye openness: {eye_openness:.3f}")
            print(f"    Expression: {expression:.3f}")
            print(f"    Pose deviation: {pose:.3f}")

            if face_features_list:
                print(f"  Per-face breakdown:")
                for face_feat in face_features_list:
                    print(f"    Face {face_feat.face_index + 1}:")
                    print(f"      Center distance: {face_feat.center_distance:.3f}")
                    print(f"      Center weight: {face_feat.center_weight:.3f}")
                    print(f"      Eye openness: {face_feat.eye_openness:.3f}")
                    print(f"      Expression: {face_feat.expression_activity:.3f}")
                    print(f"      Pose deviation: {face_feat.pose_deviation:.3f}")
                    print(f"      BBox: ({face_feat.bbox[0]:.0f}, {face_feat.bbox[1]:.0f}, "
                          f"{face_feat.bbox[2]:.0f}, {face_feat.bbox[3]:.0f})")
        else:
            print("  No faces detected (using default penalty values)")

    cap.release()

    print()
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total frames analyzed: {frame_count}")
    print(f"  Frames with multiple faces: {multi_face_count}")
    print(f"  Percentage with multi-face: {(multi_face_count/frame_count*100) if frame_count > 0 else 0:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
